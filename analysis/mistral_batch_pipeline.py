# analysis/mistral_batch_pipeline.py

import json
import logging
from pathlib import Path

from ppi_analyser.analysis.pipeline import PreprocessedSentence, _chunk
from ppi_analyser.analysis.prompts import get_prompts_batch, get_prompt_type
from ppi_analyser.analysis.sentence import _handle_no_model_batch, _parse_batch_response
from ppi_analyser.models.factory import get_mistral_batch_provider
from ppi_analyser.config import PipelineConfig
from ppi_analyser.state import SessionState

logger = logging.getLogger(__name__)

NON_IA = {0, 1, 5, 7,8}  # Forme, Lemme, Position, expansion — handled locally, not submitted to Mistral


def _custom_id(chunk_idx: int, prop_idx: int) -> str:
    return f"c{chunk_idx}_p{prop_idx}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_batch_mistral_async(
    preprocessed: list[PreprocessedSentence],
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
) -> tuple[list[PreprocessedSentence], list[list[str]]]:

    submodel = _resolve_submodel(config.models)
    provider = get_mistral_batch_provider(submodel, config.output_dir)

    # Check for saved job state from a previous interrupted run
    saved = provider.load_job_state()
    if saved:
        logger.info("Resuming Mistral batch jobs from saved state")
        all_results = _poll_and_assemble(saved["job_map"], provider, preprocessed, lemmes, config, state)
        if all_results is None:
            raise InterruptedError("Mistral batch jobs still running. Re-run to resume.")
        provider.clear_job_state()
        return preprocessed, all_results

    # Save preprocessed so we can reassemble on resume
    preprocessed_path = Path(config.output_dir) / "mistral_batch_preprocessed.json"
    _save_preprocessed(preprocessed, lemmes, preprocessed_path)

    chunks       = _chunk(preprocessed, config.batch_size)
    lemme_chunks = _chunk(lemmes, config.batch_size) if lemmes else [None] * len(chunks)

    # Submit one job per (chunk × property), skip NON_IA
    job_map        = {}  # "c{i}_p{j}" -> job_id
    cached_results = {}  # "c{i}_p{j}" -> raw_response (from analysis cache)
    prompt_map     = {}  # "c{i}_p{j}" -> {user, prompt_type} for cache saving

    for chunk_idx, (chunk, lemme_chunk) in enumerate(zip(chunks, lemme_chunks)):
        expression = lemme_chunk[0] if lemme_chunk else config.expression
        if lemme_chunk:
            state.expression_list.extend(lemme_chunk)

        system_prompts, batched_prompts = get_prompts_batch(
            expression=expression,
            forme_relevee_list=[s.forme_relevee for s in chunk],
            conv_list=[s.cleaned for s in chunk],
            locuteur_list=[s.locuteur for s in chunk],
            interlocuteur_list=[s.interlocuteurs for s in chunk],
            mode=config.mode,
        )

        for prop_idx, (system_prompt, batched_prompt) in enumerate(
            zip(system_prompts, batched_prompts)
        ):
            if prop_idx in NON_IA:
                continue
            prompt_type = get_prompt_type(system_prompt)
            if config.properties and prompt_type not in config.properties:
                continue
            if state.custom_properties_list is not None and prompt_type not in state.custom_properties_list: # fixed added custom props
                continue

            cid = _custom_id(chunk_idx, prop_idx)

            # Check analysis cache before submitting
            if state.use_analysis_cache:
                from ppi_analyser.analysis.analysis_cache import get as acache_get
                cached = acache_get(batched_prompt, "", "mistral_batch", submodel, prompt_type)
                if cached is not None:
                    logger.debug("Analysis cache HIT for %s chunk %d prop %d", prompt_type, chunk_idx, prop_idx)
                    cached_results[cid] = cached
                    continue

            job_id = provider.submit([{
                "custom_id": cid,
                "system":    system_prompt,
                "user":      batched_prompt,
            }])
            job_map[cid]    = job_id
            prompt_map[cid] = {"user": batched_prompt, "prompt_type": prompt_type}
            logger.info("Submitted job %s for chunk %d prop %d", job_id, chunk_idx, prop_idx)

    # Poll all jobs
    all_results = _poll_and_assemble(job_map, provider, preprocessed, lemmes, config, state,
                                     preprocessed_path=str(preprocessed_path),
                                     cached_results=cached_results,
                                     submodel=submodel,
                                     prompt_map=prompt_map)
    if all_results is None:
        raise InterruptedError("Mistral batch jobs still running. Re-run to resume.")

    provider.clear_job_state()
    return preprocessed, all_results


# ---------------------------------------------------------------------------
# Poll + assemble
# ---------------------------------------------------------------------------

def _poll_and_assemble(
    job_map: dict,
    provider,
    preprocessed: list[PreprocessedSentence],
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
    preprocessed_path: str | None = None,
    cached_results: dict | None = None,
    submodel: str = "",
    prompt_map: dict | None = None,
) -> list[list[str]] | None:
    """
    Poll all jobs in job_map. Returns assembled results or None if any timed out.
    job_map: {"c{i}_p{j}": job_id}
    """
    raw_results = dict(cached_results) if cached_results else {}
    timed_out   = []

    for cid, job_id in job_map.items():
        result = provider.poll_or_save(job_id, preprocessed_json=preprocessed_path)
        if result is None:
            timed_out.append(cid)
        else:
            raw_response = result.get(cid, "")
            raw_results[cid] = raw_response
            # Save to analysis cache using prompt_map for key info
            if state.use_analysis_cache and raw_response and prompt_map and cid in prompt_map:
                from ppi_analyser.analysis.analysis_cache import set as acache_set
                entry = prompt_map[cid]
                acache_set(entry["user"], "", "mistral_batch", submodel,
                           entry["prompt_type"], raw_response)
                logger.debug("Analysis cache SET for %s", entry["prompt_type"])

    if timed_out:
        combined = {"job_map": job_map, "preprocessed_json": preprocessed_path}
        state_path = Path(config.output_dir) / "mistral_batch_job.json"
        state_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        logger.warning("%d jobs timed out. State saved to %s", len(timed_out), state_path)
        return None

    return _assemble(raw_results, preprocessed, lemmes, config, state)



# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def _assemble(
    raw_results: dict,
    preprocessed: list[PreprocessedSentence],
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
) -> list[list[str]]:
    """
    Reassemble per-sentence results from raw_results.
    raw_results: {"c{chunk}_p{prop}": raw_response_string}
    Returns results_per_sentence[sent][prop].
    """
    chunks       = _chunk(preprocessed, config.batch_size)
    lemme_chunks = _chunk(lemmes, config.batch_size) if lemmes else [None] * len(chunks)
    all_results  = []

    for chunk_idx, (chunk, lemme_chunk) in enumerate(zip(chunks, lemme_chunks)):
        n_sents    = len(chunk)
        expression = lemme_chunk[0] if lemme_chunk else config.expression

        system_prompts, _ = get_prompts_batch(
            expression=expression,
            forme_relevee_list=[s.forme_relevee for s in chunk],
            conv_list=[s.cleaned for s in chunk],
            locuteur_list=[s.locuteur for s in chunk],
            interlocuteur_list=[s.interlocuteurs for s in chunk],
            mode=config.mode,
        )

        n_props              = len(system_prompts)
        results_per_property = []

        for prop_idx, system_prompt in enumerate(system_prompts):
            if prop_idx in NON_IA:
                prop_results = _handle_no_model_batch(
                    system_prompt=system_prompt,
                    conversations=[s.cleaned for s in chunk],
                    expression=expression,
                    forme_relevee_list=[s.forme_relevee for s in chunk],
                    state=state,
                    mode=config.mode,
                    n_sentences=n_sents,
                )
            else:
                cid = _custom_id(chunk_idx, prop_idx)
                prompt_type = get_prompt_type(system_prompt)
                if config.properties and prompt_type not in config.properties:
                    prop_results = [None] * n_sents
                elif state.custom_properties_list is not None and prompt_type not in state.custom_properties_list:
                        prop_results = [None] * n_sents        
                else:
                    raw_response = raw_results.get(cid, "")
                    if not raw_response:
                        similar = [k for k in raw_results if k.startswith(f"c{chunk_idx}_")]
                        logger.warning("No result for %s — keys for this chunk: %s", cid, similar)
                    prop_results = _parse_batch_response(raw_response, n_sents)

            results_per_property.append(prop_results)

        # Transpose: prop × sent -> sent × prop
        for sent_idx in range(n_sents):
            row = []
            for prop_results in results_per_property:
                if prop_results is None:
                    row.append(None)
                else:
                    row.append(
                        prop_results[sent_idx] if sent_idx < len(prop_results) else None
                    )
            all_results.append(row)

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_submodel(models: list[str]) -> str:
    for m in models:
        if m.startswith("mistral_batch_"):
            return m[len("mistral_batch_"):]
    raise ValueError("No mistral_batch_<submodel> entry found in models list")


def _save_preprocessed(preprocessed, lemmes, path: Path) -> None:
    data = {
        "preprocessed": [
            {
                "raw":            s.raw,
                "cleaned":        s.cleaned,
                "locuteur":       s.locuteur,
                "interlocuteurs": s.interlocuteurs,
                "forme_relevee":  s.forme_relevee,
            }
            for s in preprocessed
        ],
        "lemmes": lemmes,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Preprocessed sentences saved to %s", path)


def _load_preprocessed(path: str) -> tuple[list[PreprocessedSentence], list[str] | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preprocessed = [
        PreprocessedSentence(
            raw=s["raw"], cleaned=s["cleaned"], locuteur=s["locuteur"],
            interlocuteurs=s["interlocuteurs"], forme_relevee=s["forme_relevee"],
        )
        for s in data["preprocessed"]
    ]
    return preprocessed, data.get("lemmes")
