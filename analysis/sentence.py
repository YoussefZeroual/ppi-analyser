# analysis/sentence.py
import time
import logging
import concurrent.futures
from ppi_analyser.analysis.prompts import get_prompt_type

logger = logging.getLogger(__name__)


def process_sentences_batch(
    expression: str,
    forme_relevee: list[str],
    conversations: list[str],
    locuteurs: list[str],
    interlocuteurs: list[list[str]],
    state,
    models: list[str],
    mode: str,
) -> list[list[str]]:

    from ppi_analyser.analysis.prompts import get_prompts_batch

    start = time.time()
    n_sents = len(conversations)

    NON_IA = [0, 1, 5]
    n = len(models)
    models_resolved = []
    submodels_resolved = []

    for i in range(15):
        if i not in NON_IA:
            model_str = models[i % n]
            sep = model_str.find("_")
            models_resolved.append(model_str[:sep])
            submodels_resolved.append(model_str[sep + 1:])
        else:
            models_resolved.append("no_model")
            submodels_resolved.append("no_model")

    state.model_list = models_resolved
    state.submodel_list = submodels_resolved

    system_prompts, batched_prompts = get_prompts_batch(
        expression=expression,
        forme_relevee_list=forme_relevee,
        conv_list=conversations,
        locuteur_list=locuteurs,
        interlocuteur_list=interlocuteurs,
        mode=mode,
    )

    futures = []
    results_per_property = [None] * len(system_prompts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=state.n_threads) as executor:
        for i, (model, submodel, system_prompt, batched_prompt) in enumerate(
            zip(models_resolved, submodels_resolved, system_prompts, batched_prompts)
        ):
            future = executor.submit(
                _call_model_batch,
                system_prompt=system_prompt,
                prompt=batched_prompt,
                model=model,
                submodel=submodel,
                conversations=conversations,
                expression=expression,
                forme_relevee_list=forme_relevee,
                state=state,
                mode=mode,
                n_sentences=n_sents,
            )
            futures.append((i, future))

    for i, future in futures:
        results_per_property[i] = future.result()

    # transpose: results_per_property[prop][sent] -> results_per_sentence[sent][prop]
    results_per_sentence = [[] for _ in range(n_sents)]
    for prop_results in results_per_property:
        if prop_results is None:
            prop_results = [None] * n_sents
        for j in range(n_sents):
            val = prop_results[j] if j < len(prop_results) else None
            results_per_sentence[j].append(val)

    elapsed = time.time() - start
    per_sent_time = elapsed / n_sents if n_sents > 0 else 0
    for _ in range(n_sents):
        state.individual_conv_time.append(per_sent_time)

    logger.info(
        "Batch of %d sentences processed in %.2fs (%.2fs/sentence)",
        n_sents, elapsed, per_sent_time
    )

    return results_per_sentence


def _call_model_batch(
    system_prompt: str,
    prompt: str,
    model: str,
    submodel: str,
    conversations: list[str],
    expression: str,
    forme_relevee_list: list[str],
    state,
    mode: str,
    n_sentences: int,
) -> list[str]:

    if model == "no_model":
        return _handle_no_model_batch(
            system_prompt, conversations, expression,
            forme_relevee_list, state, mode, n_sentences
        )

    from ppi_analyser.models.factory import get_provider
    provider = get_provider(model, submodel, state)
    raw_response = provider.complete(system_prompt, prompt)

    with state._token_lock:
        state.total_tokens_in  += (len(system_prompt) + len(prompt)) // 4
        state.total_tokens_out += len(raw_response) // 4

    return _parse_batch_response(raw_response, n_sentences)


def _handle_no_model_batch(
    system_prompt: str,
    conversations: list[str],
    expression: str,
    forme_relevee_list: list[str],
    state,
    mode: str,
    n_sentences: int,
) -> list[str]:

    import json
    from ppi_analyser.analysis.position import get_pos

    prompt_type = get_prompt_type(system_prompt)
    results = []

    for i in range(n_sentences):
        if prompt_type == "Forme":
            val = json.dumps({
                "Propriété": forme_relevee_list[i] if i < len(forme_relevee_list) else expression,
                "Justification": "Forme relevée dans l'échange analysé"
            })
        elif prompt_type == "Lemme":
            val = json.dumps({
                "Propriété": expression,
                "Justification": "Forme choisie par défaut pour représenter la PPI analysée"
            })
        elif prompt_type == "Position":
            result = get_pos(
                conversations[i], mode,
                tokenization_mode=state.tokenization_mode,
                nlp=state.nlp,
            )
            if result:
                val = json.dumps({"Propriété": result[0], "Justification": result[1]})
            else:
                val = json.dumps({"Propriété": "Indéterminé", "Justification": "Position non calculée"})
        else:
            val = json.dumps({"Propriété": "no_model", "Justification": "no_model"})

        results.append(val)

    return results


def _parse_batch_response(raw_response: str, n_sentences: int) -> list[str]:

    import json
    import re

    cleaned = re.sub(r'^```json\s*', '', raw_response.strip())
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    results = [None] * n_sentences

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            for key, val in parsed.items():
                match = re.search(r'(\d+)', str(key))
                if match:
                    idx = int(match.group(1))
                    if idx < n_sentences:
                        results[idx] = json.dumps(val)

        elif isinstance(parsed, list):
            for i, val in enumerate(parsed):
                if i < n_sentences:
                    results[i] = json.dumps(val) if isinstance(val, dict) else str(val)

    except json.JSONDecodeError:
        objects = re.findall(r'\{[^{}]+\}', cleaned)
        for i, obj in enumerate(objects):
            if i < n_sentences:
                try:
                    json.loads(obj)
                    results[i] = obj
                except json.JSONDecodeError:
                    pass

    return results


def process_sentence(
    expression: str,
    forme_relevee: str,
    conversation: str,
    locuteur: str,
    interlocuteurs: list[str],
    sent_index: int,
    state,
    models: list[str],
    mode: str,
) -> list[str]:

    from ppi_analyser.analysis.prompts import get_prompts
    start = time.time()

    system_prompts, prompts = get_prompts(
        expression, forme_relevee, conversation,
        locuteur, interlocuteurs, mode
    )

    results = _run_parallel(
        system_prompts=system_prompts,
        prompts=prompts,
        models=models,
        conversation=conversation,
        expression=expression,
        forme_relevee=forme_relevee,
        sent_index=sent_index,
        state=state,
        mode=mode,
    )

    elapsed = time.time() - start
    state.individual_conv_time.append(elapsed)
    logger.info("Sentence %d processed in %.2fs", sent_index, elapsed)

    return results


def _run_parallel(
    system_prompts: list[str],
    prompts: list[str],
    models: list[str],
    conversation: str,
    expression: str,
    forme_relevee: str,
    sent_index: int,
    state,
    mode: str,
) -> list[str]:

    NON_IA = [0, 1, 5]
    n = len(models)

    resolved_models = []
    resolved_submodels = []
    for i in range(len(system_prompts)):
        if i not in NON_IA:
            model_str = models[i % n]
            sep = model_str.find("_")
            resolved_models.append(model_str[:sep])
            resolved_submodels.append(model_str[sep + 1:])
        else:
            resolved_models.append("no_model")
            resolved_submodels.append("no_model")

    state.model_list = resolved_models
    state.submodel_list = resolved_submodels

    futures = []
    results = [None] * len(system_prompts)
    logger.info("Exécution en parallel de %i modèles: %s", len(state.model_list), state.model_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=state.n_threads) as executor:
        for i, (model, submodel, system_prompt, prompt) in enumerate(
            zip(resolved_models, resolved_submodels, system_prompts, prompts)
        ):
            future = executor.submit(
                _call_model,
                system_prompt=system_prompt,
                prompt=prompt,
                model=model,
                submodel=submodel,
                conversation=conversation,
                expression=expression,
                forme_relevee=forme_relevee,
                sent_index=sent_index,
                state=state,
                mode=mode,
            )
            futures.append((i, future))

    for i, future in futures:
        results[i] = future.result()

    return results


def _call_model(
    system_prompt: str,
    prompt: str,
    model: str,
    submodel: str,
    conversation: str,
    expression: str,
    forme_relevee: str,
    sent_index: int,
    state,
    mode: str,
) -> str:

    from ppi_analyser.models.factory import get_provider
    provider = get_provider(model, submodel, state)
    prompt_type = get_prompt_type(system_prompt)
    logger.info("Traitement de la propriété %s par le modèle %s_%s", prompt_type, model, submodel)

    if model == "no_model":
        return provider.complete(
            system_prompt, prompt,
            expression=expression,
            forme_relevee=forme_relevee,
            conversation=conversation,
        )

    result = provider.complete(system_prompt, prompt)

    with state._token_lock:
        state.total_tokens_in  += (len(system_prompt) + len(prompt)) // 4
        state.total_tokens_out += len(result) // 4

    return result
