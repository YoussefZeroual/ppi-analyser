# analysis/sentence.py
import re
import time
import logging
import concurrent.futures
from ppi_analyser.analysis.prompts import get_prompt_type
from ppi_analyser.config import predefined_responses
ignore_response = predefined_responses.IGNORED_PROPRETY_RESPONSE
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
    properties: list[str] | None = None,
    start_offset: int = 0
) -> list[list[str]]:

    from ppi_analyser.analysis.prompts import get_prompts_batch
    from ppi_analyser.config import AnalysisMode
    start = time.time()
    n_sents = len(conversations)

    if mode == AnalysisMode.ORAL:
    	NON_IA = [0,1,5]
    else:
    	NON_IA = [0, 1, 5, 7,8]
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
            prompt_type = get_prompt_type(system_prompt)
            if properties and prompt_type not in properties:
                continue
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
                start_offset=start_offset
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
    start_offset: int = 0
) -> list[str]:

    if model == "no_model":
        return _handle_no_model_batch(
            system_prompt, conversations, expression,
            forme_relevee_list, state, mode, n_sentences,start_offset
        )
    from ppi_analyser.models.factory import get_provider
    prompt_type = get_prompt_type(system_prompt)

    if (state.custom_properties_list is not None) and (prompt_type not in state.custom_properties_list):
    	return ignore_response

    # Check analysis cache — batch responses are keyed on the concatenated prompt
    if getattr(state, 'use_analysis_cache', False):
        from ppi_analyser.analysis.analysis_cache import get as acache_get, set as acache_set
        cache_key_conv = prompt  # user prompt contains all conversations
        cached = acache_get(cache_key_conv, "", model, submodel, prompt_type)
        if cached is not None:
            return _parse_batch_response(cached, n_sentences)

    provider = get_provider(model, submodel, state)
    logger.info("traitement de la propriété %s par le modèle %s",prompt_type,submodel)
    raw_response = provider.complete(system_prompt, prompt)

    with state._token_lock:
        state.total_tokens_in  += (len(system_prompt) + len(prompt)) // 4
        state.total_tokens_out += len(raw_response) // 4

    if getattr(state, 'use_analysis_cache', False):
        acache_set(cache_key_conv, "", model, submodel, prompt_type, raw_response)

    return _parse_batch_response(raw_response, n_sentences)


def _handle_no_model_batch(
    system_prompt: str,
    conversations: list[str],
    expression: str,
    forme_relevee_list: list[str],
    state,
    mode: str,
    n_sentences: int,
    start_offset: int = 0
) -> list[str]:
    import json
    from ppi_analyser.analysis.position import get_pos
    from ppi_analyser.analysis.expansion import detect_expansion, extract_ppi_sentence
    prompt_type = get_prompt_type(system_prompt)
    results = []

    #len_nlp_object = len(state.nlp_preprocessed_turn)
        

    for i in range(n_sentences):
        if prompt_type == "Forme":
            val = json.dumps({
                "Propriété": forme_relevee_list[i] if i < len(forme_relevee_list) else expression,
                "Justification": "Forme relevée dans l'échange analysé"
            }, ensure_ascii=False)
        elif prompt_type == "Lemme":
            val = json.dumps({
                "Propriété": expression,
                "Justification": "Forme choisie par défaut pour représenter la PPI analysée"
            }, ensure_ascii=False)
        elif prompt_type == "Position":
            logger.info("traitement de la propriété %s par le modèle %s (conv %s/%s)",prompt_type,"TAL (Stanza)",i,n_sentences)                    
            result = get_pos(
                conversations[i], mode,
                tokenization_mode=state.tokenization_mode,
                nlp=state.nlp,
                state=state,
                sent_id = start_offset#i+start_offset
            )
            if result:
                val = json.dumps({"Propriété": result[0], "Justification": result[1]}, ensure_ascii=False)
            else:
                val = json.dumps({"Propriété": "Indéterminé", "Justification": "Position non calculée"}, ensure_ascii=False)
        elif prompt_type == "Expansions":
            logger.info("traitement de la propriété %s par le modèle %s",prompt_type,"TAL (Stanza)")
            from ppi_analyser.analysis.expansion import detect_expansion
            conv = conversations[i]
            ppi_text, _ = extract_ppi_sentence(conv)
            if ppi_text and state.nlp is not None:
                result = detect_expansion(state.nlp_preprocessed_turn[i+start_offset]["full_turn_nlp_doc"], ppi_text,state.nlp_preprocessed_turn[i+start_offset]["ppi_occurrence"])
                expansion_text = " ".join(w.text for w in result[0]["tokens"]) if result[0]["tokens"] else ""
            else:
                if state.nlp is None:
                    logger.debug("_handle_no_model_batch: no nlp object, skipping expansion detection")
                expansion_text = ""
            if expansion_text:
                val = json.dumps({
                    "Propriété": f"<EXP>{expansion_text}</EXP>",
                    "Justification": f"Expansion syntaxique de '{ppi_text}' détectée par analyse des dépendances"
                }, ensure_ascii=False)
            else:
                val = json.dumps({
                    "Propriété": "Aucune expansion détectée",
                    "Justification": "Aucune expansion syntaxique détectée par analyse des dépendances"
                }, ensure_ascii=False)
        elif prompt_type == "Modifieurs":
            logger.info("traitement de la propriété %s par le modèle %s",prompt_type,"TAL (Stanza)")
            from ppi_analyser.analysis.modifiers import find_modifier, format_modifiers
            if state.nlp is not None:
            	
                labels, subtrees = find_modifier(
                    state.nlp_preprocessed_turn[i+start_offset]["forme_nlp_doc"],
                    state.nlp_preprocessed_turn[i+start_offset]["expression_nlp_doc"],
                    state.nlp_preprocessed_turn[i+start_offset]["full_turn_stripped_nlp_doc"],
                    state.nlp,
                    state.nlp_preprocessed_turn[i+start_offset]["ppi_occurrence"]
                )
                result_str = format_modifiers(labels, subtrees)
                val = json.dumps({
                    "Propriété": result_str,
                    "Justification": "Modifieurs détectés par analyse des dépendances"
                }, ensure_ascii=False)
            else:
                logger.debug("_handle_no_model_batch: no nlp object, skipping modifier detection")
                val = json.dumps({
                    "Propriété": "Aucun modifieur",
                    "Justification": "Aucun modifieur détecté"
                }, ensure_ascii=False)
        else:
            val = json.dumps({"Propriété": "no_model", "Justification": "no_model"}, ensure_ascii=False)
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
                        results[idx] = json.dumps(val, ensure_ascii=False)

        elif isinstance(parsed, list):
            for i, val in enumerate(parsed):
                if i < n_sentences:
                    results[i] = json.dumps(val, ensure_ascii=False) if isinstance(val, dict) else str(val)

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
    properties: list[str] | None = None,
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
    properties: list[str] | None = None,
) -> list[str]:

    NON_IA = state.no_ia#[0, 1, 5, 7,8]
    
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
            prompt_type = get_prompt_type(system_prompt)
            if properties and prompt_type not in properties:
                continue
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
    prompt_type = get_prompt_type(system_prompt)
    logger.info("Traitement de la propriété %s par le modèle %s",prompt_type,model)
    if (state.custom_properties_list is not None) and (prompt_type not in state.custom_properties_list):
    	return ignore_response
    # Check analysis cache
    if getattr(state, 'use_analysis_cache', False):
        from ppi_analyser.analysis.analysis_cache import get as acache_get, set as acache_set
        cached = acache_get(conversation, expression, model, submodel, prompt_type)
        if cached is not None:
            return cached

    result = provider.complete(system_prompt, prompt)

    with state._token_lock:
        state.total_tokens_in  += (len(system_prompt) + len(prompt)) // 4
        state.total_tokens_out += len(result) // 4

    if getattr(state, 'use_analysis_cache', False):
        acache_set(conversation, expression, model, submodel, prompt_type, result)

    return result
