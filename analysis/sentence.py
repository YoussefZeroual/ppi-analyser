# analysis/sentence.py

import time
import logging
import concurrent.futures
from ppi_analyser.analysis.prompts import get_prompt_type
logger = logging.getLogger(__name__)


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

    NON_IA = [0, 1, 5]  # Forme, Lemme, Position — handled automatically
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
    logger.info("Exécution en parallel de %i modèles: %s",len(state.model_list),state.model_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=state.n_threads) as executor:
        for i, (model, submodel, system_prompt, prompt) in enumerate(
            zip(resolved_models, resolved_submodels, system_prompts, prompts)
        ):
            conv_copy = conversation  # explicit copy before submit
            future = executor.submit(
            lambda sp=system_prompt, p=prompt, m=model, sm=submodel,
            	c=conv_copy, e=expression, f=forme_relevee,
            	si=sent_index, st=state, mo=mode: _call_model(
            	system_prompt=sp, prompt=p, model=m, submodel=sm,
            	conversation=c, expression=e, forme_relevee=f,
            	sent_index=si, state=st, mode=mo
         	   )
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
    logger.info("Traitement de la propriété %s par le modèle %s_%s",prompt_type,model,submodel)
    if model == "no_model":
        return provider.complete(
            system_prompt, prompt,
            expression=expression,
            forme_relevee=forme_relevee,
            conversation=conversation
        )
    return provider.complete(system_prompt, prompt)
