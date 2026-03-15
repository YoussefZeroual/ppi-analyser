# analysis/prompts.py

import re
import logging

logger = logging.getLogger(__name__)
GENERAL_PROMPT_BATCH = """
Tu es un expert en analyse linguistique. Tu analyses des phrases préfabriquées d'interactions (PPI) françaises.
Tu dois analyser TOUTES les occurrences fournies et retourner UNE SEULE réponse JSON.
- Attention: les balises <PPI> </PPI> indiquent l'expression à traiter
- RÉPONDS UNIQUEMENT avec un objet JSON valide
- FORMAT REQUIS: {"sentence no.0": {"Propriété": "valeur", "Justification": "valeur"}, "sentence no.1": {...}, ...}
- INTERDIT: markdown, texte avant/après le JSON, caractère ':' dans les valeurs
- CHAQUE JUSTIFICATION NE DOIT PAS DÉPASSER DEUX LIGNES
- L'ORDRE DOIT CORRESPONDRE À L'ORDRE DES PHRASES FOURNIES
"""


def get_prompts_batch(
    expression: str,
    forme_relevee_list: list[str],
    conv_list: list[str],
    locuteur_list: list[str],
    interlocuteur_list: list[list[str]],
    mode: str,
) -> tuple[list[str], list[str]]:

    from ppi_analyser.preprocessing.speakers import get_loc_full_turn

    raw_system_prompts = load_system_prompts()
    raw_system_prompts = [GENERAL_PROMPT_BATCH + p for p in raw_system_prompts]

    batched_prompts = []

    for sp in raw_system_prompts:
        prompt_type_match = re.findall(r'Prompt_(\w+)', sp)
        if not prompt_type_match:
            continue
        prompt_type = prompt_type_match[0]

        sentence_blocks = []
        for i, (conv, loc, interloc, forme) in enumerate(
            zip(conv_list, locuteur_list, interlocuteur_list, forme_relevee_list)
        ):
            tour_parole = get_loc_full_turn(conv, "écrit_ia")[0]

            if prompt_type in ["Acception", "Portee", "Declenchement",
                               "Fonction_globale", "Fonction_specifique", "Remarques_diverses"]:
                block = f"""
--- Sentence no.{i} ---
Locuteur: {loc}
Interlocuteurs: {interloc}
Conversation: {conv}
Expression: {expression}
"""
            elif prompt_type in ["type_phrase", "Modalite", "Particularites_syntaxiques",
                                 "Expansions", "Modifieurs", "Coocurrents"]:
                block = f"""
--- Sentence no.{i} ---
Locuteur: {loc}
Tour de parole: {tour_parole}
Forme relevée: {forme}
Lemme: {expression}
"""
            else:
                block = f"""
--- Sentence no.{i} ---
Expression: {expression}
"""
            sentence_blocks.append(block)

        batched_prompt = "\n".join(sentence_blocks) + f"""
Réponds avec un JSON contenant une entrée par phrase:
{{"sentence no.0": {{"Propriété": "...", "Justification": "..."}}, "sentence no.1": {{...}}, ...}}
"""
        batched_prompts.append(batched_prompt)

    return raw_system_prompts, batched_prompts
GENERAL_PROMPT = """
Tu es un expert en analyse linguistique. Tu examines des phrases préfabriquées d'intéractions (PPI) françaises dans leur contexte conversationnel.
Tu dois donner des réponses qui concernent l'expression et la conversation fournies.
Réponds avec un style simple et claire.
- Attention: les balises <PPI> </PPI> ont été ajoutées pour t'indiquer l'expression à traiter, elles ne font pas partie du texte original.
INSTRUCTION FORMAT ABSOLU :
- ANALYSE L'OCCURRENCE DE L'EXPRESSION QUI EST À L'INTÉRIEUR DES BALISES <PPI>
- Réponds UNIQUEMENT au format JSON VALIDE avec EXACTEMENT ces 2 clés : "Propriété" et "Justification"
- Format REQUIS : {"Propriété": "valeur", "Justification": "valeur"}
- INTERDIT : aucun caractère ':' dans les valeurs, pas de markdown, pas de texte avant/après
- IMPORTANT : ton analyse doit porter sur l'expression fournie dans ce contexte particulier.
- REPONDS D'UNE MANIERE CLAIRE, CONCISE, PERTINENTE ET RAPIDE
- LA JUSTIFICATION NE DOIT PAS DÉPASSER DEUX LIGNES.
"""


def load_prompts(prompt_file: str = "prompts.txt") -> list[str]:
    START_MARKER = "start_prompt"
    END_MARKER = "end_prompt"
    try:
        with open(prompt_file, "r") as f:
            content = f.read()
        if not content:
            return []

        prompts = []
        start_idx = 0
        while True:
            start_pos = content.find(START_MARKER, start_idx)
            if start_pos == -1:
                break
            end_pos = content.find(END_MARKER, start_pos + len(START_MARKER))
            if end_pos == -1:
                break
            prompt_text = content[start_pos + len(START_MARKER):end_pos].strip()
            prompts.append(prompt_text)
            start_idx = end_pos + len(END_MARKER)
        return prompts

    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", prompt_file)
        return []


def load_system_prompts(prompt_file: str = "system_prompts.txt") -> list[str]:
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()

        prompts = []
        for match in re.finditer(r'start_prompt(.*?)end_prompt', content, re.DOTALL):
            cleaned = match.group(1).strip()
            if cleaned:
                prompts.append(cleaned)

        logger.info("Loaded %d system prompts from %s", len(prompts), prompt_file)
        return prompts

    except FileNotFoundError:
        logger.warning("system_prompts.txt not found")
        return []
    except Exception as e:
        logger.error("Error loading system prompts: %s", e)
        return []


def get_prompts(
    expression: str,
    forme_relevee: str,
    conv: str | list,
    locuteur: str | list,
    interlocuteur: list,
    mode: str,
    batch: bool = False,
) -> tuple[list[str], list[str]]:

    from ppi_analyser.preprocessing.speakers import get_loc_full_turn

    raw_system_prompts = load_system_prompts()
    raw_system_prompts = [GENERAL_PROMPT + p for p in raw_system_prompts]

    if not batch:
        tour_parole = get_loc_full_turn(conv, "écrit_ia")

    regular_prompts = []

    for sp in raw_system_prompts:
        prompt_type_match = re.findall(r'Prompt_(\w+)', sp)
        if not prompt_type_match:
            continue
        prompt_type = prompt_type_match[0]

        template_conversation = f"""
Analyse de la propriété: {prompt_type}
**Contexte de la conversation** :
- **Locuteur** : {locuteur}
- **Interlocuteurs** : {interlocuteur}
- **Conversation** : {conv}
**Expression à analyser** : **{expression}**
"""
        template_tour_parole = f"""
Analyse de la propriété: {prompt_type}
**Contexte de la conversation** :
- **Locuteur** : {locuteur}
- **Tour de parole** : {tour_parole if not batch else ''}
**Expression à analyser** : **{forme_relevee}** (forme relevée)
**Lemme** : **{expression}** (forme par défaut)
"""
        template_minimal = f"""
Analyse de la propriété: {prompt_type}
**Expression à analyser** : **{expression}**
"""
        if prompt_type in ["Acception", "Portee", "Declenchement", "Fonction_globale", "Fonction_specifique", "Remarques_diverses"]:
            regular_prompts.append(template_conversation)
        elif prompt_type in ["type_phrase", "Modalite", "Particularites_syntaxiques", "Expansions", "Modifieurs", "Coocurrents"]:
            regular_prompts.append(template_tour_parole)
        else:
            regular_prompts.append(template_minimal)

    return raw_system_prompts, regular_prompts
def get_prompt_type(system_prompt: str) -> str | None:
    match = re.search(r'Prompt_(\w+)', system_prompt)
    return match.group(1) if match else None
