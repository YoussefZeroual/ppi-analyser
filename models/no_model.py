# models/no_model.py

import re
import json
import logging
from ppi_analyser.models.base import LLMProvider
from ppi_analyser.analysis.prompts import get_prompt_type
logger = logging.getLogger(__name__)


class NoModelProvider(LLMProvider):

    def __init__(self, submodel: str = "no_model", state=None):
        self.submodel = submodel
        self.nlp = state.nlp if state else None
        self.tokenization_mode = state.tokenization_mode if state else "simple"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        expression: str = "",
        forme_relevee: str = "",
        conversation: str = "",
        mode: str = "oral",
    ) -> str:
        prompt_type = get_prompt_type(system_prompt)
        if prompt_type == "Forme":
            return f'{{"Propriété": "{forme_relevee}", "Justification": "Forme relevée dans l\'échange analysé"}}'

        if prompt_type == "Lemme":
            return f'{{"Propriété": "{expression}", "Justification": "Forme choisie par défaut pour représenter la PPI analysée"}}'

        if prompt_type == "Position":
            from ppi_analyser.analysis.position import get_pos
            result = get_pos(conversation, mode, nlp=self.nlp)
            if result:
                return f'{{"Propriété": "{result[0]}", "Justification": "{result[1]}"}}'
            return '{"Propriété": "Indéterminé", "Justification": "Position non calculée"}'

        if prompt_type == "Expansions":
            from ppi_analyser.analysis.position import (
                _extract_ppi_text,
                _get_ppi_ids_stanza,
                _get_expansion_tokens_stanza,
            )
            ppi_text = _extract_ppi_text(conversation)
            expansion_text = ""
            if ppi_text and self.nlp is not None:
                doc = self.nlp(re.sub(r'</?PPI>', '', conversation, flags=re.IGNORECASE).lower())
                for sentence in doc.sentences:
                    ppi_ids = _get_ppi_ids_stanza(sentence, ppi_text)
                    if ppi_ids:
                        exp_tokens = _get_expansion_tokens_stanza(sentence, ppi_ids)
                        expansion_text = " ".join(
                            w.text for w in exp_tokens if w.upos != "PUNCT"
                        )
                        break
            if expansion_text:
                return json.dumps({
                    "Propriété": expansion_text,
                    "Justification": f"Expansion syntaxique de '{ppi_text}' détectée par analyse des dépendances"
                }, ensure_ascii=False)
            return json.dumps({
                "Propriété": "Aucune expansion détectée",
                "Justification": "Aucune expansion syntaxique détectée par analyse des dépendances"
            }, ensure_ascii=False)
        if prompt_type == "Modifieurs":
            from ppi_analyser.analysis.position import _extract_ppi_text
            from ppi_analyser.analysis.modifiers import find_modifier, format_modifiers
            ppi_text = _extract_ppi_text(conversation)
            text_clean = re.sub(r'</?PPI>', '', conversation, flags=re.IGNORECASE)
            text_clean = re.sub(r'\[.*?\]', '', text_clean).strip()
            ppi_line = next((l for l in text_clean.split('\n') if ppi_text and ppi_text.lower() in l.lower()), text_clean)
            text_clean = re.split(r'[;,.!?…/]', ppi_line)[0].strip()
            if ppi_text and self.nlp is not None:
                labels, subtrees = find_modifier(ppi_text, expression, text_clean, self.nlp)
                result_str = format_modifiers(labels, subtrees)
                return json.dumps({
            "Propriété": result_str,
            "Justification": f"Modifieurs de '{ppi_text}' détectés par analyse des dépendances"
                }, ensure_ascii=False)
        else:
                if self.nlp is None:
                    logger.debug("NoModelProvider: no nlp object, skipping modifier detection")
                    return json.dumps({
                "Propriété": "Aucun modifieur",
                "Justification": "Aucun modifieur détecté"
                    }, ensure_ascii=False)
        return '{"Propriété": "no_model", "Justification": "no_model"}'


