# models/no_model.py

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
            result = get_pos(conversation, mode,nlp=self.nlp)
            if result:
                return f'{{"Propriété": "{result[0]}", "Justification": "{result[1]}"}}'
            return '{"Propriété": "Indéterminé", "Justification": "Position non calculée"}'

        return '{"Propriété": "no_model", "Justification": "no_model"}'


