# models/mistral.py

import logging
from ppi_analyser.models.base import LLMProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):

    def __init__(self, submodel: str, api_key: str):
        self.submodel = submodel
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from mistralai.client import Mistral

        client = Mistral(api_key=self.api_key)
        response = client.chat.complete(
            model=self.submodel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content
