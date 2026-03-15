from .base import LLMProvider

DUMMY_RESPONSE = '```json\n{"Propriété": "dummy for test purpose", "Justification": "dummy for test purpose"}\n```'

class DummyProvider(LLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return DUMMY_RESPONSE
