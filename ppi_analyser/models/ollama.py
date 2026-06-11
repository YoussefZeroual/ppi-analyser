import importlib
from .base import LLMProvider
from ppi_analyser.exceptions import OllamaConnectionError

class OllamaProvider(LLMProvider):
    def __init__(self, submodel: str, host: str = "http://localhost:11434"):
        self.submodel = submodel
        self.host = host

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ollama = importlib.import_module("ollama")
        try:
            client = ollama.Client(host=self.host)
            response = client.chat(
                model=self.submodel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response["message"]["content"]
        except ConnectionError as e:
            raise OllamaConnectionError("Could not reach Ollama server") from e
