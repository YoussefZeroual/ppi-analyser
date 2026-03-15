from ppi_analyser.config import (
    MISTRAL_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY,
    GEMINI_API_KEY, OLLAMA_HOST
)
from ppi_analyser.exceptions import ProviderNotFoundError
from .base import LLMProvider
from .ollama import OllamaProvider
from .mistral import MistralProvider
from .groq import GroqProvider
from .deepseek import DeepseekProvider
from .gemini import GeminiProvider
from .dummy import DummyProvider
from .no_model import NoModelProvider

# models/factory.py — just the function, keep your existing imports

def get_provider(model: str, submodel: str, state=None) -> LLMProvider:
    ollama_host = state.ollama_host if state else OLLAMA_HOST
    
    match model:
        case "mistral":
            return MistralProvider(submodel, MISTRAL_API_KEY)
        case "ollama":
            return OllamaProvider(submodel, ollama_host)
        case "groq":
            return GroqProvider(submodel, GROQ_API_KEY)
        case "deepseek":
            return DeepseekProvider(submodel, DEEPSEEK_API_KEY)
        case "gemini":
            return GeminiProvider(submodel, GEMINI_API_KEY)
        case "no_model":
            return NoModelProvider(submodel,state=state)
        case "dummy":
            return DummyProvider(submodel)
        case _:
            raise ProviderNotFoundError(f"Unknown model provider: '{model}'")
