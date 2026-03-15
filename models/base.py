# models/base.py

from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        ...
