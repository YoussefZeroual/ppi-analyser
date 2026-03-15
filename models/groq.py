from .base import LLMProvider

class GroqProvider(LLMProvider):
    def __init__(self, submodel: str, api_key: str):
        self.submodel = submodel
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.submodel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=1,
            max_completion_tokens=8192,
            stream=True,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
