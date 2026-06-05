from .base import LLMProvider

class DeepseekProvider(LLMProvider):
    def __init__(self, submodel: str, api_key: str):
        self.submodel = submodel
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            model="deepseek-v4-flash",
            stream=False,
        )
        return response.choices[0].message.content
