from .base import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, submodel: str, api_key: str):
        self.submodel = submodel
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.submodel)
        response = model.generate_content(user_prompt)
        return response.text
