import os
from abc import ABC, abstractmethod
from openai import OpenAI
from prompts import SYSTEM_PROMPT, USER_PROMPT

class LLMProvider(ABC):
    @abstractmethod
    def get_response(self, ocr_text: str, json_schema: dict, model: str) -> str:
        pass

class GroqProvider(LLMProvider):
    """
    LLM provider for Groq-compatible APIs (text only).
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url =  os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_response(self, ocr_text: str, json_schema: dict, model: str) -> str:
        """
        Get the response from the Groq API using OCR text.
        """
        system_prompt = SYSTEM_PROMPT
        user_prompt = f"{USER_PROMPT}\n\n{ocr_text}"

        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content