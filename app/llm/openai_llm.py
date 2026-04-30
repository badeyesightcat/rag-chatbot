from openai import OpenAI
from app.config import settings
from app.llm.base import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model  = settings.OPENAI_MODEL

    def generate(self, prompt: str) -> str:
        """Non-streaming — waits for full response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str):
        """Yields one token at a time for SSE streaming."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
