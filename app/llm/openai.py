from openai import OpenAI
from app.config import settings
from app.llm.base import BaseLLM

class OpenAILLM(BaseLLM):

    def __init__(self):
        self.client = OpenAI(api_key=settigns.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def generate(self, prompt: str) -> str:
        """Non-streaming — waits for full response before returning."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{ "role": "user", "content": prompt }],
            temperature=0.2 # low temperature = more factual, less creative
        )
        return response.choices[0].message.content
    
    def generate_stream(self, prompt: str):
        """
        Streaming — yields one token at a time.
        Used by /ask-stream endpoint.
        The for loop in event_stream() calls this generator.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True   # tells OpenAI to stream tokens back
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:   # first and last chunks may have None content
                yield token