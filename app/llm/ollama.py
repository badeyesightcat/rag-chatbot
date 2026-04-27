import httpx
from app.config import settings
from app.llm.base import BaseLLM
import json

class OllamaLLM(BaseLLM):

    def generate(self, prompt):
        r = httpx.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        return r.json()["response"]
    
    def generate_stream(self, prompt):
        # yields token by token for SSE streaming
        with httpx.stream(
            "POST",
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=120) as r:
            for line in r.iter_lines():
                data = json.loads(line)
                if token := data.get("response"):
                    yield token