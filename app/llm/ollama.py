import json
import httpx
from app.config import settings
from app.llm.base import BaseLLM


class OllamaLLM(BaseLLM):

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"]

    def generate_stream(self, prompt: str):
        """Yields one token at a time for SSE streaming."""
        with httpx.stream(
            "POST",
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("response"):
                        yield token
