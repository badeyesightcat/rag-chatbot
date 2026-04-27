# Abstraction layer — lets you swap LLMs without changing any pipeline code
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_stream(self, prompt: str): ... # generator for SSE streaming

def get_llm() -> BaseLLM:
    """Factory — reads LLM_PROVIDER from env."""
    from app.config import settings
    if settings.LLM_PROVIDER == "openai":
        from app.llm.openai import OpenAILLM
        return OpenAILLM()
    else:
        from app.llm.ollama import OllamaLLM
        return OllamaLLM()