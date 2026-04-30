from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Non-streaming: wait for full response, return as string."""
        ...

    @abstractmethod
    def generate_stream(self, prompt: str):
        """Streaming: yield one token at a time for SSE."""
        ...


def get_llm() -> BaseLLM:
    """
    Factory — reads LLM_PROVIDER from settings.
    Returns the correct adapter so the rest of the pipeline
    never needs to know which LLM is active.
    """
    from app.config import settings
    if settings.LLM_PROVIDER == "openai":
        from app.llm.openai_llm import OpenAILLM
        return OpenAILLM()
    else:
        from app.llm.ollama import OllamaLLM
        return OllamaLLM()
