import os

from .base import LLM, Schema
from .gemini import GeminiLLM
from .groq import GroqLLM
from .openai import OpenAILLM

REGISTRY: dict[str, type[LLM]] = {
    "gemini": GeminiLLM,
    "groq": GroqLLM,
    "openai": OpenAILLM,
}


def get_llm(name: str = "gemini") -> LLM:
    if name not in REGISTRY:
        raise ValueError(f"Unknown LLM: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()


def get_answer_llm() -> LLM:
    """Return the LLM to use for RAG answer generation.
    Configured via OMB_ANSWER_LLM (provider) and OMB_ANSWER_MODEL (optional model override)."""
    provider = os.environ.get("OMB_ANSWER_LLM", "groq")
    model = os.environ.get("OMB_ANSWER_MODEL")
    cls = REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown OMB_ANSWER_LLM: '{provider}'. Available: {list(REGISTRY)}")
    return cls(model) if model else cls()


def get_judge_llm() -> LLM:
    """Return the LLM to use for evaluation/judging.
    Configured via OMB_JUDGE_LLM (provider) and OMB_JUDGE_MODEL (optional model override)."""
    provider = os.environ.get("OMB_JUDGE_LLM", "gemini")
    model = os.environ.get("OMB_JUDGE_MODEL")
    cls = REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown OMB_JUDGE_LLM: '{provider}'. Available: {list(REGISTRY)}")
    return cls(model) if model else cls()
