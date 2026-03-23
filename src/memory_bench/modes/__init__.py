from .base import ResponseMode
from .rag import RAGMode
from .agentic_rag import AgenticRAGMode
from .agent import AgentMode
from ..llm.base import LLM

REGISTRY: dict[str, type[ResponseMode]] = {
    "rag": RAGMode,
    "agentic-rag": AgenticRAGMode,
    "agent": AgentMode,
}


def get_mode(name: str, llm: LLM | None = None) -> ResponseMode:
    if name not in REGISTRY:
        raise ValueError(f"Unknown mode: '{name}'. Available: {list(REGISTRY)}")
    cls = REGISTRY[name]
    if llm is not None and "llm" in cls.__init__.__code__.co_varnames:
        return cls(llm=llm)
    return cls()
