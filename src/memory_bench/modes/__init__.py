from .base import ResponseMode
from .rag import RAGMode
from .agentic_rag import AgenticRAGMode
from .agent import AgentMode
from .coding import CodingMode
from .retrieval import RetrievalMode
from ..llm.base import LLM

REGISTRY: dict[str, type[ResponseMode]] = {
    "rag": RAGMode,
    "agentic-rag": AgenticRAGMode,
    "agent": AgentMode,
    "coding": CodingMode,
    "retrieval": RetrievalMode,
}


def get_mode(name: str, llm: LLM | None = None) -> ResponseMode:
    if name not in REGISTRY:
        raise ValueError(f"Unknown mode: '{name}'. Available: {list(REGISTRY)}")
    cls = REGISTRY[name]
    init_code = getattr(cls.__init__, "__code__", None)  # object.__init__ has none
    if llm is not None and init_code is not None and "llm" in init_code.co_varnames:
        return cls(llm=llm)
    return cls()
