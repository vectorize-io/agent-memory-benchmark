import asyncio
from abc import ABC, abstractmethod

from ..memory.base import MemoryProvider
from ..models import AnswerResult


class ResponseMode(ABC):
    name: str
    description: str

    @property
    def llm_id(self) -> str | None:
        """Return the answer-generation LLM identifier, or None if not applicable."""
        return None

    @abstractmethod
    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None) -> AnswerResult:
        """Generate an answer to the query using the memory provider."""
        ...

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        """Async version. Default wraps sync answer in a thread."""
        return await asyncio.to_thread(self.answer, query, memory, task_type, user_id)

    @abstractmethod
    def answer_from_context(self, query: str, context: str, task_type: str = "open") -> AnswerResult:
        """Generate an answer using a pre-retrieved context string (skips retrieval)."""
        ...
