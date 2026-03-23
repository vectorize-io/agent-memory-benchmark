import asyncio
import time

from .base import ResponseMode
from ..memory.base import MemoryProvider
from ..models import AnswerResult


class AgentMode(ResponseMode):
    name = "agent"
    description = "Bypasses the benchmark retrieval pipeline entirely — calls the provider's own native direct_answer() for providers that have built-in agentic answering."

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id, meta=meta))

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        t0 = time.perf_counter()
        answer, context, raw = await memory.async_direct_answer(query, user_id=user_id, query_timestamp=query_timestamp)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        return AnswerResult(
            answer=answer,
            reasoning="",
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response=raw,
        )

    def answer_from_context(self, query: str, context: str, task_type: str = "open") -> AnswerResult:
        raise NotImplementedError("agent mode does not support --skip-retrieval")
