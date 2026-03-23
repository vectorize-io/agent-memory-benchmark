import asyncio
import time

from .base import ResponseMode
from .rag import RAGMode, _OPEN_SCHEMA, _MCQ_SCHEMA
from ..dataset.base import _DEFAULT_OPEN_PROMPT as _OPEN_PROMPT, _DEFAULT_MCQ_PROMPT as _MCQ_PROMPT
from ..llm.base import ToolDef
from ..llm.gemini import GeminiLLM
from ..memory.base import MemoryProvider
from ..models import AnswerResult

_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a memory recall tool.
Use the `recall` tool to search for relevant information before answering.
You may call `recall` multiple times with different queries to gather enough context.
Once you have sufficient information, provide a thorough answer to the question.

Question: {query}\
"""


class AgenticRAGMode(ResponseMode):
    name = "agentic-rag"
    description = "The LLM acts as an agent with a recall tool and can make multiple retrieval calls with different queries before finalising its answer."

    def __init__(self, llm: GeminiLLM | None = None, k: int = 10):
        self._llm = llm or GeminiLLM()
        self._rag = RAGMode(llm=self._llm, k=k)
        self.k = k

    @property
    def llm_id(self) -> str | None:
        return self._llm.model_id

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id, meta=meta))

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        k = self.k
        loop = asyncio.get_running_loop()

        retrieved_parts: list[str] = []

        def recall(query: str) -> str:  # noqa: redefined-outer-name
            # Submit async_retrieve to the running event loop from this thread
            future = asyncio.run_coroutine_threadsafe(
                memory.async_retrieve(query, k=k, user_id=user_id, query_timestamp=query_timestamp),
                loop,
            )
            docs, _ = future.result(timeout=120)
            text = "\n\n".join(f"## Memory {i + 1}\n{doc.content}" for i, doc in enumerate(docs))
            retrieved_parts.append(text)
            return text or "(no results found)"

        tool = ToolDef(
            name="recall",
            description="Search your memory bank for information relevant to the question. Call this multiple times with different queries to gather comprehensive context.",
            parameters={"query": {"type": "string", "description": "The search query to find relevant memories"}},
            required=["query"],
            fn=recall,
        )

        t0 = time.perf_counter()
        prompt = _SYSTEM_PROMPT.format(query=query)
        await asyncio.to_thread(self._llm.tool_loop, prompt, [tool])
        retrieve_ms = (time.perf_counter() - t0) * 1000

        context = "\n\n---\n\n".join(retrieved_parts)
        return self._finalize(query, context, task_type, retrieve_ms)

    def _finalize(self, query: str, context: str, task_type: str, retrieve_ms: float) -> AnswerResult:
        result = self._rag.answer_from_context(query, context, task_type)
        return AnswerResult(
            answer=result.answer,
            reasoning=result.reasoning,
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response=result.raw_response,
        )

    def answer_from_context(self, query: str, context: str, task_type: str = "open") -> AnswerResult:
        return self._rag.answer_from_context(query, context, task_type)
