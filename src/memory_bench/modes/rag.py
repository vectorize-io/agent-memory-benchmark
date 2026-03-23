import asyncio
import time

from .base import ResponseMode
from ..llm.base import LLM, Schema
from ..llm.gemini import GeminiLLM
from ..memory.base import MemoryProvider
from ..models import AnswerResult

from ..dataset.base import _DEFAULT_OPEN_PROMPT, _DEFAULT_MCQ_PROMPT

_OPEN_SCHEMA = Schema(
    properties={
        "reasoning": {"type": "string", "description": "Step-by-step explanation of how you used the context to arrive at the answer."},
        "answer":    {"type": "string", "description": "The final concise answer to the question. If the context lacks the information, say so."},
    },
    required=["reasoning", "answer"],
)

_MCQ_SCHEMA = Schema(
    properties={
        "reasoning": {"type": "string", "description": "Step-by-step explanation of which option best fits the context."},
        "choice":    {"type": "string", "description": "The letter of the chosen option: a, b, c, or d."},
    },
    required=["reasoning", "choice"],
)


class RAGMode(ResponseMode):
    name = "rag"
    description = "Default. Provider retrieves top-k documents; they are injected into an LLM prompt as context. Supports both MCQ and open-ended questions."

    def __init__(self, llm: LLM | None = None):
        from ..llm import get_answer_llm
        self._llm = llm or get_answer_llm()

    @property
    def llm_id(self) -> str | None:
        return self._llm.model_id

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id, meta=meta))

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        t0 = time.perf_counter()
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        retrieval_query = meta.get("retrieval_query") or query
        docs, raw_response = await memory.async_retrieve(retrieval_query, user_id=user_id, query_timestamp=query_timestamp)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        context = "\n\n".join(
            f"## Memory {i + 1}\n{doc.content}" for i, doc in enumerate(docs)
        )

        prompt_fn = meta.get("_prompt_fn")
        if task_type == "mcq":
            return await asyncio.to_thread(self._answer_mcq, query, context, retrieve_ms, raw_response, prompt_fn, meta)
        return await asyncio.to_thread(self._answer_open, query, context, retrieve_ms, raw_response, prompt_fn, meta)

    def answer_from_context(self, query: str, context: str, task_type: str = "open", meta: dict | None = None) -> AnswerResult:
        prompt_fn = (meta or {}).get("_prompt_fn")
        if task_type == "mcq":
            return self._answer_mcq(query, context, retrieve_ms=0.0, raw_response=None, prompt_fn=prompt_fn, meta=meta)
        return self._answer_open(query, context, retrieve_ms=0.0, raw_response=None, prompt_fn=prompt_fn, meta=meta)

    def _answer_open(self, query: str, context: str, retrieve_ms: float, raw_response: dict | None, prompt_fn=None, meta: dict | None = None) -> AnswerResult:
        if prompt_fn:
            effective_meta = {**(meta or {}), "_raw_response": raw_response}
            prompt = prompt_fn(query, context, meta=effective_meta)
        else:
            prompt = _DEFAULT_OPEN_PROMPT.format(context=context, query=query)
        data = self._llm.generate(prompt, _OPEN_SCHEMA)
        return AnswerResult(
            answer=data["answer"],
            reasoning=data["reasoning"],
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response=raw_response,
        )

    def _answer_mcq(self, query: str, context: str, retrieve_ms: float, raw_response: dict | None, prompt_fn=None, meta: dict | None = None) -> AnswerResult:
        if prompt_fn:
            effective_meta = {**(meta or {}), "_raw_response": raw_response}
            prompt = prompt_fn(query, context, meta=effective_meta)
        else:
            prompt = _DEFAULT_MCQ_PROMPT.format(context=context, query=query)
        data = self._llm.generate(prompt, _MCQ_SCHEMA)
        choice = data["choice"].strip().lower().strip("(). ")[:1]
        return AnswerResult(
            answer=choice,
            reasoning=data["reasoning"],
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response=raw_response,
        )
