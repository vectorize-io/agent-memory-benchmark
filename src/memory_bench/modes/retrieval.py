import time

from .base import ResponseMode
from ..memory.base import MemoryProvider
from ..models import AnswerResult


class RetrievalMode(ResponseMode):
    """Retrieval-only mode — no LLM anywhere in the loop.

    Used by belief-ID-scored datasets (PrecisionMemBench): the retrieved documents
    themselves are the answer, and the dataset scores the returned ID set against
    the case's must-include / must-exclude assertions.

    The retrieved documents are handed to the runner in `raw_response["documents"]`
    so the dataset's scorer can see IDs and provenance, not just rendered text.
    """

    name = "retrieval"
    description = "No LLM. Returns the provider's retrieved memories verbatim; the dataset scores the returned ID set."

    @property
    def llm_id(self) -> str | None:
        return None

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open",
               user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        import asyncio
        return asyncio.run(self.async_answer(query, memory, task_type, user_id, meta))

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open",
                           user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        k = int(meta.get("retrieval_limit") or 20)
        filters = meta.get("retrieval_filter") if getattr(memory, "supports_filters", False) else None
        if not (query or "").strip():
            # Nothing to retrieve for. Providers reject an empty query outright, and a
            # dataset that ships blank-query cases means them to return nothing.
            return AnswerResult(
                answer="0 memories retrieved", reasoning="",
                context="## Retrieved memories (0)", retrieve_time_ms=0.0,
                raw_response={"documents": [], "provider_raw": None},
            )
        t0 = time.perf_counter()
        docs, raw = await memory.async_retrieve(
            query, k=k, user_id=user_id, query_timestamp=meta.get("query_timestamp"), filters=filters
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000

        lines = [f"## Retrieved memories ({len(docs)})"]
        for i, d in enumerate(docs):
            src = f" ← {', '.join(d.source_ids)}" if getattr(d, "source_ids", None) else ""
            lines.append(f"{i + 1}. [{d.id}]{src}\n{d.content}")
        context = "\n\n".join(lines)

        return AnswerResult(
            answer=f"{len(docs)} memories retrieved",
            reasoning="",
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response={"documents": docs, "provider_raw": raw},
        )

    def answer_from_context(self, query: str, context: str, task_type: str = "open",
                            meta: dict | None = None) -> AnswerResult:
        raise NotImplementedError(
            "retrieval mode scores provider-returned document IDs, which a cached "
            "context string cannot reconstruct — re-run without --skip-retrieval."
        )
