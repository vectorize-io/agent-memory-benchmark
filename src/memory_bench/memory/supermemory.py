import os
import time

from ..models import Document
from .base import MemoryProvider

_SEARCH_LIMIT = 30
_SEARCH_THRESHOLD = 0.3
_INDEXING_BACKOFF_START_MS = 1000
_INDEXING_BACKOFF_MAX_MS = 5000


def _build_context(results: list) -> str:
    """Build context in the same format as supermemory's official memorybench prompts."""
    # Collect all chunks across results, deduplicate by content, preserve order
    seen_chunks: set[str] = set()
    all_chunks: list[str] = []
    for r in results:
        chunks = r.chunks or []
        for chunk in chunks:
            if chunk.content not in seen_chunks:
                seen_chunks.add(chunk.content)
                all_chunks.append(chunk.content)
        if r.chunk and r.chunk not in seen_chunks:
            seen_chunks.add(r.chunk)
            all_chunks.append(r.chunk)

    # Memories section — only results with actual synthesized memory text
    memory_parts = []
    for i, r in enumerate(results):
        if not r.memory:
            continue
        lines = [f"Result {i + 1}:", r.memory]

        meta = r.metadata or {}
        temporal = meta.get("temporalContext") or meta.get("temporal_context") or {}
        doc_date = temporal.get("documentDate") or temporal.get("document_date")
        event_date = temporal.get("eventDate") or temporal.get("event_date")
        if doc_date or event_date:
            parts = []
            if doc_date:
                parts.append(f"documentDate: {doc_date}")
            if event_date:
                dates = event_date if isinstance(event_date, list) else [event_date]
                parts.append(f"eventDate: {', '.join(dates)}")
            lines.append(f"Temporal Context: {' | '.join(parts)}")

        memory_parts.append("\n".join(lines))

    memories_section = "\n\n---\n\n".join(memory_parts)

    if all_chunks:
        chunks_section = "\n\n=== DEDUPLICATED CHUNKS ===\n" + "\n\n---\n\n".join(all_chunks)
    else:
        chunks_section = ""

    return memories_section + chunks_section


class SupermemoryMemoryProvider(MemoryProvider):
    name = "supermemory"
    description = "Supermemory cloud API with temporal metadata support."
    kind = "cloud"
    link = "https://supermemory.ai"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=supermemory.ai"

    def __init__(self):
        from supermemory import Supermemory

        self._api_key = os.environ["SUPERMEMORY_API_KEY"]
        self._client = Supermemory(api_key=self._api_key)

    @staticmethod
    def _user_tag(user_id: str) -> str:
        return f"user-{user_id}"

    def _get_memory_status(self, doc_id: str) -> str:
        """Call GET /v3/memories/{doc_id} — not yet exposed in the Python SDK."""
        import httpx
        r = httpx.get(
            f"https://api.supermemory.ai/v3/memories/{doc_id}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            follow_redirects=True,
        )
        return r.json().get("status", "unknown")

    def ingest(self, documents: list[Document]) -> None:
        doc_ids = []
        for doc in documents:
            kwargs = dict(content=doc.content, metadata={"doc_id": doc.id})
            if doc.user_id:
                kwargs["container_tag"] = self._user_tag(doc.user_id)
            response = self._client.add(**kwargs)
            doc_ids.append(response.id)

        # Wait for both document and memory processing to complete (mirrors awaitIndexing in their TS repo)
        pending = set(doc_ids)
        backoff = _INDEXING_BACKOFF_START_MS / 1000
        while pending:
            done = set()
            for doc_id in list(pending):
                doc = self._client.documents.get(doc_id)
                if doc.status in ("done", "failed"):
                    mem_status = self._get_memory_status(doc_id)
                    if mem_status in ("done", "failed"):
                        done.add(doc_id)
            pending -= done
            if pending:
                time.sleep(backoff)
                backoff = min(backoff * 1.2, _INDEXING_BACKOFF_MAX_MS / 1000)

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        kwargs = dict(
            q=query,
            limit=_SEARCH_LIMIT,
            threshold=_SEARCH_THRESHOLD,
            search_mode="hybrid",
            include={"summaries": True},
        )
        if user_id:
            kwargs["container_tag"] = self._user_tag(user_id)
        response = self._client.search.memories(**kwargs)

        results = response.results or []
        context = _build_context(results)

        # Return as a single Document so the RAG mode injects the full formatted context
        docs = [Document(id="supermemory-context", content=context)] if context.strip() else []
        return docs, response.model_dump()
