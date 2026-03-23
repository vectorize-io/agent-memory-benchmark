import asyncio
import json
import os
import threading
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_DATA_DIR = Path(".data/cognee")


def _chunks_from_result(r) -> list[tuple[str, str]]:
    """Extract all (text, id) pairs from a cognee CHUNKS search result.

    Cognee groups all matching chunks per dataset into one result dict:
      {'dataset_id': ..., 'search_result': [{'id': '...', 'text': '...'}, ...]}
    """
    if isinstance(r, dict):
        inner = r.get("search_result") or []
        if isinstance(inner, list):
            return [
                (c.get("text", ""), str(c.get("id", uuid.uuid4())))
                for c in inner
                if isinstance(c, dict) and c.get("text")
            ]
        text = r.get("text", r.get("content", ""))
        return [(text, str(r.get("id", uuid.uuid4())))] if text else []
    if hasattr(r, "text"):
        return [(r.text, str(getattr(r, "id", uuid.uuid4())))]
    return [(str(r), str(uuid.uuid4()))]


class CogneeMemoryProvider(MemoryProvider):
    name = "cognee"
    description = "Graph-based knowledge extraction with FastEmbed (BAAI/bge-small-en-v1.5) + OpenAI LLM."
    kind = "local"
    link = "https://cognee.ai"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=cognee.ai"

    def __init__(self):
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        os.environ["LLM_MODEL"] = "gpt-4o-mini"

        # Dedicated event loop in a background thread so cognee's async
        # internals (DB connections, etc.) share a single persistent loop.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._run(self._setup())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _setup(self):
        import cognee
        from cognee.infrastructure.llm.config import get_llm_config

        data_dir = str(_DATA_DIR.resolve())
        cognee.config.data_root_directory(data_dir)
        cognee.config.system_root_directory(data_dir)

        llm_cfg = get_llm_config()
        llm_cfg.llm_provider = "openai"
        llm_cfg.llm_api_key = os.environ.get("OPENAI_API_KEY", "")
        llm_cfg.llm_model = "gpt-4o-mini"

        # Use local fastembed embeddings — no API key needed.
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )
        emb_cfg = get_embedding_config()
        emb_cfg.embedding_provider = "fastembed"
        emb_cfg.embedding_model = "BAAI/bge-small-en-v1.5"
        emb_cfg.embedding_dimensions = 384

        self._cognee = cognee

    def _dataset_name(self, user_id: str | None) -> str:
        uid = user_id or self._default_user_id
        return f"bench_{uid.replace('-', '_')}"

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        self._run(self._ingest_async(documents))

    async def _ingest_async(self, documents: list[Document]) -> None:
        cognee = self._cognee

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        by_user: dict[str, list[Document]] = {}
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            by_user.setdefault(uid, []).append(doc)

        for user_id, docs in by_user.items():
            dataset = self._dataset_name(user_id)
            for doc in docs:
                await cognee.add(doc.content, dataset_name=dataset)
            await cognee.cognify([dataset], chunk_size=512)

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        return self._run(self._retrieve_async(query, k, user_id))

    async def _retrieve_async(
        self, query: str, k: int, user_id: str | None
    ) -> tuple[list[Document], dict | None]:
        from cognee.api.v1.search import SearchType

        dataset = self._dataset_name(user_id)
        results = await self._cognee.search(
            query_text=query,
            query_type=SearchType.CHUNKS,
            datasets=[dataset],
        )

        docs: list[Document] = []
        for r in results or []:
            for content, doc_id in _chunks_from_result(r):
                docs.append(Document(id=doc_id, content=content))
                if len(docs) >= k:
                    break

        # Serialize raw results (UUIDs etc. need str conversion)
        def _jsonable(obj):
            if isinstance(obj, dict):
                return {k: _jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_jsonable(i) for i in obj]
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        raw = _jsonable(results) if results else None
        return docs, raw
