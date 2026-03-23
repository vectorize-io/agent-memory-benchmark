import tempfile
import uuid
from pathlib import Path

from mem0 import Memory

from ..models import Document
from .base import MemoryProvider

# sentence-transformers/multi-qa-MiniLM-L6-cos-v1: 384-dim, fast, no API calls
_LOCAL_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
_LOCAL_EMBEDDING_DIMS = 384


class Mem0MemoryProvider(MemoryProvider):
    name = "mem0"
    description = "Agentic memory with Gemini 2.0 Flash for reflective extraction + local Qdrant store."
    kind = "local"
    provider = "mem0"
    variant = "local"
    link = "https://mem0.ai"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=mem0.ai"

    def __init__(self, k: int = 20):
        self.k = k
        self._memory: Memory | None = None
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None) -> None:
        qdrant_path = store_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._memory = self._build_memory(str(qdrant_path))

    def _build_memory(self, qdrant_path: str) -> Memory:
        return Memory.from_config(
            {
                "llm": {
                    "provider": "gemini",
                    "config": {"model": "gemini-2.0-flash", "temperature": 0.0},
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": _LOCAL_EMBEDDING_MODEL,
                        "embedding_dims": _LOCAL_EMBEDDING_DIMS,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "bench",
                        "path": qdrant_path,
                        "embedding_model_dims": _LOCAL_EMBEDDING_DIMS,
                        "on_disk": True,
                    },
                },
            }
        )

    def _ensure_memory(self) -> Memory:
        if self._memory is None:
            # Fallback for tests / direct use without prepare()
            self._memory = self._build_memory(tempfile.mkdtemp(prefix="mem0_bench_"))
        return self._memory

    def ingest(self, documents: list[Document]) -> None:
        memory = self._ensure_memory()
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            messages = doc.messages or [{"role": "user", "content": doc.content}]
            memory.add(messages=messages, user_id=uid, metadata={"doc_id": doc.id})

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        results = self._ensure_memory().search(query, user_id=uid, limit=self.k)
        entries = (
            results.get("results", results) if isinstance(results, dict) else results
        )
        raw = results if isinstance(results, dict) else {"results": results}
        docs = []
        for r in entries:
            lines = [r["memory"]]
            if r.get("score") is not None:
                lines.append(f"score: {r['score']:.3f}")
            if r.get("created_at"):
                lines.append(f"created: {r['created_at']}")
            if r.get("updated_at"):
                lines.append(f"updated: {r['updated_at']}")
            meta = r.get("metadata") or {}
            if meta:
                lines.append(f"metadata: {meta}")
            docs.append(Document(id=r["id"], content="\n".join(lines)))
        return docs, raw
