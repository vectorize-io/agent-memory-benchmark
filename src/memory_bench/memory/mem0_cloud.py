import os
import time
import uuid
from pathlib import Path

from rich.console import Console

from ..models import Document
from .base import MemoryProvider

_console = Console()


class Mem0CloudMemoryProvider(MemoryProvider):
    name = "mem0-cloud"
    description = "Mem0 cloud API. Async indexing — waits for indexing to complete before eval. k=20."
    kind = "cloud"
    provider = "mem0"
    variant = "cloud"

    def __init__(self, k: int = 20):
        self.k = k
        self._client = None
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def _ensure_client(self):
        if self._client is None:
            from mem0 import MemoryClient
            api_key = os.environ.get("MEM0_CLOUD_KEY") or os.environ["MEM0_API_KEY"]
            self._client = MemoryClient(api_key=api_key)
        return self._client

    def ingest(self, documents: list[Document]) -> None:
        client = self._ensure_client()

        # Delete existing memories for each user to avoid duplicates
        user_ids = {doc.user_id or self._default_user_id for doc in documents}
        for uid in user_ids:
            try:
                client.delete_all(user_id=uid)
            except Exception:
                pass

        for doc in documents:
            uid = doc.user_id or self._default_user_id
            messages = doc.messages or [{"role": "user", "content": doc.content}]
            client.add(messages=messages, user_id=uid, metadata={"doc_id": doc.id})

        # Poll until memories are indexed (async processing on mem0's servers)
        self._wait_for_indexing(client, user_ids)

    def _wait_for_indexing(self, client, user_ids: set[str], timeout: int = 300) -> None:
        """Poll until at least one memory is searchable, then wait proportionally."""
        sample_uid = next(iter(user_ids))
        deadline = time.time() + timeout
        _console.print("[dim]  Waiting for mem0 cloud indexing...[/dim]")
        while time.time() < deadline:
            try:
                result = client.get_all(filters={"user_id": sample_uid}, limit=1)
                entries = result.get("results", result) if isinstance(result, dict) else result
                if entries:
                    _console.print(f"[dim]  Indexed.[/dim]")
                    return
            except Exception:
                pass
            time.sleep(5)
        _console.print("[yellow]  Warning: timed out waiting for mem0 indexing[/yellow]")

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        results = self._ensure_client().search(query, filters={"user_id": uid}, top_k=self.k)
        entries = results.get("results", results) if isinstance(results, dict) else results
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
