"""Letta memory provider.

Documents are stored as passages in a Letta archive (one archive per isolation
unit) and retrieved with Letta's semantic passage search. Agent mode answers
through a Letta agent that has the archive attached, so the agent decides for
itself when and what to search.

Works against Letta Cloud (LETTA_API_KEY) or a self-hosted server
(LETTA_BASE_URL, e.g. http://localhost:8283).
"""

import os
import threading
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_BATCH_SIZE = 50


def _message_text(content) -> str:
    """Flatten Letta message content (a string or a list of text parts) into text."""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content or [] if getattr(part, "text", None))


def _prefix_from_store_dir(store_dir: Path) -> str:
    """Derive a stable archive-name prefix from the run's store directory."""
    parts = store_dir.parts
    try:
        idx = parts.index("_store")
        return f"amb-{parts[idx - 2]}-{parts[idx + 1]}"
    except (ValueError, IndexError):
        return "amb-bench"


class LettaMemoryProvider(MemoryProvider):
    name = "letta"
    description = (
        "Letta archival memory: documents are written as passages into a per-unit archive "
        "and retrieved by semantic search. Agent mode answers through a Letta agent that "
        "searches its own archival memory."
    )
    kind = "cloud"
    link = "https://letta.com"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=letta.com"

    def __init__(self, k: int = 20):
        self.k = k
        self._client = None
        self._prefix = "amb-bench"
        self._per_unit = False
        self._archive_ids: dict[str | None, str] = {}
        self._agent_ids: dict[str | None, str] = {}
        self._agent_locks: dict[str | None, threading.Lock] = {}
        self._lock = threading.Lock()
        self._embedding = os.environ.get("LETTA_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self._model = os.environ.get("LETTA_MODEL", "openai/gpt-4.1")
        self._max_steps = int(os.environ.get("LETTA_MAX_STEPS", "10"))

    def initialize(self) -> None:
        from letta_client import Letta

        if not os.environ.get("LETTA_API_KEY") and not os.environ.get("LETTA_BASE_URL"):
            raise RuntimeError(
                "letta provider needs LETTA_API_KEY (Letta Cloud) or LETTA_BASE_URL (self-hosted server)"
            )
        # api_key comes from LETTA_API_KEY; base_url from LETTA_BASE_URL when self-hosted.
        self._client = Letta()

    def cleanup(self) -> None:
        # Archives are kept (they hold the ingested corpus); the throwaway agents are not.
        for agent_id in self._agent_ids.values():
            try:
                self._client.agents.delete(agent_id)
            except Exception:
                pass
        self._agent_ids.clear()

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._prefix = _prefix_from_store_dir(store_dir)
        self._per_unit = unit_ids is not None
        self._archive_ids.clear()
        self._agent_ids.clear()
        self._agent_locks.clear()
        for unit in sorted(unit_ids) if unit_ids else [None]:
            self._ensure_archive(unit, reset=reset)

    def _archive_name(self, unit: str | None) -> str:
        return f"{self._prefix}-u{unit}" if unit is not None else self._prefix

    def _ensure_archive(self, unit: str | None, reset: bool = False) -> str:
        with self._lock:
            if unit in self._archive_ids:
                return self._archive_ids[unit]
            name = self._archive_name(unit)
            existing = list(self._client.archives.list(name=name, limit=100))
            if reset:
                for archive in existing:
                    self._client.archives.delete(archive.id)
                existing = []
            archive = existing[0] if existing else self._client.archives.create(
                name=name,
                description="Agent Memory Benchmark run",
                embedding=self._embedding,
            )
            self._archive_ids[unit] = archive.id
            return archive.id

    def _unit(self, user_id: str | None) -> str | None:
        return user_id if self._per_unit else None

    @staticmethod
    def _text(doc: Document) -> str:
        if doc.timestamp:
            return f"[Date: {doc.timestamp}]\n{doc.content}"
        return doc.content

    def ingest(self, documents: list[Document]) -> None:
        by_unit: dict[str | None, list[Document]] = {}
        for doc in documents:
            by_unit.setdefault(self._unit(doc.user_id), []).append(doc)

        for unit, docs in by_unit.items():
            archive_id = self._ensure_archive(unit)
            passages = [
                {"text": self._text(doc), "metadata": {"doc_id": doc.id}}
                for doc in docs
            ]
            for i in range(0, len(passages), _BATCH_SIZE):
                self._client.archives.passages.create_many(
                    archive_id, passages=passages[i : i + _BATCH_SIZE]
                )

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None
    ) -> tuple[list[Document], dict | None]:
        archive_id = self._ensure_archive(self._unit(user_id))
        results = self._client.passages.search(archive_id=archive_id, query=query, limit=k or self.k)

        docs = []
        raw_results = []
        for i, r in enumerate(results):
            passage = r.passage
            docs.append(Document(id=passage.id or f"letta-{i}", content=passage.text))
            raw_results.append(
                {
                    "id": passage.id,
                    "text": passage.text,
                    "score": r.score,
                    "tags": passage.tags,
                    "metadata": passage.metadata,
                }
            )
        return docs, {"results": raw_results}

    def _ensure_agent(self, unit: str | None) -> tuple[str, threading.Lock]:
        archive_id = self._ensure_archive(unit)
        with self._lock:
            if unit not in self._agent_ids:
                agent = self._client.agents.create(
                    name=f"{self._archive_name(unit)}-{uuid.uuid4().hex[:6]}",
                    model=self._model,
                    embedding=self._embedding,
                    include_base_tools=True,
                    message_buffer_autoclear=True,
                )
                self._client.agents.archives.attach(archive_id, agent_id=agent.id)
                self._agent_ids[unit] = agent.id
                self._agent_locks[unit] = threading.Lock()
            return self._agent_ids[unit], self._agent_locks[unit]

    def direct_answer(
        self, query: str, user_id: str | None = None, query_timestamp: str | None = None
    ) -> tuple[str, str, dict | None]:
        unit = self._unit(user_id)
        agent_id, lock = self._ensure_agent(unit)
        # A Letta agent processes messages sequentially; concurrent sends interleave.
        with lock:
            response = self._client.agents.messages.create(
                agent_id, input=query, max_steps=self._max_steps
            )

        answers: list[str] = []
        context_parts: list[str] = []
        for message in response.messages:
            if message.message_type == "assistant_message":
                answers.append(_message_text(message.content))
            elif message.message_type == "tool_return_message":
                context_parts.append(message.tool_return)

        return "\n".join(answers), "\n\n".join(context_parts), response.model_dump(mode="json")
