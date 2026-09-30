import uuid
from pathlib import Path

from atmem import Memory
from atmem.retrieve import decide_retrieval

from ..models import Document
from .base import MemoryProvider


class AtMemMemoryProvider(MemoryProvider):
    name = "atmem"
    description = (
        "Local auditable memory with deterministic extraction, lifecycle governance, "
        "SQLite persistence, lexical/graph retrieval, and calibrated direct-support selection."
    )
    kind = "local"
    link = "https://github.com/aetna000/atmem"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=github.com"
    concurrency = 1

    def __init__(self):
        self._memory: Memory | None = None
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        self.cleanup()
        database = store_dir / "atmem.db"
        if reset:
            database.unlink(missing_ok=True)
        self._memory = Memory(database, graph_recall=True, auto_vectors=False)

    def cleanup(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def _ensure_memory(self) -> Memory:
        if self._memory is None:
            self._memory = Memory(":memory:", graph_recall=True, auto_vectors=False)
        return self._memory

    @staticmethod
    def _format_content(doc: Document) -> str:
        if not doc.messages:
            return doc.content

        lines = []
        if doc.timestamp:
            lines.append(f"Date: {doc.timestamp}")
        for message in doc.messages:
            role = str(message.get("role") or "unknown").capitalize()
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) or doc.content

    def ingest(self, documents: list[Document]) -> None:
        memory = self._ensure_memory()
        for doc in documents:
            subject_id = doc.user_id or self._default_user_id
            memory.remember(
                subject_id,
                self._format_content(doc),
                force=True,
                session_id=doc.id,
                source_type="user_message",
                raw={"amb_document_id": doc.id, "source_timestamp": doc.timestamp},
            )

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        memory = self._ensure_memory()
        subject_id = user_id or self._default_user_id
        session_id = f"amb-retrieval-{uuid.uuid4().hex}"
        records = memory.recall(
            subject_id,
            query,
            session_id=session_id,
            limit=k,
            use_graph=True,
            include_scores=True,
        )
        decision = decide_retrieval(query, records)
        records_by_id = {str(record["id"]): record for record in records}
        selected_records = [
            records_by_id[record_id]
            for record_id in decision.ranked_record_ids
            if record_id in records_by_id
        ][:k]
        [retrieval] = memory.get_retrieval_log(subject_id, session_id=session_id)

        documents = []
        for record in selected_records:
            source_id = record.get("source_session_id")
            documents.append(
                Document(
                    id=str(record["id"]),
                    content=str(record["content"]),
                    user_id=subject_id,
                    source_ids=[str(source_id)] if source_id else None,
                )
            )

        raw = {
            "retrieval_id": retrieval["id"],
            "candidate_ids": retrieval["returned_ids"],
            "returned_ids": [str(record["id"]) for record in selected_records],
            "candidates": retrieval["candidates"],
            "decision": decision.to_dict(),
        }
        return documents, raw
