"""No-memory provider — the baseline arm. Ingests nothing and retrieves nothing, so an eval with
`--memory none` measures the agent/model with no memory system at all."""
from .base import MemoryProvider
from ..models import Document


class NoMemoryProvider(MemoryProvider):
    name = "none"
    description = "No memory system (baseline)."
    kind = "local"

    def ingest(self, documents: list[Document]) -> None:
        pass

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        return [], None
