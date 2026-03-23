import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from ..models import Document


class MemoryProvider(ABC):
    name: str
    description: str
    kind: str  # "local" or "cloud"
    provider: str | None = None   # provider family name; None means same as name
    variant: str | None = None    # variant label (e.g. "local", "cloud", "http"); None = no variants
    link: str | None = None       # URL to provider website / docs
    logo: str | None = None       # URL to provider logo image
    concurrency: int = 4  # max parallel queries; override to 1 for non-thread-safe providers

    def initialize(self) -> None:
        """Optional hook called once before everything else (before prepare).
        Providers that need external processes or one-time setup should implement this."""

    def cleanup(self) -> None:
        """Optional hook called after the run completes. Inverse of initialize."""

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None) -> None:
        """Optional hook called before ingest/retrieve with a persistent storage directory.
        Providers that need on-disk state (e.g. a vector store) should initialise it here.

        unit_ids — when the dataset declares an isolation_unit, this is the complete set of
        unique user_id values that will be seen during ingest/retrieve. Providers that want
        per-unit storage (e.g. one bank per episode) can create all units upfront here instead
        of relying on shared-bank tag scoping."""

    @abstractmethod
    def ingest(self, documents: list[Document]) -> None:
        """Ingest documents into memory."""
        ...

    async def async_ingest(self, documents: list[Document]) -> None:
        """Async version of ingest. Default falls back to running sync ingest in a thread."""
        return await asyncio.to_thread(self.ingest, documents)

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        """Retrieve top-k relevant documents for a query, optionally scoped to a user.
        Returns (documents, raw_response) where raw_response is the unprocessed provider response."""
        ...

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        """Async version of retrieve. Default falls back to running sync retrieve in a thread."""
        return await asyncio.to_thread(self.retrieve, query, k, user_id, query_timestamp)

    def retrieve_by_steps(self, steps: list[int], query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        """Retrieve facts for specific step/turn numbers. Default falls back to regular retrieve."""
        return self.retrieve(query, k, user_id, query_timestamp)

    async def async_retrieve_by_steps(self, steps: list[int], query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        """Async version of retrieve_by_steps."""
        return await asyncio.to_thread(self.retrieve_by_steps, steps, query, k, user_id, query_timestamp)

    def direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        """Directly answer the query using the memory system (e.g. reflect).
        Returns (answer, context_text, raw_response).
        Only implement for providers that support native agentic answering."""
        raise NotImplementedError(f"{self.name} does not support agent mode (direct_answer)")

    async def async_direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        """Async version of direct_answer. Default wraps sync version in a thread."""
        return await asyncio.to_thread(self.direct_answer, query, user_id=user_id, query_timestamp=query_timestamp)
