"""MemoryHub provider for AMB.

Uses the memoryhub SDK to talk to a MemoryHub MCP server over
streamable-HTTP.  MemoryHub stores verbatim conversations and retrieves
via hybrid vector + keyword search with reciprocal-rank fusion and
cross-encoder reranking.

Required env vars
-----------------
MEMORYHUB_URL        MCP server endpoint (e.g. https://…/mcp/)
MEMORYHUB_API_KEY    API key for session auth

Optional env vars
-----------------
MEMORYHUB_PROJECT_ID   Project for memory isolation (default: amb-benchmark)
MEMORYHUB_K            Retrieval depth (default: 70)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from memoryhub import MemoryHubClient

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)


class MemoryHubProvider(MemoryProvider):
    name = "memoryhub"
    description = (
        "MemoryHub: hybrid vector + keyword search with "
        "cross-encoder reranking and reciprocal-rank fusion."
    )
    kind = "cloud"
    link = "https://github.com/redhat-ai-americas/memory-hub"
    concurrency = 1

    def __init__(self):
        self._url: str | None = None
        self._api_key: str | None = None
        self._project_id: str | None = None
        self._k: int = 70

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        self._url = os.environ.get("MEMORYHUB_URL")
        self._api_key = os.environ.get("MEMORYHUB_API_KEY")
        self._project_id = os.environ.get("MEMORYHUB_PROJECT_ID", "amb-benchmark")
        self._k = int(os.environ.get("MEMORYHUB_K", "70"))
        if not self._url or not self._api_key:
            raise RuntimeError(
                "MEMORYHUB_URL and MEMORYHUB_API_KEY are required. "
                "Point MEMORYHUB_URL at the MCP server's streamable-HTTP endpoint."
            )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        asyncio.run(self._ingest(documents))

    async def _ingest(self, documents: list[Document]) -> None:
        async with MemoryHubClient(url=self._url, api_key=self._api_key) as client:
            try:
                await client.create_project(
                    self._project_id,
                    description="AMB benchmark memory isolation",
                )
                logger.info("Created project %s", self._project_id)
            except Exception:
                logger.debug("Project %s already exists", self._project_id)

            for i, doc in enumerate(documents):
                owner = f"amb-{doc.user_id}" if doc.user_id else "amb-default"
                result = await client.write(
                    content=doc.content,
                    scope="project",
                    project_id=self._project_id,
                    owner_id=owner,
                    content_type="experiential",
                    force=True,
                )

                if not (result.memory):
                    logger.warning(
                        "Write returned no memory for doc %s", doc.id,
                    )

                if (i + 1) % 50 == 0:
                    logger.info("Ingested %d/%d documents", i + 1, len(documents))

        logger.info("Ingestion complete: %d documents", len(documents))

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        return asyncio.run(self._retrieve(query, k, user_id))

    async def _retrieve(
        self,
        query: str,
        k: int | None,
        user_id: str | None,
    ) -> tuple[list[Document], dict | None]:
        effective_k = self._k if (k is None or k == 10) else k
        owner = f"amb-{user_id}" if user_id else "amb-default"

        async with MemoryHubClient(url=self._url, api_key=self._api_key) as client:
            results = await client.search(
                query=query,
                max_results=effective_k,
                owner_id=owner,
                project_id=self._project_id,
                weight_threshold=0.0,
                mode="full_only",
                max_response_tokens=0,
            )

        documents = [
            Document(id=m.id, content=m.content, user_id=user_id)
            for m in results.results
        ]
        return documents, None

    def cleanup(self) -> None:
        pass
