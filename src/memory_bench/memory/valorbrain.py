"""
ValorBrain memory provider for the Agent Memory Benchmark.

ValorBrain is a hybrid memory engine: BM25 + dense vectors (pgvector) + RRF +
graph reranking + BGE cross-encoder rerank, all on PostgreSQL with RLS.

This provider talks to a running ValorBrain engine instance via its REST API.
Set VALORBRAIN_URL (default http://localhost:7438) and VALORBRAIN_TOKEN.

Each AMB isolation unit (e.g. BEAM conversation) becomes a ValorBrain collection.
"""

import json
import logging
import os
import time
import urllib.parse
import urllib.request

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)


class ValorBrainMemoryProvider(MemoryProvider):
    name = "valorbrain"
    description = (
        "ValorBrain hybrid memory engine — BM25 + dense (pgvector) + RRF + "
        "graph reranking + BGE cross-encoder rerank on PostgreSQL."
    )
    kind = "cloud"
    link = "https://valor.digital"
    concurrency = 4

    def __init__(self):
        self._base = os.environ.get("VALORBRAIN_URL", "http://localhost:7438").rstrip("/")
        self._token = os.environ.get("VALORBRAIN_TOKEN", "")
        self._tenant = os.environ.get("VALORBRAIN_BENCHMARK_TENANT_ID", "")
        self._ingested_collections: set[str] = set()

    # ── HTTP helper ──────────────────────────────────────────────────────

    def _post(self, path: str, body: dict, timeout: float = 120) -> dict:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **({"x-tenant-id": self._tenant} if self._tenant else {}),
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_stats_collection(self, collection: str, timeout: float = 30) -> dict:
        """GET /stats?collection=X.

        /stats is a GET endpoint; POSTing a body to it returns {"error": ...}.
        The provider used _post("/stats", {...}) for both the exists-check and
        the index-wait — both silently dead (every check saw an error response,
        existence never matched, _wait_index burned its full 120s timeout per
        collection). This GET uses the real contract: response carries
        collectionDocuments / collectionHybridPending / collectionEmbedFailed.
        """
        url = f"{self._base}/stats?collection={urllib.parse.quote(collection)}"
        req = urllib.request.Request(
            url,
            headers={
                **({"x-tenant-id": self._tenant} if self._tenant else {}),
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── Provider interface ───────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        # Re-window large chunks into ~8k-char windows (matching production's
        # 6-turn windows). The AMB chunks at 100k chars; our retrieval expects
        # ~38 smaller windows per conversation, not 7 huge ones. Without this,
        # the search pool is too small for good coverage.
        WINDOW_SIZE = 8000
        WINDOW_OVERLAP = 800  # ~1 turn overlap, like our janelaTurnos-1
        collections_seen: set[str] = set()
        for doc in documents:
            raw = (doc.user_id or "amb-default").lower()
            # Check if production-chunked collection already exists (beam-100k-N)
            existing = f"beam-100k-{raw}" if raw.isdigit() else raw
            try:
                stats = self._get_stats_collection(existing, timeout=15)
                if stats.get("collectionDocuments", 0) > 0:
                    self._ingested_collections.add(existing)
                    continue  # skip ingest — data already chunked and indexed
            except Exception:
                pass
            collection = existing  # use beam-100k-{raw} consistently with retrieve()
            collections_seen.add(collection)

            content = doc.content
            if len(content) <= WINDOW_SIZE:
                windows = [(doc.id or f"{collection}/0", content)]
            else:
                windows = []
                wi = 0
                pos = 0
                while pos < len(content):
                    end = min(pos + WINDOW_SIZE, len(content))
                    windows.append((f"{doc.id}_w{wi}", content[pos:end]))
                    wi += 1
                    if end >= len(content):
                        break
                    pos = end - WINDOW_OVERLAP

            for w_path, w_content in windows:
                body = {
                    "content": w_content,
                    "collection": collection,
                    "path": w_path,
                    "content_type": "conversation",
                }
                if doc.timestamp:
                    body["event_at"] = doc.timestamp
                try:
                    self._post("/documents", body, timeout=300)
                except Exception as e:
                    logger.warning("ValorBrain ingest failed for %s/%s: %s", collection, w_path, e)

        # Wait for hybrid index on each collection that received documents.
        for collection in collections_seen:
            if collection not in self._ingested_collections:
                self._wait_index(collection)
                self._refine_collection(collection)
                self._ingested_collections.add(collection)

    def _wait_index(self, collection: str, timeout: float = 120) -> None:
        """Poll GET /stats?collection= until the collection's hybrid index is ready.

        Bails early when documents exist, nothing is pending, and some embeds
        have permanently failed — those will not recover within the timeout
        (the embed worker retries failed docs on an ~1h cadence), so waiting
        the full window only stalls the run.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                stats = self._get_stats_collection(collection, timeout=30)
                pending = stats.get("collectionHybridPending", 0)
                total = stats.get("collectionDocuments", 0)
                failed = stats.get("collectionEmbedFailed", 0)
                if total > 0 and pending == 0:
                    return
                if total > 0 and pending == failed and failed > 0:
                    return  # everything indexed that will be indexed
            except Exception:
                pass
            time.sleep(2)

    def _refine_collection(self, collection: str, timeout: float = 300) -> None:
        """Extract observations + consolidate after ingest, before queries.

        Without this, /memory/prepare queries arrive before Phase 1.5 has
        extracted observations from the conversation docs. The answering LLM
        gets raw text instead of pre-extracted facts.
        """
        try:
            result = self._post(
                "/api/v1/memory/refine",
                {"collection": collection},
                timeout=timeout,
            )
            logger.info(
                "Refine %s: observed=%s extracted=%s consolidated=%s",
                collection,
                result.get("observed", 0),
                result.get("extracted", 0),
                result.get("consolidated", 0),
            )
        except Exception as e:
            logger.warning("Refine failed for %s: %s (continuing)", collection, e)

    def retrieve(
        self,
        query: str,
        k: int = 20,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        # Map AMB user_id (e.g. "1") to our production chunked collections
        # (e.g. "beam-100k-1") which have proper 6-turn windows (38 docs/conv).
        raw = user_id or "amb-default"
        collection = f"beam-100k-{raw}" if raw.isdigit() else raw

        # /memory/prepare delivers the full pipeline (funnel + multitrecho + rerank).
        # delivered_documents are snippeted server-side (6k). This is the production
        # path — same endpoint Hermes uses, validated by the benchmark.
        body: dict = {"message": query, "collection": collection}
        try:
            data = self._post("/api/v1/memory/prepare", body, timeout=60)
            funnel = data.get("funnel") or {}
            delivered = funnel.get("delivered_documents", [])
            if delivered:
                # Separate synthetic docs (digest, facts) from conversation windows.
                # Embed synthetic as a header at the TOP of the first memory,
                # so the reader sees it as context, not as competing memories.
                synthetic = [d for d in delivered if d.get("path", "").startswith("__")]
                conversations = [d for d in delivered if not d.get("path", "").startswith("__")]
                
                synth_text = ""
                if synthetic:
                    parts = []
                    for d in synthetic:
                        c = d.get("content", "").strip()
                        if c:
                            parts.append(c)
                    synth_text = "\n\n".join(parts)
                
                docs = []
                for i, d in enumerate(conversations):
                    content = d.get("content", "")
                    if not content:
                        continue
                    # Prepend synthetic context to the FIRST conversation doc
                    if i == 0 and synth_text:
                        content = (
                            "=== CONVERSATION SUMMARY (use for factual questions) ===\n"
                            + synth_text
                            + "\n=== END SUMMARY ===\n\n"
                            + content
                        )
                    docs.append(Document(id=d.get("path", ""), content=content, user_id=user_id))
                
                if docs:
                    return docs, data
        except Exception as e:
            logger.warning("ValorBrain prepare failed: %s", e)

        # Fallback: /search
        try:
            data = self._post("/search", {"query": query, "mode": "hybrid", "limit": k,
                "collection": collection, "compact": False}, timeout=60)
        except Exception as e:
            return [], {"error": str(e)}
        docs = [Document(id=r.get("docid",""), content=(r.get("body") or r.get("snippet",""))[:6000], user_id=user_id)
                for r in data.get("results",[]) if r.get("body") or r.get("snippet")]
        return docs, data

