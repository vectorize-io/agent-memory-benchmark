"""Hindsight knowledge banks over plain HTTP: files and text written into a bank, then search.

No LLM anywhere: the only model is the server's knowledge embedder. Used with
``--mode retrieval`` on the knowledge datasets (dataset/knowledge.py).

Configuration:
  HINDSIGHT_KNOWLEDGE_URL   server base URL (default http://localhost:8888)
  HINDSIGHT_KNOWLEDGE_KEY   API key, sent as a bearer token (optional for a local server)
  HINDSIGHT_KNOWLEDGE_BANK  with --skip-ingestion: search this existing bank instead of the
                            one the previous run of this split recorded
  AMB_KNOWLEDGE_SEARCH_CONCURRENCY / AMB_KNOWLEDGE_UPLOAD_CONCURRENCY   requests in flight (4 / 4)
  AMB_KNOWLEDGE_BANK_CONFIG  JSON of ingest overrides set on each new bank before its first upload,
                            e.g. {"text_passage_tokens": 2048}

A run creates one bank per split and keeps it (its id is in the run's _store directory), so
``--skip-ingestion`` can search it again. Delete old banks on a shared server yourself.

Published runs (one run each, not averaged over repeats):

``hindsight-kb`` (ViDoRe V3 PDF and markdown, 8 subsets, own-language queries; ATM-Bench)
  Server: hindsight main 80bd005fc for ViDoRe V3; ATM-Bench ran fully (ingest and search) on hindsight
  main fdae5ed70 (#6704), with the same settings. 80bd005fc has #6704 and #6746 (a PDF page counts as its own document in
  the rerank, and no regex rules on page text). Stock knowledge-bank settings. Only the providers are set:
    HINDSIGHT_API_ENABLE_KB=true
    HINDSIGHT_API_LLM_PROVIDER=none
    HINDSIGHT_API_KB_EMBEDDINGS_PROVIDER=openai          (embeddinggemma-2, the local default's model,
    HINDSIGHT_API_KB_EMBEDDINGS_MODEL=embeddinggemma-2    served by Hindsight Inference)
    HINDSIGHT_API_KB_EMBEDDINGS_BASE_URL=<Hindsight Inference>/v1
    HINDSIGHT_API_KB_EMBEDDINGS_MODALITIES=text,image,audio,video
    HINDSIGHT_API_KB_RERANKER_URL=<Hindsight Inference>/v1/rerank
    HINDSIGHT_API_KB_RERANKER_MODEL=hs-decide-1-flash-mm
  Everything else is the server's default: 512-token text passages, search, one rerank request with
  page pictures, search budget mid. AMB commit 9738aa0. Each split has a Postgres of its own and ran
  in a fresh server. The markdown banks were ingested by 80bd005fc; the PDF banks by
  hindsight main fdae5ed70, whose ingest is the same (the commits between change no PDF, image or
  video ingest), and searched on 80bd005fc. Every bank was checked complete (all ready, none failed):
    uv run amb run --dataset <ds> --split <split> --memory hindsight-knowledge --mode retrieval \
      --skip-answer                                         (ingest)
    uv run amb run ... --skip-ingestion --category <english|french>   (search; none for ATM-Bench)
  french for finance_fr, energy and physics, english for the other ViDoRe subsets.

``hindsight-no-filenames`` / ``hindsight-filenames`` (LegalBench-RAG mini)
  Server: hindsight main fdae5ed70 (#6704), the same settings as above. This set is plain text;
  #6746 (in 80bd005fc) changes nothing in how text passages are scored.
  ``hindsight-no-filenames`` uploads each contract under a neutral name, the default;
  ``hindsight-filenames`` sets AMB_LEGALBENCH_NAMES=original to keep the corpus file names, as a
  user would upload them. Each run ingests in a fresh server, then searches:
    uv run amb run --dataset legalbench-rag --split mini --memory hindsight-knowledge --mode retrieval

  One run each: the reranker varies between runs (the same LegalBench-RAG bank and code: 90.6 and
  93.4 recall@5), so a difference of a point or two on one subset is within that.
  Video is not published for this version yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Sequence
from pathlib import Path

import httpx

from .base import MemoryProvider
from ..models import Document

_log = logging.getLogger(__name__)

PREFIX = "/v1/default/knowledge-banks"
SEARCH_CONCURRENCY = int(os.getenv("AMB_KNOWLEDGE_SEARCH_CONCURRENCY", "4"))
UPLOAD_CONCURRENCY = int(os.getenv("AMB_KNOWLEDGE_UPLOAD_CONCURRENCY", "4"))
SEARCH_RETRIES = 3
#: Waits on a 429 before a call gives up: with the 30 s cap, about seven minutes of throttling.
RATE_LIMIT_RETRIES = 20
#: The search budget asked for: the API's largest, so every ranked candidate comes back.
SCORING_BUDGET = 1_000_000
INGEST_TIMEOUT = float(os.getenv("AMB_KNOWLEDGE_INGEST_TIMEOUT", str(6 * 3600)))
FILES_PER_UPLOAD = 16
#: The text ingest's ceiling per request.
TEXTS_PER_UPLOAD = 100
_BANK_FILE = "knowledge_bank_id"


class KnowledgeApi:
    """The knowledge-bank routes. They are hidden from the OpenAPI spec, so the published
    client has no methods for them."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(600.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs) -> bytes:
        # A deployed server sits behind a rate limiter (nginx answers 429 with no Retry-After), and a
        # search answers 503 when its embedder was rate limited: wait and ask again instead of failing the run.
        for attempt in range(RATE_LIMIT_RETRIES):
            r = await self._http.request(method, f"{PREFIX}{path}", **kwargs)
            if r.status_code not in (429, 503):
                break
            await asyncio.sleep(float(r.headers.get("retry-after") or min(2**attempt, 30)))
        if r.status_code >= 300:
            raise httpx.HTTPStatusError(f"{method} {path}: {r.status_code} {r.text[:500]}", request=r.request, response=r)
        return r.content

    async def create_bank(self, bank_id: str) -> None:
        await self._call("POST", "", json={"bank_id": bank_id})
        overrides = os.getenv("AMB_KNOWLEDGE_BANK_CONFIG")
        if overrides:  # before any upload: a bank splits its files with the settings it had then
            await self._call("PATCH", f"/{bank_id}/config", json={"updates": json.loads(overrides)})

    async def delete_bank(self, bank_id: str) -> None:
        await self._call("DELETE", f"/{bank_id}")

    async def bank(self, bank_id: str) -> dict:
        return json.loads(await self._call("GET", f"/{bank_id}"))

    async def ingest_files(self, bank_id: str, docs: Sequence[Document]) -> None:
        files = [("files", (d.filename or Path(d.path).name, Path(d.path).read_bytes())) for d in docs]
        options = json.dumps({"files": [{"document_id": d.id} for d in docs]})
        # A large upload now and then gets a 500 from a Postgres connection the file save broke;
        # the same request again succeeds, and with the same document ids it is an upsert.
        for attempt in range(3):
            try:
                await self._call("POST", f"/{bank_id}/files/ingest", files=files, data={"request": options})
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt == 2:
                    raise
            await asyncio.sleep(5)

    async def ingest_texts(self, bank_id: str, docs: Sequence[Document]) -> None:
        items = [
            {"document_id": d.id, "text": d.content, "filename": d.filename or f"{d.id}.txt", "tags": list(d.tags or [])}
            for d in docs
        ]
        await self._call("POST", f"/{bank_id}/ingest", json={"items": items, "async": True})

    async def failed(self, bank_id: str, limit: int = 20) -> list[dict]:
        # The list endpoint caps a page at 500, so walk pages until `limit` (or the end).
        docs: list[dict] = []
        while len(docs) < limit:
            page_size = min(500, limit - len(docs))
            params = {"status": "failed", "limit": page_size, "offset": len(docs)}
            page = json.loads(await self._call("GET", f"/{bank_id}/documents", params=params))["documents"]
            docs.extend(page)
            if len(page) < page_size:
                break
        return docs

    async def search(self, bank_id: str, query: str, top_k: int, max_tokens: int | None = None) -> list[Document]:
        """The ``top_k`` best passages, as (document id, page). The API takes a token budget, not
        a count: by default ask for the deepest one and keep the ranks that are scored. With
        ``max_tokens`` (a dataset scored on the returned text), search at that budget and keep the text."""
        budget = max_tokens or SCORING_BUDGET
        raw = await self._call("POST", f"/{bank_id}/search", json={"query": query, "max_tokens": budget})
        results = json.loads(raw)["results"][:top_k]
        return [
            Document(id=r["document_id"], content=(r.get("text") or "") if max_tokens else "", page=r.get("page"))
            for r in results
        ]


async def _upload(api: KnowledgeApi, bank_id: str, documents: Sequence[Document]) -> None:
    # Several requests in flight: one at a time, a deployed server indexes faster than the
    # client uploads.
    gate = asyncio.Semaphore(UPLOAD_CONCURRENCY)

    async def send(upload: Awaitable[None]) -> None:
        async with gate:
            await upload

    files = [d for d in documents if d.path]
    texts = [d for d in documents if not d.path]
    await asyncio.gather(
        *(send(api.ingest_files(bank_id, files[m : m + FILES_PER_UPLOAD])) for m in range(0, len(files), FILES_PER_UPLOAD)),
        *(send(api.ingest_texts(bank_id, texts[m : m + TEXTS_PER_UPLOAD])) for m in range(0, len(texts), TEXTS_PER_UPLOAD)),
    )


async def ingest(api: KnowledgeApi, bank_id: str, documents: Sequence[Document], timeout: float = INGEST_TIMEOUT) -> dict:
    """Create the bank, upload every document and wait until each is ready or failed."""
    await api.create_bank(bank_id)
    started = time.perf_counter()
    await _upload(api, bank_id, documents)
    uploaded = time.perf_counter() - started
    # One bank read per poll, not one per document.
    retried = False
    while True:
        bank = await api.bank(bank_id)
        if bank["ready_count"] + bank["failed_count"] >= len(documents):
            if not bank["failed_count"] or retried:
                break
            # A remote embedder drops a long request now and then and the server fails that
            # ingest's documents. Writing them again once (same ids: an upsert) keeps a blip from
            # failing a run; a document that fails twice is reported.
            retried = True
            failed = {d["document_id"] for d in await api.failed(bank_id, limit=len(documents))}
            await _upload(api, bank_id, [d for d in documents if d.id in failed])
            await asyncio.sleep(5.0)
            continue
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f"{bank_id}: {bank['ready_count']}/{len(documents)} ready after {timeout:.0f}s")
        await asyncio.sleep(1.0)
    wall = time.perf_counter() - started
    return {
        "documents": len(documents),
        "passages": bank["passage_count"],
        "failed": bank["failed_count"],
        "upload_s": round(uploaded, 2),
        "wall_s": round(wall, 2),
    }


class HindsightKnowledgeMemoryProvider(MemoryProvider):
    name = "hindsight-knowledge"
    description = "Hindsight knowledge banks over HTTP: files (PDF, image, video) and text written into a knowledge bank, then searched. No LLM: the server's knowledge embedder is the only model."
    kind = "cloud"
    provider = "hindsight"
    variant = "knowledge"
    link = "https://github.com/vectorize-io/hindsight"
    concurrency = SEARCH_CONCURRENCY
    #: Takes ``retrieval_filter``, only for ``max_tokens``: a dataset scored on the returned text.
    supports_filters = True

    def __init__(self) -> None:
        self._url = os.environ.get("HINDSIGHT_KNOWLEDGE_URL", "http://localhost:8888")
        self._key = os.environ.get("HINDSIGHT_KNOWLEDGE_KEY") or None
        self._bank_id: str | None = None
        self._search_api: KnowledgeApi | None = None
        self._search_loop: asyncio.AbstractEventLoop | None = None
        self.ingest_stats: dict | None = None

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        recorded = store_dir / _BANK_FILE
        if not reset:
            self._bank_id = os.environ.get("HINDSIGHT_KNOWLEDGE_BANK") or (
                recorded.read_text().strip() if recorded.exists() else None
            )
            if not self._bank_id:
                raise RuntimeError(f"--skip-ingestion: no bank recorded in {recorded}; set HINDSIGHT_KNOWLEDGE_BANK")
            return
        # store_dir is outputs/<dataset>/<run>/_store/<split>/<category>
        slug = re.sub(r"[^a-z0-9-]+", "-", f"{store_dir.parts[-5]}-{store_dir.parts[-2]}".lower()).strip("-")
        self._bank_id = f"amb-kb-{slug}-{uuid.uuid4().hex[:6]}"
        recorded.write_text(self._bank_id)

    def ingest(self, documents: list[Document]) -> None:
        async def run() -> dict:
            api = KnowledgeApi(self._url, self._key)
            try:
                stats = await ingest(api, self._bank_id, documents)
                if stats["failed"]:
                    failed = [f"{d['document_id']}: {d.get('error')}" for d in await api.failed(self._bank_id)]
                    raise RuntimeError(f"{self._bank_id}: {stats['failed']} documents failed: {failed}")
                return stats
            finally:
                await api.aclose()

        self.ingest_stats = asyncio.run(run())
        _log.info("knowledge bank %s: %s", self._bank_id, self.ingest_stats)

    def _api(self) -> KnowledgeApi:
        # The runner answers every query inside one event loop; an httpx client is bound to the
        # loop it was made in, so one client per loop.
        loop = asyncio.get_running_loop()
        if self._search_api is None or self._search_loop is not loop:
            self._search_api, self._search_loop = KnowledgeApi(self._url, self._key), loop
        return self._search_api

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                             query_timestamp: str | None = None, filters: dict | None = None) -> tuple[list[Document], dict | None]:
        # A remote embedder behind a tunnel drops a connection now and then (the server says
        # 503): a few retries keep one blip from failing a run; anything else still raises.
        for attempt in range(SEARCH_RETRIES + 1):
            try:
                return await self._api().search(self._bank_id, query, k, (filters or {}).get("max_tokens")), None
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", 503)
                if attempt == SEARCH_RETRIES or status < 500:
                    raise
            await asyncio.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        return asyncio.run(self.async_retrieve(query, k, user_id, query_timestamp))
