"""Hindsight over plain HTTP for supportbench: the same corpus stored three ways, answered by reflect.

The three providers differ only in where each kind of document goes (dataset/supportbench.py says
how a document's kind is read):

=================================  ==============================  ==============================
provider                           memory bank                     knowledge bank (linked)
=================================  ==============================  ==============================
``hindsight-support-memory``       manuals, chats, exceptions      —
``hindsight-support-knowledge``    nothing                         manuals, chats, exceptions
``hindsight-support-linked``       chats                           manuals, exceptions
=================================  ==============================  ==============================

One memory bank and one knowledge bank per run. A customer's chats and exceptions carry the tag
``user:<id>``, the manuals carry none, and every reflect asks for ``tags=[user:<id>]`` with
``tags_match="any"`` — that customer's documents plus the untagged shared ones. Whether the
knowledge-bank side honours that scoping is the server's business, and the ``exception`` questions
are what catch it when it does not.

Both sides get the manuals as the same PDF files: the memory bank through ``/files/retain`` (parsed
to markdown, then fact extraction), the knowledge bank through its own file ingest.

Cost and time, in each row's meta:

- ``query_usd`` / ``query_tokens``: the reflect's own token usage (``usage`` on the response) priced
  at ``AMB_SUPPORT_LLM_MODEL`` (default gemma-4-26b-a4b) with MEMORY_PRICES. Embedding and
  reranking are not LLM calls and are not counted.
- ``ingest_usd``: the server's LLM spend on the memory bank once ingestion settled (retain, file
  parsing with a vision model if the server has one, consolidation). The knowledge bank makes no
  LLM call. None on a server that does not record its LLM requests (``llm_trace`` off, as on dev).
- ``ingest_s``: wall time of the whole ingestion, both banks.

Configuration:
  HINDSIGHT_HTTP_URL / HINDSIGHT_HTTP_KEY   the server (default http://localhost:8888, no key)
  AMB_SUPPORT_LLM_MODEL                     the model the server's reflect runs, for pricing
  AMB_SUPPORT_INGEST_TIMEOUT                seconds to wait for retain to settle (default 6 h)
  AMB_ENABLE_OBSERVATIONS=1                 let the memory bank consolidate (off, as in the other suites)

The server needs ``HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS=true`` for the knowledge and linked arms.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import httpx

from ..dataset.supportbench import scenario
from ..models import Document
from .base import MemoryProvider
from .hindsight_knowledge import KnowledgeApi
from .hindsight_knowledge import ingest as knowledge_ingest
from .hscoding import MEMORY_PRICES, priced_usage

_log = logging.getLogger(__name__)

BANKS = "/v1/default/banks"
INGEST_TIMEOUT = float(os.getenv("AMB_SUPPORT_INGEST_TIMEOUT", str(6 * 3600)))
LLM_MODEL = os.getenv("AMB_SUPPORT_LLM_MODEL", "gemma-4-26b-a4b")
_BANKS_FILE = "support_banks.json"


def user_tag(user_id: str) -> str:
    return f"user:{user_id}"


def is_manual(doc: Document) -> bool:
    return doc.path is not None


def is_chat(doc: Document) -> bool:
    return doc.messages is not None


def query_usd(usage: dict | None, model: str = LLM_MODEL) -> float | None:
    """A reflect's token usage priced at ``model``; None when the model or the usage is unknown."""
    price = MEMORY_PRICES.get(model)
    if not usage or price is None:
        return None
    inp, cached = usage.get("input_tokens") or 0, usage.get("cached_tokens") or 0
    out = (usage.get("output_tokens") or 0) + (usage.get("thoughts_tokens") or 0)
    return round(((inp - cached) * price["input"] + cached * price["cached"] + out * price["output"]) / 1e6, 6)


class _Api:
    """The memory-bank routes this provider uses."""

    def __init__(self, base_url: str, api_key: str | None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(600.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(self, method: str, path: str, **kwargs) -> dict:
        # A rate-limited server answers bursts with 429, and a deployed one's gateway times a long
        # reflect out now and then (504 stream timeout): wait and ask again instead of failing the run.
        for attempt in range(8):
            r = await self._http.request(method, f"{BANKS}/{path}", **kwargs)
            if r.status_code != 429 and not (r.status_code >= 500 and attempt < 3):
                break
            await asyncio.sleep(float(r.headers.get("retry-after") or min(2**attempt, 30)))
        if r.status_code >= 300:
            raise httpx.HTTPStatusError(f"{method} {path}: {r.status_code} {r.text[:500]}", request=r.request, response=r)
        return r.json() if r.content else {}


class _SupportProvider(MemoryProvider):
    kind = "cloud"
    provider = "hindsight"
    link = "https://github.com/vectorize-io/hindsight"
    concurrency = int(os.getenv("AMB_CONCURRENCY", "4"))
    #: Which side each kind of document goes to: "memory" or "knowledge".
    route: dict[str, str]

    def __init__(self) -> None:
        self._url = os.environ.get("HINDSIGHT_HTTP_URL", "http://localhost:8888")
        self._key = os.environ.get("HINDSIGHT_HTTP_KEY") or None
        self._memory_bank = ""
        self._knowledge_bank: str | None = None
        self._ingest: dict = {}
        self._clients: dict[asyncio.AbstractEventLoop, _Api] = {}

    @property
    def _linked(self) -> bool:
        return "knowledge" in self.route.values()

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        recorded = store_dir / _BANKS_FILE
        if not reset:
            if not recorded.exists():
                raise RuntimeError(f"--skip-ingestion: no banks recorded in {recorded}")
            saved = json.loads(recorded.read_text())
            self._memory_bank, self._knowledge_bank, self._ingest = saved["memory"], saved["knowledge"], saved["ingest"]
            return
        slug = re.sub(r"[^a-z0-9-]+", "-", self.name.removeprefix("hindsight-support-")).strip("-")
        run = uuid.uuid4().hex[:6]
        self._memory_bank = f"amb-support-{slug}-{run}"
        self._knowledge_bank = f"amb-support-kb-{slug}-{run}" if self._linked else None
        self._store = recorded

    def _side(self, doc: Document) -> str:
        return self.route["manual" if is_manual(doc) else "chat" if is_chat(doc) else "exception"]

    def ingest(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._ingest = asyncio.run(self._aingest(documents))
        self._ingest["ingest_s"] = round(time.perf_counter() - started, 1)
        self._store.write_text(json.dumps({"memory": self._memory_bank, "knowledge": self._knowledge_bank, "ingest": self._ingest}))
        _log.info("%s: ingested %s", self.name, self._ingest)

    async def _aingest(self, documents: list[Document]) -> dict:
        api = _Api(self._url, self._key)
        try:
            enable_obs = os.environ.get("AMB_ENABLE_OBSERVATIONS", "") in ("1", "true")
            await api.call("PUT", self._memory_bank, json={"enable_observations": enable_obs})
            memory = [d for d in documents if self._side(d) == "memory"]
            knowledge = [dataclasses.replace(d, tags=[user_tag(d.user_id)]) if d.user_id else d
                         for d in documents if self._side(d) == "knowledge"]
            jobs = [self._retain(api, memory)]
            if self._knowledge_bank:
                kb = KnowledgeApi(self._url, self._key)
                jobs.append(knowledge_ingest(kb, self._knowledge_bank, knowledge))
            results = await asyncio.gather(*jobs)
            if self._knowledge_bank:
                await kb.aclose()
                await api.call("PATCH", f"{self._memory_bank}/config", json={"updates": {"knowledge_banks": [self._knowledge_bank]}})
            try:
                cost = await asyncio.to_thread(priced_usage, self._get, self._memory_bank)
            except httpx.HTTPStatusError as exc:
                # A server that does not record its LLM requests (llm_trace off) has no such route:
                # the ingest cost is unknown there, not zero.
                if exc.response.status_code != 404:
                    raise
                _log.warning("%s: the server does not report LLM usage; ingest cost unknown", self._memory_bank)
                cost = {"usd": None}
            return {"memory_docs": len(memory), "knowledge_docs": len(knowledge), "knowledge": results[1] if len(results) > 1 else None,
                    "ingest_usd": cost["usd"], "ingest_cost": cost}
        finally:
            await api.aclose()

    async def _retain(self, api: _Api, docs: list[Document]) -> None:
        if not docs:
            return
        manuals = [d for d in docs if is_manual(d)]
        if manuals:
            files = [("files", (d.filename, Path(d.path).read_bytes(), "application/pdf")) for d in manuals]
            meta = {"files_metadata": [{"document_id": d.id, "context": d.context} for d in manuals]}
            await api.call("POST", f"{self._memory_bank}/files/retain", files=files, data={"request": json.dumps(meta)})
        items = [
            {"content": d.content, "document_id": d.id, "context": d.context, "tags": [user_tag(d.user_id)],
             **({"timestamp": d.timestamp} if d.timestamp else {})}
            for d in docs if not is_manual(d)
        ]
        if items:
            await api.call("POST", f"{self._memory_bank}/memories", json={"items": items, "async": True})
        deadline = time.perf_counter() + INGEST_TIMEOUT
        while True:
            stats = await api.call("GET", f"{self._memory_bank}/stats")
            if stats.get("failed_operations"):
                raise RuntimeError(f"{self._memory_bank}: {stats['failed_operations']} retain operation(s) failed")
            if not stats.get("pending_operations"):
                return
            if time.perf_counter() > deadline:
                raise TimeoutError(f"{self._memory_bank}: retain still pending after {INGEST_TIMEOUT:.0f}s")
            await asyncio.sleep(5.0)

    def _get(self, path: str) -> dict:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        r = httpx.get(f"{self._url}{BANKS}/{path}", headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def _api(self) -> _Api:
        # httpx clients are bound to the loop they were made in; the runner answers in one loop.
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = _Api(self._url, self._key)
        return self._clients[loop]

    async def async_direct_answer(self, query: str, user_id: str | None = None,
                                  query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        s = scenario()
        context = f"{s['agent_context']} The customer is {s['users'][user_id]['name']}."
        if query_timestamp:
            context += f" The question is being asked on {query_timestamp}."
        body = {"query": query, "context": context, "tags": [user_tag(user_id)], "tags_match": "any"}
        started = time.perf_counter()
        response = await self._api().call("POST", f"{self._memory_bank}/reflect", json=body)
        usage = response.get("usage") or {}
        answer = response.get("text") or ""
        meta = {
            "reflect_s": round(time.perf_counter() - started, 2),
            "query_tokens": usage.get("total_tokens"),
            "query_usd": query_usd(usage),
            "ingest_usd": self._ingest.get("ingest_usd"),
            "ingest_s": self._ingest.get("ingest_s"),
        }
        return answer, answer, {"meta": meta}

    def direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        return asyncio.run(self.async_direct_answer(query, user_id, query_timestamp))

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        raise NotImplementedError(f"{self.name} answers with reflect: run it with --mode agent")


class HindsightSupportMemoryProvider(_SupportProvider):
    name = "hindsight-support-memory"
    description = "supportbench arm: manuals, chats and approved exceptions all retained into one memory bank; reflect answers."
    variant = "support-memory"
    route = {"manual": "memory", "chat": "memory", "exception": "memory"}


class HindsightSupportKnowledgeProvider(_SupportProvider):
    name = "hindsight-support-knowledge"
    description = "supportbench arm: manuals, chats and approved exceptions all in one knowledge bank, linked to an empty memory bank; reflect answers."
    variant = "support-knowledge"
    route = {"manual": "knowledge", "chat": "knowledge", "exception": "knowledge"}


class HindsightSupportLinkedProvider(_SupportProvider):
    name = "hindsight-support-linked"
    description = "supportbench arm: chats in a memory bank, manuals and approved exceptions in a linked knowledge bank; reflect answers."
    variant = "support-linked"
    route = {"manual": "knowledge", "chat": "memory", "exception": "knowledge"}
