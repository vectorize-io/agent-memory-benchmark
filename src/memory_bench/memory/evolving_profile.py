"""Evolving Profile (EP 6.2+) memory provider for Agent Memory Benchmark (AMB).

Connects to a local or self-hosted Evolving Profile Memory API (default:
``http://127.0.0.1:12088``, override with ``EP_API_URL``) using only the Python
standard library — no additional dependencies required.

Architecture highlights:
  - Per-unit isolated banks (``amb-ep-<unit_id>``) created and cleaned up per run
  - Timestamped document ingestion with structured-message formatting
  - Multi-facet recall + chronological ``read_source`` original-document readback
    with explicit ``source_ids`` attribution
"""
from __future__ import annotations

import collections
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..models import Document
from .base import MemoryProvider


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


def _safe_bank_id(prefix: str, unit_id: str | None) -> str:
    raw = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(unit_id or "default")).strip("-")
    return f"{prefix}-{raw[:48]}" if raw else f"{prefix}-default"


class EvolvingProfileMemoryProvider(MemoryProvider):
    """Evolving Profile (EP 6.2+) local-first memory provider with read_source readback."""

    name = "evolving-profile"
    description = (
        "Evolving Profile (EP 6.2+): local-first AI memory, 5D conditional preferences "
        "& agent experience control plane (built on Hindsight) with multi-facet recall "
        "and chronological original-source readback."
    )
    kind = "local"
    provider = "evolving-profile"
    variant = "local-readback"
    link = "https://github.com/ccygod/evolving-profile"
    concurrency = 2
    supports_filters = False

    def __init__(
        self,
        *,
        api_url: str | None = None,
        budget: str = "high",
        max_tokens: int = 6000,
        bank_prefix: str = "amb-ep",
        use_readback: bool = True,
        batch_size: int = 3,
    ) -> None:
        self.api_url = (api_url or os.environ.get("EP_API_URL") or "http://127.0.0.1:12088").rstrip("/")
        self.budget = budget
        self.max_tokens = max_tokens
        self.bank_prefix = bank_prefix
        self.use_readback = use_readback
        self.batch_size = max(1, batch_size)
        self._created_banks: set[str] = set()
        self._doc_order: dict[str, dict[str, str]] = collections.defaultdict(dict)

    def _bank_url(self, user_id: str | None) -> tuple[str, str]:
        bid = _safe_bank_id(self.bank_prefix, user_id)
        return bid, f"{self.api_url}/v1/default/banks/{bid}"

    @staticmethod
    def _format_content(doc: Document) -> str:
        """Render Document content cleanly, preserving structured turns when present."""
        messages = doc.messages
        if not messages and doc.content.strip().startswith("["):
            try:
                parsed = json.loads(doc.content)
                if isinstance(parsed, list):
                    messages = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        if messages and isinstance(messages, list):
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = str(msg.get("role") or "user").capitalize()
                    content = str(msg.get("content") or "").strip()
                    if content:
                        parts.append(f"{role}: {content}")
            if parts:
                text = "\n".join(parts)
                if doc.timestamp:
                    text = f"[Date: {doc.timestamp}]\n{text}"
                return text

        return doc.content

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        if reset and unit_ids:
            for uid in unit_ids:
                bid, base = self._bank_url(uid)
                try:
                    _http_json("DELETE", base, timeout=30.0)
                except Exception:
                    pass
                self._created_banks.discard(bid)
                self._doc_order.pop(bid, None)

    def ingest(self, documents: list[Document]) -> None:
        by_user: dict[str, list[Document]] = collections.defaultdict(list)
        for doc in documents:
            uid = str(doc.user_id or "default")
            by_user[uid].append(doc)

        for uid, docs in by_user.items():
            bid, base = self._bank_url(uid)
            self._created_banks.add(bid)
            items: list[dict[str, Any]] = []
            for idx, doc in enumerate(docs):
                item: dict[str, Any] = {
                    "content": self._format_content(doc),
                    "document_id": doc.id,
                    "context": str(doc.context or f"Document {doc.id}"),
                }
                if doc.timestamp:
                    item["timestamp"] = str(doc.timestamp)
                if doc.tags:
                    item["tags"] = list(doc.tags)
                items.append(item)
                self._doc_order[bid][doc.id] = str(doc.timestamp or f"{idx:06d}")

            for i in range(0, len(items), self.batch_size):
                _http_json(
                    "POST",
                    f"{base}/memories",
                    {"items": items[i : i + self.batch_size], "async": False},
                    timeout=300.0,
                )

    def _read_source(self, base: str, doc_id: str) -> str:
        encoded = urllib.parse.quote(doc_id, safe="")
        try:
            resp = _http_json("GET", f"{base}/documents/{encoded}", timeout=30.0)
            return str(resp.get("original_text") or resp.get("content") or resp.get("text") or "").strip()
        except Exception:
            return ""

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        bid, base = self._bank_url(user_id)
        facets = [
            query,
            f"timeline, preferences, updates, and supporting facts related to: {query[:180]}",
        ]
        seen_ids: set[str] = set()
        recalled: list[dict[str, Any]] = []
        doc_scores: dict[str, float] = collections.defaultdict(float)

        for f_idx, fq in enumerate(facets):
            payload: dict[str, Any] = {
                "query": fq,
                "budget": self.budget,
                "max_tokens": self.max_tokens,
            }
            if query_timestamp:
                payload["query_timestamp"] = query_timestamp
            resp = _http_json("POST", f"{base}/memories/recall", payload, timeout=90.0)
            weight = 1.5 if f_idx == 0 else 1.0
            for rank, item in enumerate(resp.get("results", [])):
                rid = str(item.get("id") or f"{f_idx}:{rank}")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    recalled.append(item)
                did = item.get("document_id") or item.get("source_document_id") or ""
                if did:
                    doc_scores[str(did)] += weight / (rank + 1.0)

        out_docs: list[Document] = []
        readback_count = 0
        if self.use_readback and doc_scores:
            top_dids = [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[: max(1, min(k, 8))]]
            order_map = self._doc_order.get(bid, {})
            top_dids.sort(key=lambda d: order_map.get(d, d))
            for did in top_dids:
                txt = self._read_source(base, did)
                if txt:
                    readback_count += 1
                    out_docs.append(
                        Document(
                            id=did,
                            content=txt,
                            user_id=user_id,
                            source_ids=[did],
                            context="ep_read_source",
                        )
                    )

        for item in recalled:
            if len(out_docs) >= k:
                break
            txt = str(item.get("text") or "").strip()
            if txt:
                did = item.get("document_id") or item.get("source_document_id")
                out_docs.append(
                    Document(
                        id=str(item.get("id") or ""),
                        content=txt,
                        user_id=user_id,
                        timestamp=item.get("occurred_start"),
                        source_ids=[str(did)] if did else None,
                        context="ep_recall",
                    )
                )

        return out_docs[:k], {
            "bank_id": bid,
            "recalled_count": len(recalled),
            "readback_docs": readback_count,
        }

    def cleanup(self) -> None:
        for bid in list(self._created_banks):
            try:
                _http_json("DELETE", f"{self.api_url}/v1/default/banks/{bid}", timeout=30.0)
            except Exception:
                pass
            self._created_banks.discard(bid)
