"""Tree Ring Memory provider for Agent Memory Benchmark.

Uses the local ``tree-ring`` Rust CLI through JSONL import/export-friendly
commands. Set ``TREE_RING_BIN`` to point at a non-PATH binary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import Document
from .base import MemoryProvider


class TreeRingMemoryProvider(MemoryProvider):
    name = "tree-ring"
    description = (
        "Local-first Rust CLI memory using SQLite/FTS, ring lifecycle metadata, "
        "sensitivity checks, and project-scoped recall. Requires the tree-ring "
        "binary on PATH or TREE_RING_BIN."
    )
    kind = "local"
    link = "https://github.com/TerminallyLazy/Tree-Ring-Memory"
    logo = (
        "https://raw.githubusercontent.com/TerminallyLazy/Tree-Ring-Memory/main/"
        "assets/tree-ring-memory-logo.png"
    )
    concurrency = 1

    def __init__(self) -> None:
        self._binary = os.environ.get("TREE_RING_BIN", "tree-ring")
        self._root: Path | None = None
        self._default_project = "amb"

    def initialize(self) -> None:
        self._binary = self._resolve_binary(self._binary)
        completed = subprocess.run(
            [self._binary, "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "tree-ring CLI is not runnable. Install Tree Ring Memory or set "
                f"TREE_RING_BIN. stderr: {completed.stderr.strip()}"
            )

    def prepare(
        self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True
    ) -> None:
        del unit_ids
        self._root = store_dir / "tree-ring"
        if reset and self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._run("init")

    def ingest(self, documents: list[Document]) -> None:
        if not documents:
            return

        root = self._ensure_root()
        batch_path = root / f"amb-ingest-{uuid.uuid4().hex}.jsonl"
        try:
            lines = [
                json.dumps(self._event_for_document(doc), ensure_ascii=False)
                for doc in documents
            ]
            batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._run("import", str(batch_path))
        finally:
            batch_path.unlink(missing_ok=True)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        del query_timestamp
        args = ["recall", query, "--limit", str(k), "--include-sensitive"]
        project = self._project_for(user_id)
        if project:
            args.extend(["--project", project])

        payload = self._run(*args)
        raw = json.loads(payload) if payload else []

        docs: list[Document] = []
        for item in raw:
            memory = item.get("memory", {})
            source = memory.get("source") or {}
            doc_id = source.get("ref") or memory.get("id", "")
            content = self._content_for_memory(memory, item)
            if content.strip():
                docs.append(Document(id=str(doc_id), content=content, user_id=project))
        return docs, {"provider": self.name, "results": raw}

    def _event_for_document(self, doc: Document) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        project = self._project_for(doc.user_id)
        return {
            "id": self._stable_memory_id(doc.id, project),
            "created_at": now,
            "updated_at": now,
            "project": project,
            "agent_profile": None,
            "scope": "eval",
            "ring": "cambium",
            "event_type": "benchmark_document",
            "summary": self._format_document(doc),
            "details": "",
            "source": {
                "type": "benchmark",
                "ref": doc.id,
                "quote": "",
            },
            "tags": ["amb", f"doc_id:{doc.id}"],
            "salience": 0.5,
            "confidence": 0.5,
            "sensitivity": "normal",
            "retention": "normal",
            "expires_at": None,
            "supersedes": [],
            "superseded_by": None,
            "links": [],
            "review": {
                "needs_review": False,
                "review_reason": None,
                "reviewed_at": None,
                "reviewed_by": None,
            },
        }

    def _format_document(self, doc: Document) -> str:
        parts = [f"Doc ID: {doc.id}"]
        if doc.user_id:
            parts.append(f"User ID: {doc.user_id}")
        if doc.timestamp:
            parts.append(f"Timestamp: {doc.timestamp}")
        if doc.context:
            parts.append(f"Context: {doc.context}")

        messages = doc.messages
        if not messages and doc.content.strip().startswith("["):
            try:
                parsed = json.loads(doc.content)
                if isinstance(parsed, list):
                    messages = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "message").title()
                content = str(msg.get("content") or "").strip()
                if content:
                    parts.append(f"{role}: {content}")
        elif doc.content.strip():
            parts.append(doc.content)

        return "\n".join(parts)

    def _content_for_memory(self, memory: dict[str, Any], item: dict[str, Any]) -> str:
        lines = []
        source = memory.get("source") or {}
        if source.get("ref"):
            lines.append(f"Doc ID: {source['ref']}")
        if memory.get("project"):
            lines.append(f"Project: {memory['project']}")
        lines.append(str(memory.get("summary") or ""))
        if item.get("score") is not None:
            lines.append(f"score: {item['score']:.3f}")
        return "\n".join(lines)

    def _run(self, *args: str) -> str:
        root = self._ensure_root()
        completed = subprocess.run(
            [self._binary, "--root", str(root), "--json", *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            command = (
                f"{Path(self._binary).name} --root <store> --json "
                f"{args[0] if args else ''}"
            )
            raise RuntimeError(
                "tree-ring command failed: "
                f"{command}\nstdout: {completed.stdout[-2000:]}\nstderr: {completed.stderr[-2000:]}"
            )
        return completed.stdout.strip()

    def _ensure_root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Tree Ring provider was used before prepare()")
        return self._root

    def _resolve_binary(self, value: str) -> str:
        expanded = Path(value).expanduser()
        has_path_separator = os.sep in value or (
            os.altsep is not None and os.altsep in value
        )
        if has_path_separator:
            if not expanded.exists():
                raise RuntimeError(f"TREE_RING_BIN does not exist: {expanded}")
            return str(expanded)

        resolved = shutil.which(value)
        if resolved is None:
            raise RuntimeError(
                "tree-ring CLI not found. Install Tree Ring Memory or set TREE_RING_BIN "
                "to a built tree-ring executable."
            )
        return resolved

    def _project_for(self, user_id: str | None) -> str:
        return str(user_id) if user_id is not None else self._default_project

    @staticmethod
    def _stable_memory_id(doc_id: str, project: str) -> str:
        digest = hashlib.sha256(f"{project}\0{doc_id}".encode("utf-8")).hexdigest()
        return f"mem_amb_{digest[:24]}"
