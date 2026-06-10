"""Context Swarm Memory provider for Agent Memory Benchmark.

This file is copied into an AMB checkout by `npm run amb:patch`. It keeps
AMB's public runner unchanged while delegating retrieval to the TypeScript
CSM implementation in the repo pointed to by `CSM_REPO_DIR`.

Transport: a warm CSM bridge server (`npm run amb:csm:serve` in the CSM repo)
is started once in `initialize()` and queried over localhost HTTP. This is
the ingest-once / query-many replacement for the original per-query
subprocess bridge, which paid a Node spawn + corpus rebuild on every
retrieval. The server holds AMB documents in memory only; CSM's durable
memory is never touched.

Environment:
  CSM_REPO_DIR                       path to the context-swarm-memory checkout (required)
  GEMINI_API_KEY / GOOGLE_API_KEY    key for CSM's internal retrieval model (the
                                     CSM repo's gitignored .env is auto-loaded too)
  CSM_AMB_MODEL / CSM_MODEL          CSM internal retrieval model (default gemini-3.5-flash)
  CSM_AMB_MODEL_CONTEXT              CSM context-assembly budget in tokens (default 8192)
  CSM_AMB_RETURN_K                   override AMB's k for returned documents
  CSM_AMB_TELEMETRY_JSONL            per-query CSM token/latency sidecar path
  CSM_AMB_SERVER_CMD                 override server launch command (default: npm run -s amb:csm:serve -- --port 0)
  CSM_AMB_SERVER_STARTUP_TIMEOUT_SEC server readiness deadline (default 120)
  CSM_AMB_SERVER_LOG                 server stderr log path (default: <store_dir>/csm-server-stderr.log or temp)
  CSM_AMB_RETRIEVE_TIMEOUT_SEC       per-retrieve HTTP timeout (default 600)
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path

from ..models import Document
from .base import MemoryProvider


class CSMMemoryProvider(MemoryProvider):
    name = "csm"
    description = "Context Swarm Memory bridge backed by the TypeScript CSM repo (warm localhost service)."
    kind = "local"
    provider = "context-swarm-memory"
    variant = "amb-bridge"
    link = "https://github.com/muhamadjawdatsalemalakoum/context-swarm-memory"
    concurrency = 1

    def __init__(self) -> None:
        self._store_dir: Path | None = None
        self._documents_path: Path | None = None
        self._repo_dir = Path(os.environ.get("CSM_REPO_DIR", "")).expanduser()
        if not str(self._repo_dir):
            self._repo_dir = Path.cwd()
        self._model = os.environ.get("CSM_AMB_MODEL") or os.environ.get("CSM_MODEL") or "gemini-3.5-flash"
        self._proc: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._stderr_log = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._ensure_server()

    def cleanup(self) -> None:
        if self._port is not None:
            try:
                self._post("/shutdown", {}, timeout=5.0)
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._stderr_log is not None:
            try:
                self._stderr_log.close()
            except Exception:
                pass
        self._proc = None
        self._port = None
        self._stderr_log = None

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._documents_path = self._store_dir / "documents.jsonl"
        if not self._repo_dir.exists():
            raise RuntimeError(
                f"CSM_REPO_DIR does not exist: {self._repo_dir}. "
                "Set CSM_REPO_DIR to the context-swarm-memory checkout."
            )
        self._ensure_server()

        if reset:
            self._post("/reset", {})
            if self._documents_path.exists():
                self._documents_path.unlink()
            return

        # Resume path: replay the durable document log into the (empty) server
        # so `--skip-ingestion`-style runs see the previously ingested state.
        if self._documents_path.exists():
            health = self._get("/healthz")
            if int(health.get("documents", 0)) == 0:
                docs = []
                with self._documents_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            docs.append(json.loads(line))
                if docs:
                    self._post("/ingest", {"documents": docs})

    # ── data path ────────────────────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        if self._documents_path is None:
            raise RuntimeError("CSMMemoryProvider.prepare() must run before ingest().")
        rows = [
            {
                "id": doc.id,
                "content": doc.content,
                "user_id": doc.user_id,
                "timestamp": doc.timestamp,
                "context": doc.context,
            }
            for doc in documents
        ]
        # Durable log first (resume/audit), then the warm server.
        with self._documents_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
        self._post("/ingest", {"documents": rows})

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        self._ensure_server()
        return_k = int(os.environ.get("CSM_AMB_RETURN_K", str(k)))

        started = time.perf_counter()
        payload = self._post(
            "/retrieve",
            {
                "query": query,
                "k": return_k,
                "user_id": user_id,
                "query_timestamp": query_timestamp,
            },
            timeout=float(os.environ.get("CSM_AMB_RETRIEVE_TIMEOUT_SEC", "600")),
        )

        docs = [
            Document(
                id=str(item.get("id", f"csm-doc-{idx}")),
                content=str(item.get("content", "")),
                user_id=item.get("user_id"),
                timestamp=item.get("timestamp"),
                context=item.get("context"),
            )
            for idx, item in enumerate(payload.get("documents", []))
        ]
        raw = payload.get("raw_response") or {}
        raw["bridge_wall_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._append_telemetry(query, return_k, user_id, docs, raw)
        return docs, raw

    # ── warm server management ───────────────────────────────────────────────

    def _ensure_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None and self._port is not None:
            return
        if not self._repo_dir.exists():
            raise RuntimeError(
                f"CSM_REPO_DIR does not exist: {self._repo_dir}. "
                "Set CSM_REPO_DIR to the context-swarm-memory checkout."
            )

        cmd_override = os.environ.get("CSM_AMB_SERVER_CMD")
        if cmd_override:
            cmd = shlex.split(cmd_override, posix=os.name != "nt")
        else:
            npm = shutil.which("npm")
            if not npm:
                raise RuntimeError("npm not found on PATH (Node 22+ is required for the CSM bridge).")
            cmd = [npm, "run", "-s", "amb:csm:serve", "--", "--port", "0"]

        log_path = os.environ.get("CSM_AMB_SERVER_LOG")
        if not log_path:
            base = self._store_dir if self._store_dir is not None else Path(tempfile.gettempdir())
            log_path = str(Path(base) / "csm-server-stderr.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._stderr_log = open(log_path, "a", encoding="utf-8")

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._repo_dir),
            stdout=subprocess.PIPE,
            stderr=self._stderr_log,
            text=True,
            encoding="utf-8",
        )

        deadline = time.monotonic() + float(os.environ.get("CSM_AMB_SERVER_STARTUP_TIMEOUT_SEC", "120"))
        port: int | None = None
        assert self._proc.stdout is not None
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"CSM bridge server exited with code {self._proc.returncode} before becoming ready. "
                        f"See log: {log_path}"
                    )
                time.sleep(0.05)
                continue
            if line.startswith("AMB_CSM_SERVER_READY"):
                try:
                    port = int(line.strip().split("port=", 1)[1])
                except (IndexError, ValueError) as err:
                    raise RuntimeError(f"Unparseable CSM server ready line: {line!r}") from err
                break
        if port is None:
            raise RuntimeError(f"CSM bridge server did not become ready in time. See log: {log_path}")
        self._port = port

        # Keep draining stdout so the child never blocks on a full pipe.
        def _drain(stream) -> None:
            try:
                for _ in stream:
                    pass
            except Exception:
                pass

        threading.Thread(target=_drain, args=(self._proc.stdout,), daemon=True).start()

    def _post(self, route: str, body: dict, timeout: float = 120.0) -> dict:
        return self._request("POST", route, body, timeout)

    def _get(self, route: str, timeout: float = 30.0) -> dict:
        return self._request("GET", route, None, timeout)

    def _request(self, method: str, route: str, body: dict | None, timeout: float) -> dict:
        if self._port is None:
            raise RuntimeError("CSM bridge server is not running (initialize() not called?).")
        url = f"http://127.0.0.1:{self._port}{route}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")[:4000]
            raise RuntimeError(f"CSM bridge {method} {route} failed: HTTP {err.code} :: {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"CSM bridge {method} {route} failed: {err.reason}") from err

    # ── telemetry ────────────────────────────────────────────────────────────

    def _append_telemetry(
        self,
        query: str,
        return_k: int,
        user_id: str | None,
        docs: list[Document],
        raw: dict,
    ) -> None:
        telemetry_path = os.environ.get("CSM_AMB_TELEMETRY_JSONL")
        if not telemetry_path:
            return

        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        record = {
            "provider": "context-swarm-memory",
            "llm_provider": raw.get("llm_provider"),
            "llm_model": raw.get("llm_model"),
            "bridge_mode": raw.get("mode"),
            "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
            "query": query,
            "user_id": user_id,
            "return_k": return_k,
            "docs_returned": len(docs),
            "doc_ids": [doc.id for doc in docs],
            "returned_doc_chars": sum(len(doc.content or "") for doc in docs),
            "bridge_wall_time_ms": raw.get("bridge_wall_time_ms"),
            # `inputTokens`/`outputTokens` cover every LLM call the bridge made.
            # In retrieve-only mode (the default since 2026-06) that is the
            # probe/recall/synthesis pipeline; the legacy internal answer call
            # only appears when CSM_AMB_WITH_INTERNAL_ANSWER=1, so the
            # csm_internal_answer_* fields are null otherwise.
            "csm_internal_input_tokens": raw.get("inputTokens"),
            "csm_internal_output_tokens": raw.get("outputTokens"),
            "csm_internal_total_tokens": (
                _num(raw.get("inputTokens")) + _num(raw.get("outputTokens"))
            ),
            "csm_pipeline_input_tokens": meta.get("pipelineInputTokens"),
            "csm_pipeline_output_tokens": meta.get("pipelineOutputTokens"),
            "csm_pipeline_latency_ms": meta.get("pipelineLatencyMs"),
            "csm_internal_answer_input_tokens": meta.get("finalCallInputTokens"),
            "csm_internal_answer_output_tokens": meta.get("finalCallOutputTokens"),
            "csm_internal_answer_latency_ms": meta.get("finalCallLatencyMs"),
            "csm_probe_count": meta.get("probeCount"),
            "csm_recall_count": meta.get("recallCount"),
            "csm_context_tokens_before_amb_capsule": meta.get("contextTokens"),
            "csm_packet_tokens": meta.get("packetTokens"),
            "csm_retrieved_event_count": len(meta.get("csmRetrievedEventIds") or []),
            "csm_packed_event_count": len(meta.get("packedEventIds") or []),
            "csm_returned_event_count": len(raw.get("returnedEventIds") or []),
            "csm_evidence_capsule": raw.get("evidenceCapsule"),
            "amb_intent": raw.get("ambIntent"),
        }

        path = Path(telemetry_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def _num(value) -> float:
    return value if isinstance(value, (int, float)) else 0
