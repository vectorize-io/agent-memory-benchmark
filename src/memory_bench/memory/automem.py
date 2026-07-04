"""AutoMem provider for the Agent Memory Benchmark.

Self-spinning: initialize() brings up AutoMem (GHCR image) + FalkorDB + Qdrant via
docker compose with FastEmbed local embeddings (no API keys); cleanup() tears it down.
ingest() POSTs /memory (chunked, backdated); retrieve() GETs /recall and extracts
content from item["memory"]["content"] (top-level "content" is always empty).
"""
from __future__ import annotations
import atexit, http.client, json, os, re, socket, subprocess, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path
from ..models import Document
from .base import MemoryProvider

_COMPOSE = Path(__file__).parent / "automem_compose.yml"
_DEFAULT_IMAGE = "ghcr.io/verygoodplugins/automem:amb-v1"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:48] or "x"


def _chunk_text(text: str, max_chars: int) -> list:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, buf = [], ""
    for seg in re.split(r"(?<=[.!?])\s+|\n+", text):
        seg = seg.strip()
        if not seg:
            continue
        while len(seg) > max_chars:
            if buf:
                chunks.append(buf); buf = ""
            chunks.append(seg[:max_chars]); seg = seg[max_chars:]
        if not seg:
            continue
        if len(buf) + 1 + len(seg) <= max_chars:
            buf = f"{buf} {seg}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = seg
    if buf:
        chunks.append(buf)
    return chunks


def _extract_content(item: dict) -> str:
    mem = item.get("memory") if isinstance(item.get("memory"), dict) else {}
    return mem.get("content") or item.get("content") or mem.get("summary") or ""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class AutoMemMemoryProvider(MemoryProvider):
    name = "automem"
    description = "AutoMem graph+vector memory (FalkorDB+Qdrant) self-spun via Docker with FastEmbed; relations traversed on recall."
    kind = "local"
    provider = "automem"
    variant = "docker"
    link = "https://github.com/verygoodplugins/automem"
    concurrency = 4

    def __init__(self):
        self._image = os.environ.get("AUTOMEM_IMAGE", _DEFAULT_IMAGE)
        self._token = os.environ.get("AUTOMEM_TOKEN", "benchmark-token")
        self._max_chars = int(os.environ.get("AUTOMEM_MAX_CHARS", "1800"))
        self._batch_size = int(os.environ.get("AUTOMEM_BATCH_SIZE", "50"))  # /memory/batch, max 500
        self._k_override = os.environ.get("AUTOMEM_RECALL_K")
        self._enrich_settle_s = int(os.environ.get("AUTOMEM_ENRICH_SETTLE_SECONDS", "120"))
        # Backpressure during ingest so the async enrichment queue can't grow unbounded
        # and OOM-kill the container (set max_pending<=0 to disable).
        self._enrich_drain_every = int(os.environ.get("AUTOMEM_ENRICH_DRAIN_EVERY", "100"))
        self._enrich_max_pending = int(os.environ.get("AUTOMEM_ENRICH_MAX_PENDING", "150"))
        self._project = f"automem_amb_{uuid.uuid4().hex[:8]}"
        self._endpoint = None
        self._run_tag = f"ambrun-{uuid.uuid4().hex[:8]}"
        self._compose_env = None

    def initialize(self) -> None:
        # Honor pre-set ports if a caller pins them (e.g. to reach the stack from
        # another network namespace); otherwise pick free ones for concurrent runs.
        api = int(os.environ.get("AUTOMEM_API_PORT") or _free_port())
        falk = int(os.environ.get("AUTOMEM_FALKOR_PORT") or _free_port())
        qdr = int(os.environ.get("AUTOMEM_QDRANT_PORT") or _free_port())
        self._compose_env = {**os.environ, "AUTOMEM_IMAGE": self._image,
                             "AUTOMEM_API_PORT": str(api), "AUTOMEM_FALKOR_PORT": str(falk),
                             "AUTOMEM_QDRANT_PORT": str(qdr)}
        subprocess.run(["docker", "compose", "-p", self._project, "-f", str(_COMPOSE), "up", "-d"],
                       env=self._compose_env, check=True)
        # Ensure the stack is torn down even if the harness crashes before calling cleanup().
        atexit.register(self.cleanup)
        # Default localhost; override AUTOMEM_HOST to reach the stack on another host.
        self._endpoint = f"http://{os.environ.get('AUTOMEM_HOST', 'localhost')}:{api}"
        for _ in range(60):
            try:
                body = self._req("GET", "/health")
                if body.get("status") in {"ok", "healthy", "degraded"} and body.get("falkordb") == "connected":
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("AutoMem stack did not become healthy")

    def cleanup(self) -> None:
        if self._compose_env is not None:
            subprocess.run(["docker", "compose", "-p", self._project, "-f", str(_COMPOSE), "down", "-v"],
                           env=self._compose_env, check=False)

    def _req(self, method, path, *, params=None, body=None, _retries=5):
        url = f"{self._endpoint}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"X-Api-Key": self._token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        for attempt in range(_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError:
                # A real HTTP response (e.g. 400 for over-limit content) — never retry;
                # callers handle these (ingest skips 400s).
                raise
            except (urllib.error.URLError, http.client.RemoteDisconnected,
                    http.client.IncompleteRead, ConnectionError, TimeoutError, OSError):
                # AutoMem drops connections under ingest/enrichment load; retry with backoff
                # so a transient blip doesn't kill a multi-thousand-query run.
                if attempt == _retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 20))

    def _scoped_tags(self, user_id):
        tags = [self._run_tag]
        if user_id:
            tags.append(f"ambuser-{_slug(user_id)}")
        return tags

    def ingest(self, documents) -> None:
        # Flatten docs into chunked memory items, then POST /memory/batch (which
        # batch-embeds + UNWINDs graph writes). One-at-a-time POST /memory is ~0.8s/doc
        # on CPU FastEmbed; batching is the difference between hours and days.
        items = []
        for doc in documents:
            tags = self._scoped_tags(doc.user_id)
            pieces = _chunk_text(doc.content, self._max_chars)
            for i, piece in enumerate(pieces):
                meta = {"amb_doc_id": doc.id, "amb_user_id": doc.user_id, "amb_run": self._run_tag}
                if len(pieces) > 1:
                    meta["amb_chunk"] = i
                item = {"content": piece, "tags": tags, "importance": 0.6, "metadata": meta}
                if doc.timestamp:
                    item["timestamp"] = doc.timestamp
                items.append(item)
        posted = 0
        for start in range(0, len(items), self._batch_size):
            batch = items[start:start + self._batch_size]
            try:
                self._req("POST", "/memory/batch", body={"memories": batch})
                posted += len(batch)
            except urllib.error.HTTPError:
                # A bad item (e.g. over-limit) fails the whole batch; fall back to
                # per-item posts for this batch so one bad chunk doesn't drop the rest.
                for item in batch:
                    try:
                        self._req("POST", "/memory", body=item)
                        posted += 1
                    except urllib.error.HTTPError as exc:
                        if exc.code == 400:
                            continue
                        raise
            # Bound the async enrichment backlog (no-op when ENRICHMENT_ENABLED=false).
            if posted % self._enrich_drain_every < self._batch_size:
                self._bound_enrichment_queue()
        self._settle_enrichment()

    def _bound_enrichment_queue(self) -> None:
        if self._enrich_max_pending <= 0:
            return
        for _ in range(120):  # up to ~4 min of backpressure
            try:
                pending = self._req("GET", "/health").get("enrichment", {}).get("pending", 0)
            except Exception:
                return
            if pending <= self._enrich_max_pending:
                return
            time.sleep(2)

    def _settle_enrichment(self) -> None:
        if self._enrich_settle_s <= 0:
            return
        waited = 0
        while waited < self._enrich_settle_s:
            try:
                pending = self._req("GET", "/health").get("enrichment", {}).get("pending", 0)
            except Exception:
                pending = 0
            if not pending:
                return
            time.sleep(3); waited += 3

    def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
        k_eff = int(self._k_override) if self._k_override else k
        params = {"query": query, "limit": k_eff, "tags": self._scoped_tags(user_id),
                  "tag_mode": "all", "tag_match": "exact", "recency_bias": "auto",
                  "expand_relations": "true", "expand_respect_tags": "true"}
        resp = self._req("GET", "/recall", params=params)
        results = resp.get("results", []) if isinstance(resp, dict) else []
        docs = [Document(id=str(r.get("id") or ""), content=_extract_content(r)) for r in results]
        return docs, {"results": results, "count": len(results)}
