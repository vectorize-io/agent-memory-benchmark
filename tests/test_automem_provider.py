import io, json
from contextlib import contextmanager
from memory_bench.memory.automem import (
    _chunk_text, _extract_content, _free_port, AutoMemMemoryProvider,
)
from memory_bench.models import Document
import memory_bench.memory.automem as m


def test_chunk_long_text_under_limit():
    text = " ".join(f"sentence {i}." for i in range(600))
    assert len(text) > 1800
    chunks = _chunk_text(text, 1800)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 for c in chunks)

def test_chunk_short_text_single():
    assert _chunk_text("hello world", 1800) == ["hello world"]

def test_extract_content_from_nested_memory():
    assert _extract_content({"id": "m1", "memory": {"content": "answer", "summary": "s"}}) == "answer"

def test_extract_content_falls_back_to_summary():
    assert _extract_content({"id": "m1", "memory": {"content": "", "summary": "fallback"}}) == "fallback"

def test_extract_content_empty_when_missing():
    assert _extract_content({"id": "m1"}) == ""

def test_free_port_returns_open_port():
    p = _free_port()
    assert 1024 < p < 65536


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = responses; self.calls = []
    @contextmanager
    def urlopen(self, req, timeout=None):
        body = req.data.decode() if req.data else None
        self.calls.append((req.get_method(), req.full_url, dict(req.headers), body))
        payload = self.responses.pop(0) if self.responses else {}
        yield io.BytesIO(json.dumps(payload).encode())


def test_ingest_batches_chunks(monkeypatch):
    fake = _FakeHTTP([{"stored": 99} for _ in range(10)])
    monkeypatch.setattr(m.urllib.request, "urlopen", fake.urlopen)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"; p._run_tag = "ambrun-test"
    p._enrich_settle_s = 0; p._enrich_max_pending = 0
    long_doc = Document(id="d", content=" ".join(f"s{i}." for i in range(600)), user_id="u1")
    p.ingest([long_doc])
    batches = [c for c in fake.calls if c[1].endswith("/memory/batch")]
    assert len(batches) >= 1  # chunks go out as a batch, not one-by-one
    items = json.loads(batches[0][3])["memories"]
    assert len(items) > 1
    for it in items:
        assert len(it["content"]) <= 1800
        assert "ambrun-test" in it["tags"]

def test_retrieve_extracts_nested_content(monkeypatch):
    fake = _FakeHTTP([{"results": [{"id": "m9", "memory": {"content": "answer"}}]}])
    monkeypatch.setattr(m.urllib.request, "urlopen", fake.urlopen)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"; p._run_tag = "ambrun-test"
    docs, _ = p.retrieve("q", k=5, user_id="u1")
    assert docs[0].content == "answer"
    assert "expand_relations=true" in fake.calls[0][1]


def test_req_retries_remote_disconnected(monkeypatch):
    import http.client
    calls = {"n": 0}
    class _OK:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'
    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection")
        return _OK()
    monkeypatch.setattr(m.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"
    assert p._req("GET", "/health") == {"ok": 1}
    assert calls["n"] == 2  # retried past the disconnect


def test_req_does_not_retry_http_error(monkeypatch):
    import urllib.error
    calls = {"n": 0}
    def boom(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)
    monkeypatch.setattr(m.urllib.request, "urlopen", boom)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"
    try:
        p._req("POST", "/memory", body={"content": "x"})
        raised = False
    except urllib.error.HTTPError:
        raised = True
    assert raised and calls["n"] == 1  # 400 raised immediately, not retried
