"""Unit tests for the Evolving Profile (EP 6.2+) AMB provider."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_bench.memory import REGISTRY, get_memory_provider
from memory_bench.memory.evolving_profile import EvolvingProfileMemoryProvider
from memory_bench.models import Document


def test_registry_registers_evolving_profile() -> None:
    assert "evolving-profile" in REGISTRY
    provider = get_memory_provider("evolving-profile")
    assert isinstance(provider, EvolvingProfileMemoryProvider)
    assert provider.name == "evolving-profile"
    assert provider.kind == "local"
    assert provider.supports_filters is False


def test_ingest_retrieve_readback_and_cleanup(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    stored_docs: dict[str, str] = {}

    def fake_http_json(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        calls.append((method, url, payload))
        if method == "POST" and url.endswith("/memories"):
            for item in (payload or {}).get("items", []):
                stored_docs[item["document_id"]] = item["content"]
            return {"status": "ok"}
        if method == "POST" and url.endswith("/memories/recall"):
            return {
                "results": [
                    {"id": "mem-2", "text": "User switched to espresso", "document_id": "doc-2"},
                    {"id": "mem-1", "text": "User liked green tea", "document_id": "doc-1"},
                ]
            }
        if method == "GET" and "/documents/" in url:
            did = url.rsplit("/documents/", 1)[-1]
            return {"original_text": stored_docs.get(did, "")}
        return {}

    monkeypatch.setattr("memory_bench.memory.evolving_profile._http_json", fake_http_json)

    provider = EvolvingProfileMemoryProvider(api_url="http://127.0.0.1:12088", bank_prefix="test-ep")
    provider.prepare(Path("/tmp/amb-test"), unit_ids={"user-a"}, reset=True)
    provider.ingest(
        [
            Document(
                id="doc-1",
                content="",
                user_id="user-a",
                timestamp="2026-01-01T10:00:00Z",
                messages=[{"role": "user", "content": "I drink green tea every morning."}],
            ),
            Document(
                id="doc-2",
                content="Updated preference: I switched to espresso in March.",
                user_id="user-a",
                timestamp="2026-03-01T10:00:00Z",
                context="preference update",
            ),
        ]
    )

    docs, meta = provider.retrieve("What does the user drink in the morning?", k=5, user_id="user-a")
    assert meta is not None
    assert meta["bank_id"] == "test-ep-user-a"
    assert meta["readback_docs"] == 2
    # Chronological readback places doc-1 before doc-2
    assert [d.id for d in docs[:2]] == ["doc-1", "doc-2"]
    assert docs[0].source_ids == ["doc-1"]
    assert "[Date: 2026-01-01T10:00:00Z]" in docs[0].content
    assert "User: I drink green tea every morning." in docs[0].content

    provider.cleanup()
    delete_urls = [url for method, url, _ in calls if method == "DELETE"]
    assert any(url.endswith("/v1/default/banks/test-ep-user-a") for url in delete_urls)
