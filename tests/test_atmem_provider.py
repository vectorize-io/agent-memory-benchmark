from memory_bench.memory.atmem import AtMemMemoryProvider
from memory_bench.models import Document


def test_atmem_provider_retrieves_scoped_memory_with_source_evidence(tmp_path):
    provider = AtMemMemoryProvider()
    provider.prepare(tmp_path)
    try:
        provider.ingest(
            [
                Document(
                    id="alice-color",
                    content="My favorite color is teal.",
                    user_id="alice",
                ),
                Document(
                    id="bob-color",
                    content="My favorite color is orange.",
                    user_id="bob",
                ),
            ]
        )

        documents, raw = provider.retrieve(
            "What is my favorite color?", k=5, user_id="alice"
        )
    finally:
        provider.cleanup()

    assert len(documents) == 1
    assert "teal" in documents[0].content
    assert "orange" not in documents[0].content
    assert documents[0].source_ids == ["alice-color"]
    assert raw["returned_ids"] == [documents[0].id]
    assert raw["retrieval_id"].startswith("ret_")
    assert raw["decision"]["support_class"] == "direct_support"


def test_atmem_provider_formats_structured_messages():
    document = Document(
        id="session-1",
        content="fallback",
        timestamp="2026-09-10T00:00:00Z",
        messages=[
            {"role": "user", "content": "I prefer window seats."},
            {"role": "assistant", "content": "Understood."},
        ],
    )

    assert AtMemMemoryProvider._format_content(document) == (
        "Date: 2026-09-10T00:00:00Z\n"
        "User: I prefer window seats.\n"
        "Assistant: Understood."
    )
