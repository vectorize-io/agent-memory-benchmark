"""supportbench: the scenario holds together and each arm routes every document where it says, no network."""


import pytest

from memory_bench.dataset import REGISTRY, supportbench
from memory_bench.memory import REGISTRY as MEMORY
from memory_bench.memory.hindsight_support import query_usd
from memory_bench.models import QueryResult


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    # Stand-in manuals: the real PDFs are a 40 MB download.
    fake = [tmp_path / m["filename"] for m in supportbench.scenario()["manuals"]]
    for p in fake:
        p.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(supportbench, "manual_paths", lambda: fake)
    return REGISTRY["supportbench"]()


def test_scenario_references_resolve(dataset):
    s = supportbench.scenario()
    known = {m["id"] for m in s["manuals"]} | {e["id"] for e in s["exceptions"]}
    known |= {c["id"] for u in s["users"].values() for c in u["chats"]}
    for q in s["questions"]:
        assert q["user"] in s["users"], q["id"]
        assert q["category"] in supportbench.CATEGORIES, q["id"]
        for src in q["sources"]:
            assert src.split(" ")[0] in known, f"{q['id']}: unknown source {src}"
    # Every category has at least two questions, and an exception pair is one question asked by two customers.
    for cat in supportbench.CATEGORIES:
        assert sum(q["category"] == cat for q in s["questions"]) >= 2, cat
    pairs: dict[str, list[dict]] = {}
    for q in s["questions"]:
        if q["category"] == "exception":
            pairs.setdefault(q["pair"], []).append(q)
    for pair, qs in pairs.items():
        assert len(qs) == 2 and qs[0]["question"] == qs[1]["question"] and qs[0]["user"] != qs[1]["user"], pair


def test_documents_by_kind(dataset):
    docs = dataset.load_documents("nextcloud")
    manuals = [d for d in docs if d.path]
    assert len(manuals) == 2 and all(d.user_id is None for d in manuals)
    assert all(d.user_id for d in docs if not d.path)
    # Every document's context ends in its id; an exception note's context never names its customer.
    names = [u["name"] for u in supportbench.scenario()["users"].values()]
    for d in docs:
        assert d.context.endswith(f"[{d.id}]"), d.id
        if not d.path and not d.messages:
            assert not any(n in d.context or n in d.content for n in names), d.id
    # A user filter keeps the shared manuals and drops everyone else's chats.
    only = dataset.load_documents("nextcloud", user_ids={"marta"})
    assert {d.user_id for d in only} == {None, "marta"}


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("hindsight-support-memory", {"manual": "memory", "chat": "memory", "exception": "memory"}),
        ("hindsight-support-knowledge", {"manual": "knowledge", "chat": "knowledge", "exception": "knowledge"}),
        ("hindsight-support-linked", {"manual": "knowledge", "chat": "memory", "exception": "knowledge"}),
    ],
)
def test_arm_routing(dataset, provider, expected):
    arm = MEMORY[provider]()
    for doc in dataset.load_documents("nextcloud"):
        kind = "manual" if doc.path else "chat" if doc.messages else "exception"
        assert arm._side(doc) == expected[kind], (provider, doc.id)


def test_query_usd_prices_cached_and_reasoning():
    usage = {"input_tokens": 1_000_000, "cached_tokens": 0, "output_tokens": 500_000, "thoughts_tokens": 500_000}
    assert query_usd(usage, "gemma-4-26b-a4b") == pytest.approx(0.024 + 0.10)
    assert query_usd(usage, "unpriced-model") is None
    assert query_usd(None) is None


def _row(qid: str, category: str, correct: bool, reason: str, usd: float | None) -> QueryResult:
    return QueryResult(query_id=qid, query="", answer="", reasoning="", context="", context_tokens=0,
                       retrieve_time_ms=2000.0, gold_answers=[], correct=correct, judge_reason=reason,
                       meta={"category": category, "query_usd": usd, "ingest_usd": 1.5, "ingest_s": 300.0})


def test_summary_counts_traps_and_cost(dataset):
    rows = [_row("a", "conflict", False, "TRAP: repeats 10 MiB", 0.001), _row("b", "conflict", True, "ok", 0.003),
            _row("c", "docs_only", True, "ok", 0.002)]
    m = dataset.summary_metrics(rows)
    assert m["trap_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["accuracy_conflict"] == 0.5 and m["accuracy_docs_only"] == 1.0
    assert m["avg_query_usd"] == pytest.approx(0.002) and m["ingest_usd"] == 1.5 and m["ingest_s"] == 300.0 and m["avg_answer_s"] == 2.0


def test_judge_prompt_carries_the_trap(dataset):
    q = dataset.load_queries("nextcloud", category="conflict")[0]
    prompt = dataset.get_judge_prompt_fn(q.meta["category"], q.meta)(q.query, q.gold_answers, "an answer")
    assert q.meta["must_not_claim"] in prompt and "TRAP:" in prompt


def test_every_document_names_its_source(dataset):
    from memory_bench.memory.hindsight_support import SOURCE_MISSION, source_of

    sources = {source_of(d) for d in dataset.load_documents("nextcloud")}
    assert sources == {"documentation", "conversation", "approved_exception"}
    # The mission ranks exactly the sources the documents carry.
    assert all(f"`{s}`" in SOURCE_MISSION for s in sources)
