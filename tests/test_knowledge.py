"""Knowledge-bank retrieval: scoring and score-type labelling, with stub corpora and no network."""

import json
import math
from pathlib import Path

import pytest

from memory_bench.dataset import REGISTRY
from memory_bench.dataset import knowledge_data as data
from memory_bench.dataset.knowledge import (
    LegalBenchRagDataset,
    text_spans,
    AtmBenchRetrievalDataset,
    KnowledgeDataset,
    VidorePdfDataset,
    ndcg_at_k,
    ranked_documents,
    recall_at_k,
)
from memory_bench.memory.base import MemoryProvider
from memory_bench.models import Document
from memory_bench.modes.retrieval import RetrievalMode
from memory_bench.runner import EvalRunner

ROOT = Path(__file__).parents[1]


def test_ndcg_uses_linear_gain_on_graded_qrels():
    # trec_eval: gain = grade, discount log2(rank + 1). Exponential gain (2^g - 1) would give 3 for grade 2.
    relevant = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)  # b at rank 1, a at rank 2
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], relevant, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b"], relevant, 10) == 1.0
    assert ndcg_at_k(["x"] * 10 + ["a"], relevant, 10) == 0.0


def test_recall_counts_relevant_in_the_top_k():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 3) == 1.0
    assert recall_at_k(["a"], {}, 10) == 0.0


def test_ranked_documents_dedupes_and_maps_pages_and_text_records():
    hits = [
        Document(id="report", content="", page=2),
        Document(id="report", content="", page=2),  # second passage of the same page
        Document(id="photo1~text", content=""),
        Document(id="report", content="", page=1),
        Document(id="photo1", content=""),
    ]
    page_ids = {"report": ("p1", "p2"), "photo1~text": ("photo1",)}
    assert ranked_documents(hits, page_ids) == ["p2", "photo1", "p1"]


class _StubDataset(KnowledgeDataset):
    name = "stub-knowledge"
    description = "stub"
    splits = ["s"]
    group_axis = "Language"

    def _load(self, split):
        return data.RankedSuite(
            name="stub",
            items=(
                data.TextItem(id="d1", text="one", filename="d1.md"),
                data.TextItem(id="d2", text="two", filename="d2.md"),
                data.FileItem(id="doc", path=Path("/nowhere/doc.pdf")),
            ),
            queries=(
                data.RankedQuery(id="en:1", text="q1", group="en", relevant={"d1": 1}),
                data.RankedQuery(id="en:2", text="q2", group="en", relevant={"d2": 1}),
                data.RankedQuery(id="fr:1", text="q1 fr", group="fr", relevant={"d1": 1}),
                data.RankedQuery(id="fr:2", text="q2 fr", group="fr", relevant={"p2": 1}),
            ),
            page_ids={"doc": ("p1", "p2")},
        )


def test_query_limit_is_per_group_and_oracle_keeps_judged_pages():
    ds = _StubDataset()
    assert [q.id for q in ds.load_queries("s", limit=1)] == ["en:1", "fr:1"]
    # --category keeps one group's queries (a ViDoRe language); the corpus stays whole.
    assert {q.meta["group"] for q in ds.load_queries("s", category="fr")} == {"fr"}
    assert len(ds.load_documents("s", category="fr")) == len(ds.load_documents("s"))
    docs = ds.load_documents("s", ids={"p2"})  # a gold page keeps the whole PDF it is in
    assert [(d.id, d.path, d.filename) for d in docs] == [("doc", "/nowhere/doc.pdf", "doc.pdf")]


def test_score_retrieval_returns_the_headline_metric_and_records_the_ranking():
    ds = _StubDataset()
    q = ds.load_queries("s")[3]  # relevant: page 2 of "doc"
    hit, reason, score = ds.score_retrieval(q, [Document(id="d1", content=""), Document(id="doc", content="", page=2)])
    assert hit is True
    assert score == pytest.approx(1 / math.log2(3))
    assert q.meta["ranking"] == ["d1", "p2"]
    assert q.meta["metrics"]["recall@1"] == 0.0 and q.meta["metrics"]["recall@10"] == 1.0
    assert "rank 2" in reason


def test_atm_bench_scores_recall_at_10():
    ds = AtmBenchRetrievalDataset()
    ds.__dict__["_suites"] = {"atm-bench": data.RankedSuite(
        name="atm", items=(), queries=(data.RankedQuery(id="q", text="t", group="all", relevant={"a": 1, "b": 1}),),
        page_ids={"a~text": ("a",)},
    )}
    q = ds.load_queries("atm-bench")[0]
    _, _, score = ds.score_retrieval(q, [Document(id="a~text", content="")] + [Document(id=f"x{i}", content="") for i in range(20)])
    assert score == 0.5  # one of the two evidence items in the top 10, via its text record


def test_legalbench_scores_marked_characters_in_the_top_5_hits():
    doc = "Intro.\n\nClause A says x.\n\nClause B says y.\n\nClause A says x."
    ds = LegalBenchRagDataset()
    gold_b = (doc.index("Clause B"), doc.index("Clause B") + len("Clause B says y."))
    ds.__dict__["_suites"] = {"mini": data.RankedSuite(
        name="lb", items=(data.TextItem(id="c", text=doc, filename="c.txt"),),
        queries=(data.RankedQuery(id="q", text="t", group="cuad", relevant={"c": 1}),),
        spans={"q": (("c", *gold_b),)},
    )}
    q = ds.load_queries("mini")[0]
    assert q.meta["retrieval_limit"] == 5 and q.meta["retrieval_filter"] == {"max_tokens": 4096}
    # A run of passages joined by a blank line lands on one place, the second "Clause A" after B.
    assert text_spans(doc, "Clause B says y.\n\nClause A says x.") == [gold_b, (doc.rindex("Clause A"), len(doc))]
    hit, _, recall = ds.score_retrieval(q, [Document(id="c", content="Clause B says y.\n\nClause A says x.")])
    assert hit and recall == 1.0
    assert q.meta["metrics"]["precision@5"] == 0.5 and q.meta["unplaced_chars"] == 0
    _, _, recall = ds.score_retrieval(q, [Document(id="c", content="Intro.")] * 5 + [Document(id="c", content="Clause B says y.")])
    assert recall == 0.0  # the 6th hit is not scored


class _StubSearch(MemoryProvider):
    """Returns d1 first for every query: query 1 and its translation are perfect, query 2 misses."""

    name = "stub-search"
    description = "stub"
    kind = "local"

    def ingest(self, documents):
        self.ingested = documents

    def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
        return [Document(id="d1", content=""), Document(id="d2", content="")][:k], None


def test_a_run_saves_score_type_per_query_scores_and_rankings(tmp_path):
    ds = _StubDataset()
    summary = EvalRunner(output_dir=tmp_path).run(dataset=ds, split="s", memory=_StubSearch(), mode=RetrievalMode())
    saved = json.loads((tmp_path / "stub-knowledge" / "stub-search" / "retrieval" / "s.json").read_text())
    assert summary.score_type == saved["score_type"] == "ndcg@10"
    scores = {r["query_id"]: r["score"] for r in saved["results"]}
    assert scores["en:1"] == 1.0 and scores["en:2"] == pytest.approx(1 / math.log2(3)) and scores["fr:2"] == 0.0
    assert saved["accuracy"] == pytest.approx(sum(scores.values()) / 4)  # the headline is the mean nDCG@10
    assert saved["metrics"]["ndcg@10"] == pytest.approx(saved["accuracy"], abs=1e-4)
    row = next(r for r in saved["results"] if r["query_id"] == "en:1")
    assert row["meta"]["ranking"] == ["d1", "d2"] and row["category_axes"] == {"Language": ["en"]}
    assert saved["judge_llm"] is None and saved["answer_llm"] is None
    # The manifest the viewer reads picks the score type up from the head of the file.
    head = json.dumps(saved, indent=2)[:4096]
    assert '"score_type": "ndcg@10"' in head


def test_redacted_datasets_keep_query_text_out_of_results(tmp_path):
    class Redacted(_StubDataset):
        redact_query_text = True

    EvalRunner(output_dir=tmp_path).run(dataset=Redacted(), split="s", memory=_StubSearch(), mode=RetrievalMode())
    saved = json.loads((tmp_path / "stub-knowledge" / "stub-search" / "retrieval" / "s.json").read_text())
    assert {r["query"] for r in saved["results"]} == {""}


def test_every_external_row_names_its_dataset_score_type():
    external = json.loads((ROOT / "external_results.json").read_text())
    for name, cls in REGISTRY.items():
        for split, rows in external.get(name, {}).items():
            assert split in cls.splits or split == getattr(cls, "average_split", None), (name, split)
            for row in rows:
                assert row.get("score_type", "accuracy") == cls.score_type, (name, split, row["memory"])
                assert row["source_url"].startswith("https://"), (name, row["memory"])


def test_catalog_labels_knowledge_datasets_with_their_score_type():
    catalog = json.loads((ROOT / "catalog.json").read_text())["datasets"]
    for name, cls in REGISTRY.items():
        if issubclass(cls, KnowledgeDataset):
            entry = catalog[name]
            assert entry["score_type"] == cls.score_type and entry["section"] == "knowledge", name
            assert entry["task"] == "Search only"
    assert catalog["vidore-v3-markdown"]["average_split"] == "average"
    assert catalog["atm-bench-retrieval"]["score_type"] == "recall@10"
    assert catalog["locomo"]["score_type"] == "accuracy"
    assert VidorePdfDataset.score_type == "ndcg@10"
