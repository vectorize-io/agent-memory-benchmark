"""Knowledge-bank retrieval: public corpora with published retriever leaderboards, scored on
the ranking a search returns. Search only, no answer LLM and no judge.

Run with ``--mode retrieval`` against a provider that writes files and text into a knowledge
bank (``hindsight-knowledge``):

| dataset                | splits                         | score (``score_type``)        |
|------------------------|--------------------------------|-------------------------------|
| ``vidore-v3-markdown`` | 8 ViDoRe V3 tasks, page text   | nDCG@10, all 6 query languages |
| ``vidore-v3-pdf``      | 8 ViDoRe V3 tasks, whole PDFs  | nDCG@10                        |
| ``miracl-vision``      | 18 languages, page images      | nDCG@10                        |
| ``video-retrieval``    | MSR-VTT 1k-A, DiDeMo, YouCook2 | nDCG@10                        |
| ``browsecomp-plus``    | evidence / gold judgments      | nDCG@10                        |
| ``atm-bench-retrieval``| photos + videos + emails       | recall@10 (the board's R@10)   |
| ``legalbench-rag``     | mini: 776 questions, 714 contracts | character recall@5         |

A split is one corpus (one bank). The corpus is always whole — a smaller pool makes every
query easier — and ``--query-limit`` only cuts queries: per query language on ViDoRe (the
first N of each language are the same N questions translated), the first N elsewhere.

Each query's headline metric is its ``score`` and the run's ``accuracy`` field holds their
mean, labelled by ``score_type`` everywhere it is shown. Leaderboards that publish one number
over several splits (ViDoRe's 8 tasks, MIRACL's 18 languages, the 3 video sets) compare to
the mean over the splits, which the viewer shows as the ``average`` split.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from . import knowledge_data as data
from .base import Dataset
from ..models import Document, Query

#: Hits asked of the search: the knowledge-bank API's ceiling. Several passages of one document
#: can come back, so the distinct-document list is shorter and recall@100 there is a floor.
RANKED_TOP_K = 200
#: Ranked ids kept per query in a result file (the per-query drill-down).
KEPT_RANKING = 100
RECALL_CUTOFFS = (1, 5, 10, 100)


# --- metrics (trec_eval's definitions, as MTEB / ViDoRe / BrowseComp-Plus report them) ----------


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """nDCG@k with trec_eval's LINEAR gain and ``log2(rank + 1)`` discount (graded qrels)."""
    gain = sum(relevant.get(d, 0) / math.log2(rank + 2) for rank, d in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(rank + 2) for rank, g in enumerate(sorted(relevant.values(), reverse=True)[:k]))
    return gain / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of the relevant documents in the top k."""
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0) / len(relevant)


def ranked_documents(hits: Sequence[Document], page_ids: Mapping[str, Sequence[str]] | None = None) -> list[str]:
    """Each judged id once, at the rank of its best passage (max-sim). With ``page_ids``, a hit on
    page n of a multi-page document stands for that page's id; a document with one id and no
    pages (ATM-Bench's ``<item>~text`` records) stands for that id."""

    def judged(hit: Document) -> str:
        pages = (page_ids or {}).get(hit.id)
        return pages[min((hit.page or 1), len(pages)) - 1] if pages else hit.id

    return list(dict.fromkeys(judged(h) for h in hits))


def query_metrics(ranked: Sequence[str], relevant: Mapping[str, int]) -> dict[str, float]:
    """One query's metrics, 0-1."""
    out = {"ndcg@10": ndcg_at_k(ranked, relevant, 10)}
    out |= {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in RECALL_CUTOFFS}
    return out


# --- datasets -------------------------------------------------------------------------------------


class KnowledgeDataset(Dataset):
    """A ranked-retrieval corpus per split. Subclasses name the splits and the loader."""

    task_type = "retrieval"
    score_type = "ndcg@10"
    #: The split name the viewer shows the mean over all splits under (None: no cross-split number).
    average_split: str | None = None
    #: The result axis a query's group is shown under (ViDoRe: its query language), or None.
    group_axis: str | None = None
    scoring_note: str = ""

    def _load(self, split: str) -> data.RankedSuite:
        raise NotImplementedError

    def suite(self, split: str) -> data.RankedSuite:
        """The split's corpus and queries, read once per dataset object."""
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}' for {self.name}. Available: {self.splits}")
        suites = self.__dict__.setdefault("_suites", {})
        if split not in suites:
            suites[split] = self._load(split)
        return suites[split]

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        taken: dict[str, int] = {}
        queries = []
        for q in self.suite(split).queries:
            if category is not None and q.group != category:
                continue  # the query's group (its language on ViDoRe); the corpus is never cut
            if limit is not None and taken.get(q.group, 0) >= limit:
                continue  # the limit is per group: per query language on ViDoRe
            taken[q.group] = taken.get(q.group, 0) + 1
            relevant = dict(q.relevant)
            queries.append(Query(
                id=q.id,
                query=q.text,
                gold_ids=list(relevant),
                gold_answers=list(relevant),
                meta={"split": split, "group": q.group, "relevant": relevant, "retrieval_limit": RANKED_TOP_K},
            ))
        return queries

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        suite = self.suite(split)
        docs = []
        for item in suite.items:
            # --oracle: keep a document when any id it is judged as is gold.
            if ids is not None and not ids.intersection(self._judged_ids(suite, item.id)):
                continue
            if isinstance(item, data.FileItem):
                docs.append(Document(id=item.id, content="", path=str(item.path), filename=item.path.name))
            else:
                docs.append(Document(id=item.id, content=item.text, filename=item.filename, tags=list(item.tags) or None))
        return docs[:limit] if limit is not None else docs

    @staticmethod
    def _judged_ids(suite: data.RankedSuite, document_id: str) -> Sequence[str]:
        return suite.page_ids.get(document_id) or (document_id,)

    def supports_oracle(self) -> bool:
        return True

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        return {self.group_axis: [meta["group"]]} if self.group_axis and meta.get("group") else {}

    def score_retrieval(self, query: Query, retrieved: list[Document]) -> tuple[bool, str, float]:
        """(hit@10, reason, the headline metric). Writes the ranking and every metric into
        ``query.meta``, which the result row carries, for the per-query view."""
        relevant = query.meta["relevant"]
        ranked = ranked_documents(retrieved, self.suite(query.meta["split"]).page_ids)
        metrics = query_metrics(ranked, relevant)
        query.meta["metrics"] = {k: round(v, 4) for k, v in metrics.items()}
        query.meta["ranking"] = ranked[:KEPT_RANKING]
        first = next((i + 1 for i, d in enumerate(ranked) if relevant.get(d, 0) > 0), None)
        hit = any(relevant.get(d, 0) > 0 for d in ranked[:10])
        found = sum(1 for d in ranked[:10] if relevant.get(d, 0) > 0)
        reason = (
            f"nDCG@10 {metrics['ndcg@10']:.3f} · {found}/{len(relevant)} relevant in the top 10 · "
            + (f"first relevant at rank {first}" if first else "no relevant document returned")
        )
        return hit, reason, metrics[self.score_type]

    def summary_metrics(self, results: list) -> dict:
        rows = [r.meta["metrics"] for r in results if (r.meta or {}).get("metrics")]
        if not rows:
            return {}
        return {"metrics": {k: round(sum(m[k] for m in rows) / len(rows), 4) for k in rows[0]}}

    def summarize_run(self, results: list, console) -> None:
        metrics = self.summary_metrics(results).get("metrics", {})
        if metrics:
            console.print("  ".join(f"[bold]{k}[/bold] {v * 100:.2f}" for k, v in metrics.items()))


_VIDORE_NOTE = (
    "nDCG@10 (trec_eval linear gain) over every query of all 6 query languages, against ViDoRe's "
    "page-level qrels. The average split is the mean over the 8 tasks, the number Mixedbread's "
    "table reports. Search only: no LLM, no reranker, no query rewriting."
)


class VidoreMarkdownDataset(KnowledgeDataset):
    name = "vidore-v3-markdown"
    description = "ViDoRe V3, 8 enterprise PDF tasks (EN + FR) as the page markdown the dataset ships, one text document per page; text-to-page search in 6 query languages."
    splits = list(data.VIDORE_TASKS)
    average_split = "average"
    group_axis = "Language"
    scoring_note = _VIDORE_NOTE
    links = [
        {"label": "Dataset", "url": "https://huggingface.co/collections/vidore/vidore-benchmark-v3"},
        {"label": "Mixedbread table", "url": "https://www.mixedbread.com/evals/vidore-v3-text"},
    ]

    def _load(self, split: str) -> data.RankedSuite:
        return data.vidore_markdown(split, queries_per_language=None)


class VidorePdfDataset(KnowledgeDataset):
    name = "vidore-v3-pdf"
    description = "ViDoRe V3, 8 enterprise PDF tasks as the raw source PDFs: the server reads and splits each file itself, and a hit on page n stands for that page."
    splits = list(data.VIDORE_TASKS)
    average_split = "average"
    group_axis = "Language"
    scoring_note = _VIDORE_NOTE + " Harder than the markdown version: nobody parses the pages for the server."
    links = [{"label": "Dataset", "url": "https://huggingface.co/collections/vidore/vidore-benchmark-v3"}]

    def _load(self, split: str) -> data.RankedSuite:
        return data.vidore_documents(split, queries_per_language=None)


class MiraclVisionDataset(KnowledgeDataset):
    name = "miracl-vision"
    description = "MIRACL-Vision: Wikipedia page images in 18 languages (338,734 images, 7,898 queries), each language searched on its own corpus. Images stored at up to 896 px."
    splits = list(data.MIRACL_VISION_LANGS)
    average_split = "average"
    scoring_note = "nDCG@10 per language against MIRACL's qrels; the average split is the mean over the 18 languages. Each image is written under its article title."
    links = [{"label": "Dataset", "url": "https://huggingface.co/datasets/nvidia/miracl-vision"}]

    def _load(self, split: str) -> data.RankedSuite:
        return data.miracl_vision(split, queries=None)


class VideoRetrievalDataset(KnowledgeDataset):
    name = "video-retrieval"
    description = "Text to video on MSR-VTT 1k-A test (1,000 clips), DiDeMo test (1,004 videos, paragraph queries) and YouCook2 val (3,179 recipe-step clips)."
    splits = ["msrvtt", "didemo", "youcook2"]
    average_split = "average"
    scoring_note = "nDCG@10 with one relevant clip per caption; the average split is the mean over the three sets, the number Mixedbread publishes."
    links = [{"label": "Mixedbread evals", "url": "https://www.mixedbread.com/evals.md"}]

    def _load(self, split: str) -> data.RankedSuite:
        if split == "msrvtt":
            return data.captioned_videos("msrvtt_1k", captions_per_video=None)
        return data.video_clips(split)


class BrowseCompPlusDataset(KnowledgeDataset):
    name = "browsecomp-plus"
    description = "BrowseComp-Plus retriever track: a fixed corpus of 100,195 web pages and 830 research questions, scored on the official evidence or gold judgments."
    splits = ["evidence", "gold"]
    redact_query_text = True  # the dataset asks that its decrypted questions never be published
    scoring_note = "nDCG@10 over the official qrels (qrel_evidence.txt: pages needed to answer; qrel_golds.txt: pages that hold the answer). Query text is kept out of result files."
    links = [
        {"label": "Leaderboard", "url": "https://huggingface.co/spaces/Tevatron/BrowseComp-Plus"},
        {"label": "Dataset", "url": "https://huggingface.co/datasets/Tevatron/browsecomp-plus"},
    ]

    def _load(self, split: str) -> data.RankedSuite:
        return data.browsecomp_plus(split, queries=None)  # type: ignore[arg-type]


class AtmBenchRetrievalDataset(KnowledgeDataset):
    name = "atm-bench-retrieval"
    description = "ATM-Bench as search: 3,759 photos, 533 videos and 6,742 emails (four years of one person's data), 1,013 questions scored on their human-labelled evidence items."
    splits = ["atm-bench"]
    score_type = "recall@10"
    scoring_note = "R@10: the share of a question's evidence items in the top 10 results. Each photo and video goes in as the raw file and as the benchmark's own text record (time, place, caption, OCR, tags); a hit on either counts as the item."
    links = [
        {"label": "Leaderboard", "url": "https://atmbench.github.io/leaderboard.html"},
        {"label": "Dataset", "url": "https://huggingface.co/datasets/Jingbiao/ATM-Bench"},
    ]

    def _load(self, split: str) -> data.RankedSuite:
        return data.atm_bench(queries=None)


def text_spans(document: str, text: str) -> list[tuple[int, int]]:
    """Where a hit's text sits in its document. The search returns a run of passages joined by a
    blank line, each an exact slice of the document, without offsets; each one is looked up after
    the previous one, so a run lands on one place even in a contract that repeats itself."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in (p.strip() for p in text.split("\n\n")):
        if not piece:
            continue
        at = document.find(piece, cursor)
        if at < 0:
            at = document.find(piece)
        if at < 0:
            continue  # not a slice of the document: counted as unplaced in the query's meta
        spans.append((at, at + len(piece)))
        cursor = at + len(piece)
    return spans


def _merged(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _overlap(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


class LegalBenchRagDataset(KnowledgeDataset):
    name = "legalbench-rag"
    description = "LegalBench-RAG mini: 776 questions over 714 contracts (NDAs, M&A agreements, commercial contracts, privacy policies), scored on the exact characters lawyers marked."
    splits = ["mini"]
    score_type = "recall@5"
    group_axis = "Set"
    #: The hits scored, and the text they may weigh in all: the search API's default budget.
    TOP = 5
    MAX_TOKENS = 4096
    scoring_note = (
        "Character recall@5, as legalbenchrag/run_benchmark.py computes it: the share of the marked "
        "characters that the top 5 results' text covers. The 4 sets weigh the same (194 questions each). "
        "The search runs at the API's default budget of 4,096 tokens, so the 5 results carry about as much "
        "text as 5 chunks of a typical RAG setup. Precision (the share of returned text that is marked) is "
        "in each query's metrics. Documents go in with neutral names. Search only: no LLM."
    )
    links = [
        {"label": "Dataset", "url": "https://github.com/zeroentropy-ai/legalbenchrag"},
        {"label": "Conare table", "url": "https://conare.ai/benchmarks"},
    ]

    def _load(self, split: str) -> data.RankedSuite:
        return data.legalbench_rag(queries=None)

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        queries = super().load_queries(split, category, limit)
        for q in queries:
            q.meta["retrieval_limit"] = self.TOP
            q.meta["retrieval_filter"] = {"max_tokens": self.MAX_TOKENS}
        return queries

    def score_retrieval(self, query: Query, retrieved: list[Document]) -> tuple[bool, str, float]:
        suite = self.suite(query.meta["split"])
        texts = self.__dict__.setdefault("_texts", {i.id: i.text for i in suite.items})
        gold: dict[str, list[tuple[int, int]]] = {}
        for doc_id, start, end in suite.spans[query.id]:
            gold.setdefault(doc_id, []).append((start, end))
        found: dict[str, list[tuple[int, int]]] = {}
        returned = unplaced = 0
        for hit in retrieved[: self.TOP]:
            spans = text_spans(texts.get(hit.id, ""), hit.content or "")
            returned += sum(e - s for s, e in spans)
            unplaced += max(0, len((hit.content or "").replace("\n\n", "")) - sum(e - s for s, e in spans))
            found.setdefault(hit.id, []).extend(spans)
        found = {d: _merged(s) for d, s in found.items()}
        marked = sum(e - s for spans in gold.values() for s, e in spans)
        covered = sum(_overlap(gold[d], found.get(d, [])) for d in gold)
        recall = covered / marked if marked else 0.0
        precision = covered / sum(e - s for spans in found.values() for s, e in spans) if found else 0.0
        query.meta["metrics"] = {"recall@5": round(recall, 4), "precision@5": round(precision, 4)}
        query.meta["ranking"] = [h.id for h in retrieved[: self.TOP]]
        query.meta["returned_chars"] = returned
        query.meta["unplaced_chars"] = unplaced
        reason = f"{covered}/{marked} marked characters in the top {self.TOP} ({returned} characters returned)"
        return covered > 0, reason, recall


KNOWLEDGE_DATASETS: list[type[KnowledgeDataset]] = [
    VidoreMarkdownDataset,
    VidorePdfDataset,
    MiraclVisionDataset,
    VideoRetrievalDataset,
    BrowseCompPlusDataset,
    AtmBenchRetrievalDataset,
    LegalBenchRagDataset,
]
