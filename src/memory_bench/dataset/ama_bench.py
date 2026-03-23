"""
AMA-Bench dataset (https://huggingface.co/datasets/AMA-bench/AMA-bench).
Paper: https://arxiv.org/html/2602.22769v2 (ICLR 2026 Memory Agent Workshop)

Structure
---------
208 episodes × 12 QA pairs = 2,496 queries. Single "test" split.

Each episode contains:
  - trajectory: list of {turn_idx, action, observation}
  - qa_pairs:   list of {question, answer, question_uuid, type}

Documents = one per episode (flattened trajectory text).
Queries   = one per QA pair; gold_ids = [episode_id].
Categories = question type: A (Recall), B (Causal Inference),
             C (State Updating), D (State Abstraction).

Scoring: LLM-as-judge binary (yes/no) — matches official leaderboard methodology.
"""
from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

SPLITS = ["test"]

# Question-type categories (memory capability axis)
_QA_TYPES = ["A", "B", "C", "D"]
_QA_TYPE_LABELS = {
    "A": "Recall",
    "B": "Causal Inference",
    "C": "State Updating",
    "D": "State Abstraction",
}

# Domain categories (task axis)
_DOMAINS = ["EMBODIED_AI", "Game", "OPENWORLD_QA", "SOFTWARE", "TEXT2SQL", "WEB"]

# Both are valid category values
CATEGORIES = _QA_TYPES + _DOMAINS


def _format_trajectory(row: dict) -> str:
    """Flatten an episode into a plain-text document."""
    parts = [f"# Task\n{row['task']}",  "# Agent Trajectory"]
    for turn in row["trajectory"]:
        # Collapse blank lines within the observation so semantic chunkers
        # don't split the accessibility tree away from the step header.
        obs = (turn['observation'] or '').replace('\n\n', '\n')
        parts.append(
            f"Step {turn['turn_idx']}:\n"
            f"  Action: {turn['action']}\n"
            f"  Observation: {obs}"
        )
    return "\n\n".join(parts)


class AmaBenchDataset(Dataset):
    """
    AMA-Bench: long-horizon agent trajectory memory benchmark.
    Tests Recall, Causal Inference, State Updating, and State Abstraction.
    """

    name = "ama-bench"
    published = True
    description = "Agent trajectory memory: recall, causal inference, state tracking (ICLR 2026)."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "episode"
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2602.22769"},
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/AMA-bench/AMA-bench"},
    ]

    def _load_raw(self, split: str) -> list[dict]:
        from datasets import load_dataset
        cache = dataset_cache_dir("ama-bench")
        ds = load_dataset(
            "AMA-bench/AMA-bench",
            split=split,
            cache_dir=str(cache),
        )
        return list(ds)

    def categories(self, split: str) -> list[str] | None:
        return CATEGORIES

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes = {}
        if meta.get("qa_type"):
            axes["QA Type"] = [meta["qa_type"]]
        if meta.get("domain"):
            axes["Domain"] = [meta["domain"]]
        return axes

    def category_type(self, split: str, category: str) -> str:
        # QA types (A/B/C/D) are query-level filters — every episode has all types,
        # so per-category doc/token counts equal the full split and are meaningless.
        return "query" if category in _QA_TYPES else "doc"

    def default_judge_llm(self):
        from ..llm.openai import OpenAILLM
        return OpenAILLM("gpt-5.2")

    def get_judge_prompt_fn(self, category: str | None, meta: dict | None = None):
        """AMA-Bench aligned judge: uses episode_id for context (no task_description — causes false negatives)."""
        episode_id = (meta or {}).get("episode_id", "")

        def _judge(query: str, gold_answers: list[str], answer: str) -> str:
            gold_str = gold_answers[0] if gold_answers else ""
            context = f"Episode ID: {episode_id}" if episode_id else ""
            return f"""You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context}

Question: {query}

Reference Answer: {gold_str}

Predicted Answer: {answer}

Evaluate whether the predicted answer is correct.
Set correct=true if the predicted answer matches the reference answer factually.
Set correct=false if the predicted answer is wrong, incomplete, or contradicts the reference.
Minor wording differences and reasonable paraphrasing are fine."""

        return _judge

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        rows = self._load_raw(split)
        queries: list[Query] = []

        for row in rows:
            ep_id = str(row["episode_id"])
            domain = row.get("domain", "")
            task_type = row.get("task_type", "")

            for qa in row["qa_pairs"]:
                qtype = qa["type"]
                if category and category not in (qtype, domain):
                    continue
                queries.append(Query(
                    id=qa["question_uuid"],
                    query=qa["question"],
                    gold_ids=[ep_id],
                    gold_answers=[qa["answer"]],
                    user_id=ep_id,
                    meta={
                        "episode_id": ep_id,
                        "qa_type": qtype,
                        "qa_type_label": _QA_TYPE_LABELS.get(qtype, qtype),
                        "domain": domain,
                        "task_type": task_type,
                        "task": row.get("task", ""),
                    },
                ))

        if limit:
            queries = queries[:limit]
        return queries

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        rows = self._load_raw(split)
        documents: list[Document] = []

        # When a domain category is requested, filter episodes to that domain only.
        # QA-type categories (A/B/C/D) are query-level filters — every episode has all
        # types, so no document filtering is needed for those.
        domain_filter = category if category in _DOMAINS else None

        for row in rows:
            ep_id = str(row["episode_id"])
            if ids is not None and ep_id not in ids:
                continue
            if user_ids is not None and ep_id not in user_ids:
                continue
            if domain_filter and row.get("domain") != domain_filter:
                continue
            content = _format_trajectory(row)
            documents.append(Document(id=ep_id, content=content, user_id=ep_id))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def split_stats(self, split: str) -> dict:
        """Single-pass computation of full stats including per-category breakdown."""
        rows = self._load_raw(split)
        n_docs = len(rows)
        total_tokens = sum(r.get("total_tokens", 0) for r in rows)
        avg_tokens = total_tokens // n_docs if n_docs else 0

        # Per-category accumulators: queries, doc token sums, doc id sets
        cat_queries: dict[str, int] = {c: 0 for c in CATEGORIES}
        cat_tokens: dict[str, int] = {c: 0 for c in CATEGORIES}
        cat_docs: dict[str, set] = {c: set() for c in CATEGORIES}

        for row in rows:
            ep_id = str(row["episode_id"])
            domain = row.get("domain", "")
            ep_tokens = row.get("total_tokens", 0)
            for qa in row["qa_pairs"]:
                qtype = qa["type"]
                if qtype in cat_queries:
                    cat_queries[qtype] += 1
                    cat_docs[qtype].add(ep_id)
                    cat_tokens[qtype] = sum(
                        r.get("total_tokens", 0) for r in rows if str(r["episode_id"]) in cat_docs[qtype]
                    ) if ep_id not in cat_docs[qtype] else cat_tokens[qtype]
            if domain in cat_queries:
                cat_queries[domain] += len(row["qa_pairs"])
                cat_docs[domain].add(ep_id)
                cat_tokens[domain] += ep_tokens

        # Recompute type token totals cleanly (avoid per-qa recalculation above)
        ep_tokens_map = {str(r["episode_id"]): r.get("total_tokens", 0) for r in rows}
        for qtype in _QA_TYPES:
            cat_tokens[qtype] = sum(ep_tokens_map[eid] for eid in cat_docs[qtype])

        categories = {}
        for c in CATEGORIES:
            n = len(cat_docs[c])
            t = cat_tokens[c]
            cat_type = self.category_type(split, c)
            categories[c] = {
                "type": cat_type,
                "queries": cat_queries[c],
                "docs": n if cat_type == "doc" else None,
                "total_tokens": t if cat_type == "doc" else None,
                "avg_tokens_per_doc": (t // n if n else 0) if cat_type == "doc" else None,
            }

        return {
            "queries": sum(len(r["qa_pairs"]) for r in rows),
            "docs": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "categories": categories,
        }

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="AMA-Bench dataset stats")
        table.add_column("Split", style="bold")
        table.add_column("Episodes", justify="right")
        table.add_column("QA Pairs", justify="right")
        table.add_column("Avg turns", justify="right")
        table.add_column("Avg tokens", justify="right")

        try:
            rows = self._load_raw("test")
            n_ep = len(rows)
            n_qa = sum(len(r["qa_pairs"]) for r in rows)
            avg_turns = sum(r.get("num_turns", 0) for r in rows) // n_ep if n_ep else 0
            avg_tokens = sum(r.get("total_tokens", 0) for r in rows) // n_ep if n_ep else 0
            table.add_row("test", str(n_ep), f"{n_qa:,}", str(avg_turns), f"{avg_tokens:,}")
        except Exception as e:
            table.add_row("test", f"err: {e}", "", "", "")

        console.print(table)
