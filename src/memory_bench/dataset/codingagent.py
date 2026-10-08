"""
Synthetic coding-agent memory dataset.

This small seed split tests whether a memory system helps with engineering
continuity across sessions: reusing prior decisions, avoiding stale facts, and
remembering CI/review constraints without loading a whole repo into context.
"""
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .base import Dataset
from ..models import Document, Query

SPLITS = ["synthetic"]

_CATEGORIES = [
    "ci_failure_recurrence",
    "review_preference",
    "release_or_docs_sync",
]


class CodingAgentDataset(Dataset):
    """Small synthetic dataset for coding-agent memory behavior."""

    name = "codingagent"
    description = "Synthetic multi-session engineering traces for coding-agent memory."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "case"
    links = [
        {"label": "Hindsight issue", "url": "https://github.com/vectorize-io/hindsight/issues/2347"},
    ]

    def _data_path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unknown CodingAgent split '{split}'. Available: {SPLITS}")
        return Path(__file__).parents[3] / "data" / "codingagent" / split / "cases.json"

    def _load_cases(self, split: str) -> list[dict]:
        with open(self._data_path(split), encoding="utf-8") as f:
            return json.load(f)

    def categories(self, split: str) -> list[str] | None:
        return _CATEGORIES

    def category_type(self, split: str, category: str) -> str:
        return "query"

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        if category := meta.get("category"):
            axes["Category"] = [category]
        if case_id := meta.get("case_id"):
            axes["Case"] = [case_id]
        return axes

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        queries: list[Query] = []
        for case in self._load_cases(split):
            case_id = case["id"]
            for raw_query in case.get("queries", []):
                query_category = raw_query.get("category", case.get("category"))
                if category and query_category != category:
                    continue
                queries.append(Query(
                    id=f"{case_id}_{raw_query['id']}",
                    query=raw_query["query"],
                    gold_ids=[
                        f"{case_id}_{doc_id}"
                        for doc_id in raw_query.get("gold_document_ids", [])
                    ],
                    gold_answers=raw_query.get("gold_answers", []),
                    user_id=case_id,
                    meta={"case_id": case_id, "category": query_category},
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
        documents: list[Document] = []
        for case in self._load_cases(split):
            case_id = case["id"]
            if user_ids is not None and case_id not in user_ids:
                continue
            for raw_doc in case.get("documents", []):
                doc_id = f"{case_id}_{raw_doc['id']}"
                if ids is not None and doc_id not in ids:
                    continue
                documents.append(Document(
                    id=doc_id,
                    content=raw_doc["content"],
                    user_id=case_id,
                    timestamp=raw_doc.get("timestamp"),
                    context=f"codingagent:{case_id}",
                ))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="CodingAgent synthetic dataset stats")
        table.add_column("Split", style="bold")
        table.add_column("Cases", justify="right")
        table.add_column("Documents", justify="right")
        table.add_column("Queries", justify="right")
        table.add_column("Categories", justify="right")

        for split in SPLITS:
            cases = self._load_cases(split)
            documents = sum(len(case.get("documents", [])) for case in cases)
            queries = sum(len(case.get("queries", [])) for case in cases)
            categories = {
                query.get("category", case.get("category"))
                for case in cases
                for query in case.get("queries", [])
            }
            table.add_row(split, str(len(cases)), str(documents), str(queries), str(len(categories)))

        console.print(table)
