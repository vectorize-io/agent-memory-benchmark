"""sde-bench — a coding-agent benchmark, bound into the OMB runner.

Unlike the QA/recall datasets, each "query" is a bug-fix TASK: the agent must edit a real repo so the
hidden test passes. There are no gold answers — correctness is pytest pass/fail, produced by the
`coding` ResponseMode (which builds the repo, runs the agent with interventions, and grades). So
`task_type = "coding"`, `load_documents` is empty (the memory bank is prepared out of band by the
sdebench backfill for now), and scoring is handled by the runner's coding branch.

Tasks live in the `sde-bench` submodule at `sdebench/datasets/boltons-*`; the runner/harness is
`sdebench/harness/run.py`.
"""
import json
from pathlib import Path

from .base import Dataset
from ..models import Document, Query

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS = _REPO_ROOT / "sdebench" / "datasets"


class SdebenchDataset(Dataset):
    name = "sdebench"
    description = "Does memory help a coding agent? Bug-fix tasks whose obvious fix fails a hidden test."
    splits = ["boltons"]
    task_type = "coding"
    published = False
    links = [{"label": "Dataset", "url": "https://github.com/vectorize-io/sde-bench"}]

    def _task_files(self) -> list[Path]:
        return sorted(_DATASETS.glob("boltons-*/tasks/main/task.json"))

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        queries: list[Query] = []
        for tj in self._task_files():
            t = json.loads(tj.read_text())
            cat = t.get("category")
            # the PRIMARY AMB category is `source` (history vs conversation) — the benchmark's core
            # "where does the decision live" axis (2 values). The decision-type (`category`) and `tier`
            # remain as secondary breakdown axes via get_result_categories.
            if category and t.get("source") != category:
                continue
            queries.append(Query(
                id=t["task_id"],
                query=t["bug_report"],
                gold_ids=[],
                gold_answers=[],          # coding: no gold answer, graded by tests
                user_id=t["task_id"],
                meta={
                    "source": t.get("source"), "tier": t.get("tier"), "category": cat,
                    "codebase": t.get("codebase"), "module": t.get("module"),
                    "function": t.get("function"),  # used to focus the reflect query on the changed code
                    "task_json": str(tj),  # the CodingMode passes this to run.py --task
                },
            ))
        return queries[:limit] if limit else queries

    def load_documents(self, split: str, category: str | None = None, limit: int | None = None,
                       ids: set[str] | None = None, user_ids: set[str] | None = None) -> list[Document]:
        # AMB does NO ingestion for the coding task — memory is entirely the plugin's domain (the
        # coding mode triggers the plugin's own backfill, which decides what/how to ingest from the
        # built repo + the task's conversations). So there is nothing for the OMB provider to ingest.
        return []

    def categories(self, split: str) -> list[str] | None:
        # PRIMARY category = source (history / conversation) — 2 values, the benchmark's main axis.
        srcs = {json.loads(tj.read_text()).get("source") for tj in self._task_files()}
        return sorted(s for s in srcs if s)

    def category_type(self, split: str, category: str):
        return "query"

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        for key, label in (("source", "Source"), ("tier", "Tier"), ("category", "Category")):
            if meta.get(key):
                axes[label] = [meta[key]]
        return axes

    def supports_oracle(self) -> bool:
        return False
