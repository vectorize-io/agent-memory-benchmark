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
import os
import subprocess
from pathlib import Path

from .base import Dataset
from ..models import Document, Query

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS = _REPO_ROOT / "sdebench" / "datasets"

# boltons host clone (the retrieval-noise corpus), pinned at the fork ref used by every task.
_BOLTONS_HOST = Path(os.environ.get("SDEBENCH_BOLTONS_HOST") or (Path.home() / "dev" / "_sdebench_hosts" / "boltons"))
_BOLTONS_REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"
_GIT_DOCS = int(os.environ.get("SDEBENCH_GIT_DOCS", "400"))  # how many recent commits to ingest as noise


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
            if category and cat != category:
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
                    "task_json": str(tj),  # the CodingMode passes this to run.py --task
                },
            ))
        return queries[:limit] if limit else queries

    def load_documents(self, split: str, category: str | None = None, limit: int | None = None,
                       ids: set[str] | None = None, user_ids: set[str] | None = None) -> list[Document]:
        """The coding memory: each task's developer chat (where the F decisions live) + the boltons
        git history (retrieval noise + the H decision's rationale). The OMB memory provider ingests
        these; the CodingMode then reflects/retrieves over them per task. Chats come FIRST so a small
        --doc-limit keeps the decisive documents. user_id is None => one shared bank (ranking under noise).

        Note: omdset's H decision is a commit planted by its build.py into the *built* repo, not the
        host clone, so it is not yet in this corpus (a per-task planted-commit pass is a follow-up)."""
        docs: list[Document] = []
        for tj in self._task_files():
            t = json.loads(tj.read_text())
            conv = t.get("conversations")
            if conv:
                text = "\n".join(f"{c['role'].upper()}: {c['text']}" for c in conv)
                docs.append(Document(id=f"chat:{t['task_id']}", content=text,
                                     context=f"developer conversation about {t['codebase']}"))
        docs += self._git_documents(limit=_GIT_DOCS)
        return docs[:limit] if limit else docs

    def _git_documents(self, limit: int) -> list[Document]:
        if not _BOLTONS_HOST.exists() or limit <= 0:
            return []
        US, RS = "\x1f", "\x1e"
        out = subprocess.run(
            ["git", "-C", str(_BOLTONS_HOST), "log", f"-n{limit}",
             f"--format=%H{US}%aI{US}%s{US}%b{RS}", _BOLTONS_REF],
            capture_output=True, text=True,
        ).stdout
        docs: list[Document] = []
        for rec in out.split(RS):
            rec = rec.strip()
            if not rec:
                continue
            parts = rec.split(US)
            if len(parts) < 3:
                continue
            sha, aiso, subj = parts[0], parts[1], parts[2]
            body = parts[3] if len(parts) > 3 else ""   # commits with no body yield 3 fields
            content = f"git commit {sha[:12]}\n{subj}"
            if body.strip():
                content += "\n\n" + body.strip()
            docs.append(Document(id=f"git:{sha[:12]}", content=content, timestamp=aiso or None,
                                 context="boltons git history"))
        return docs

    def categories(self, split: str) -> list[str] | None:
        cats = {json.loads(tj.read_text()).get("category") for tj in self._task_files()}
        return sorted(c for c in cats if c)

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
