"""sde-bench — a coding-agent benchmark, bound into the OMB runner.

Unlike the QA/recall datasets, each "query" is a bug-fix TASK: the agent must edit a real repo so the
hidden test passes. There are no gold answers — correctness is pytest pass/fail, produced by the
`coding` ResponseMode (which builds the repo, runs the agent with interventions, and grades). So
`task_type = "coding"` and scoring is handled by the runner's coding branch.

Memory flows through the STANDARD pipeline: `load_documents` exposes each task's knowledge corpus
(the decision — a past developer chat or a documented git commit — plus decoy conversations and the
host repo's commit noise), isolated per task (`isolation_unit = "task"`, user_id = task_id). Any
registered MemoryProvider ingests it via the runner and serves the coding mode's generic
retrieve->inject path; the `hscoding` provider instead runs the Hindsight plugin's own deepen engine
over the BUILT repo (its ingest is repo-native, not document-list-based) and delivery happens
agent-side through the plugin.

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
_HOST = Path(os.environ.get("SDEBENCH_BOLTONS_HOST", "") or (Path.home() / "dev" / "_sdebench_hosts" / "boltons"))
_HOST_REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"
_NOISE_COMMITS = 100  # host-history noise window (matches the plugin's default gitlog scope order)


def task_json_path(task_id: str) -> Path:
    """Map a task_id (e.g. boltons-slalog-001) to its task.json. Convention: <codebase>-001."""
    codebase = task_id.rsplit("-", 1)[0]
    return _DATASETS / codebase / "tasks" / "main" / "task.json"


def _render_chat(turns: list[dict]) -> str:
    return "\n".join(f"{'Developer' if t.get('role', 'user') != 'assistant' else 'Assistant'}: "
                     f"{t.get('text', '')}" for t in turns)


def _host_noise_commits() -> list[tuple[str, str]]:
    """(sha, subject+body) for the last _NOISE_COMMITS host commits — the same retrieval noise every
    task's bank faces. Empty (with the corpus correspondingly thinner) when the host clone is absent."""
    if not (_HOST / ".git").exists():
        return []
    us = "\x1f"
    out = subprocess.run(["git", "-C", str(_HOST), "log", f"-n{_NOISE_COMMITS}",
                          f"--format=%h{us}%s%n%b{us}", _HOST_REF],
                         capture_output=True, text=True).stdout
    commits = []
    for chunk in out.split(us + "\n"):
        if us in chunk:
            sha, msg = chunk.split(us, 1)
            commits.append((sha.strip(), msg.strip()))
    return commits


class SdebenchDataset(Dataset):
    name = "sdebench"
    description = "Does memory help a coding agent? Bug-fix tasks whose obvious fix fails a hidden test."
    splits = ["boltons"]
    task_type = "coding"
    isolation_unit = "task"   # every task is an independent project: per-task banks/stores
    published = False
    links = [{"label": "Dataset", "url": "https://github.com/vectorize-io/sde-bench"}]

    def _task_files(self) -> list[Path]:
        files = sorted(_DATASETS.glob("boltons-*/tasks/main/task.json"))
        # SDE_TASK_FILTER: comma-separated substrings matched against the task dir name
        # (e.g. "slalog,unitparse" or "-amended") — for running a subset that isn't an
        # alphabetical prefix (-q N is first-N-alphabetically, not a sample).
        flt = os.environ.get("SDE_TASK_FILTER", "").strip()
        if flt:
            subs = [s.strip() for s in flt.split(",") if s.strip()]
            files = [f for f in files if any(s in f.parents[2].name for s in subs)]
        return files

    def get_isolation_id(self, doc: Document) -> str | None:
        return doc.user_id

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        queries: list[Query] = []
        for tj in self._task_files():
            t = json.loads(tj.read_text())
            cat = t.get("category")
            # the PRIMARY AMB category is `source` (history vs conversation) — the benchmark's core
            # "where does the decision live" axis. The decision-type (`category`) and `tier`
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
                    "function": t.get("function"),
                    "task_json": str(tj),  # the CodingMode passes this to run.py --task
                },
            ))
        return queries[:limit] if limit else queries

    def load_documents(self, split: str, category: str | None = None, limit: int | None = None,
                       ids: set[str] | None = None, user_ids: set[str] | None = None) -> list[Document]:
        """Each task's knowledge corpus, namespaced by task_id:
        - the DECISION: past developer chat(s) (conversation/amended sources) or the documented
          decision commit (history sources, from task.json's decision_subject/rationale)
        - decoy developer conversations (shared pool — retrieval noise)
        - the host repo's recent commit messages (shared — history noise)
        Generic providers ingest exactly this; the `hscoding` provider ignores the document list and
        runs the plugin's deepen engine over the BUILT repo instead (same knowledge, repo-native)."""
        decoys_path = _DATASETS / "gen" / "decoy_conversations.json"
        decoys = json.loads(decoys_path.read_text()) if decoys_path.exists() else []
        noise = _host_noise_commits()
        docs: list[Document] = []
        for tj in self._task_files():
            t = json.loads(tj.read_text())
            if category and t.get("source") != category:
                continue
            tid = t["task_id"]
            if user_ids is not None and tid not in user_ids:
                continue
            conv = t.get("conversations") or []
            chats = conv if conv and isinstance(conv[0], list) else ([conv] if conv else [])
            for ci, chat in enumerate(chats):
                docs.append(Document(
                    id=f"{tid}:chat{ci}", content=_render_chat(chat), user_id=tid,
                    messages=[{"role": m.get("role", "user"), "content": m.get("text", "")} for m in chat],
                    context="past developer conversation about this project"))
            if t.get("decision_subject"):
                docs.append(Document(
                    id=f"{tid}:decision-commit",
                    content=f"Git commit: {t['decision_subject']}\n\n{t.get('decision_rationale', '')}",
                    user_id=tid, context="a commit message from this project's git history"))
            for d in decoys:
                docs.append(Document(
                    id=f"{tid}:{d.get('id', 'decoy')}", content=_render_chat(d.get("turns", [])),
                    user_id=tid, context="past developer conversation about this project"))
            for sha, msg in noise:
                docs.append(Document(
                    id=f"{tid}:git-{sha}", content=f"Git commit: {msg}", user_id=tid,
                    context="a commit message from this project's git history"))
        if ids is not None:
            docs = [d for d in docs if d.id in ids]
        return docs[:limit] if limit else docs

    def categories(self, split: str) -> list[str] | None:
        # PRIMARY category = source (history / conversation) — the benchmark's main axis.
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

    def dataset_stats(self, console, sample_size: int = 200) -> None:
        tasks = [json.loads(tj.read_text()) for tj in self._task_files()]
        def census(key):
            out: dict[str, int] = {}
            for t in tasks:
                out[t.get(key) or "?"] = out.get(t.get(key) or "?", 0) + 1
            return ", ".join(f"{k}={v}" for k, v in sorted(out.items()))
        docs = self.load_documents("boltons")
        per_unit = len(docs) // max(len(tasks), 1)
        console.print(f"[bold]sdebench[/bold] — {len(tasks)} tasks (boltons-hosted bug fixes)")
        console.print(f"  source:   {census('source')}")
        console.print(f"  tier:     {census('tier')}")
        console.print(f"  category: {census('category')}")
        console.print(f"  corpus:   {len(docs)} documents (~{per_unit}/task: decision chat(s)/commit "
                      f"+ decoy conversations + host-history noise)")
