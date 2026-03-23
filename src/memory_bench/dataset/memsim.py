"""
MemSim / MemDaily dataset (https://github.com/nuster1128/MemSim).

The dataset is Chinese-language, multiple-choice, with six QA types:
  simple | conditional | comparative | aggregative | post_processing | noisy

Each trajectory has a message_list (the memories) and a single QA object.
gold_ids → target_step_id entries map questions back to their evidence messages.
"""
import json
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

_DATA_URL = (
    "https://raw.githubusercontent.com/nuster1128/MemSim/master/"
    "benchmark/rawdata/memdaily.json"
)
_CACHE_PATH = Path.home() / ".cache" / "memory_bench" / "memdaily.json"

SPLITS = ["simple", "conditional", "comparative", "aggregative", "post_processing", "noisy"]


class MemSimDataset(Dataset):
    """
    MemSim benchmark (MemDaily subset).

    Data is auto-downloaded from the public GitHub repo on first use and
    cached at ~/.cache/memory_bench/memdaily.json.
    You can also set MEMSIM_DATA_PATH to point at a local copy.
    """

    name = "memsim"
    description = "Chinese daily-life memory simulation with diverse QA types."
    splits = SPLITS
    task_type = "mcq"
    links = [
        {"label": "GitHub", "url": "https://github.com/nuster1128/MemSim"},
    ]

    def __init__(self) -> None:
        import os
        env = os.environ.get("MEMSIM_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data(self) -> dict:
        if self._local_path:
            path = self._local_path
        else:
            path = dataset_cache_dir("memsim") / "memdaily.json"
            if not path.exists():
                print(f"Downloading MemSim data → {path} …")
                urllib.request.urlretrieve(_DATA_URL, path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_trajectories(self, split: str) -> list[dict]:
        data = self._get_data()
        if split not in data:
            raise ValueError(f"Unknown MemSim split '{split}'. Available: {list(data)}")
        scenarios = data[split]
        trajectories: list[dict] = []
        for scenario, items in scenarios.items():
            for traj in items:
                traj = dict(traj)
                traj["_scenario"] = scenario
                trajectories.append(traj)
        return trajectories

    @staticmethod
    def _format_message(msg) -> str:
        """Format a raw message dict into a readable string."""
        if isinstance(msg, str):
            return msg
        parts = [msg.get("message", "")]
        if msg.get("time"):
            parts.append(f"(time: {msg['time']})")
        if msg.get("place"):
            parts.append(f"(place: {msg['place']})")
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        trajectories = self._load_trajectories(split)
        queries: list[Query] = []

        for traj in trajectories:
            qa = traj.get("QA")
            if not qa:
                continue

            tid = traj["tid"]
            choices: dict[str, str] = qa.get("choices") or {}
            ground_truth: str = qa.get("ground_truth", "")
            full_answer: str = choices.get(ground_truth, qa.get("answer", ""))

            choice_lines = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
            query_text = (
                f"{qa['question']}\n\n{choice_lines}" if choice_lines else qa["question"]
            )

            gold_ids = [f"{tid}_{step}" for step in qa.get("target_step_id", [])]
            gold_answers = [full_answer]
            if ground_truth and ground_truth not in gold_answers:
                gold_answers.append(ground_truth)

            queries.append(Query(
                id=f"{tid}_{qa['qid']}",
                query=query_text,
                gold_ids=gold_ids,
                gold_answers=gold_answers,
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
    ) -> list[Document]:
        trajectories = self._load_trajectories(split)
        documents: list[Document] = []

        for traj in trajectories:
            tid = traj["tid"]
            for msg in traj.get("message_list", []):
                mid = msg.get("mid", len(documents)) if isinstance(msg, dict) else len(documents)
                doc_id = f"{tid}_{mid}"
                if ids is not None and doc_id not in ids:
                    continue
                documents.append(Document(id=doc_id, content=self._format_message(msg)))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="MemSim (MemDaily) dataset stats")
        table.add_column("Domain",         style="bold")
        table.add_column("Queries",        justify="right")
        table.add_column("Trajectories",   justify="right")
        table.add_column("Docs (msgs)",    justify="right")
        table.add_column("Avg msgs/traj",  justify="right")

        total_q = total_d = 0
        for domain in SPLITS:
            trajs = self._load_trajectories(domain)
            n_queries = sum(1 for t in trajs if t.get("QA"))
            n_docs    = sum(len(t.get("message_list", [])) for t in trajs)
            avg       = int(n_docs / len(trajs)) if trajs else 0
            total_q  += n_queries
            total_d  += n_docs
            table.add_row(domain, f"{n_queries:,}", f"{len(trajs):,}", f"{n_docs:,}", f"~{avg}")

        table.add_section()
        table.add_row("TOTAL", f"{total_q:,}", "", f"{total_d:,}", "", style="bold")
        console.print(table)
