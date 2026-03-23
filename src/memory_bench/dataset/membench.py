import json
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .base import Dataset
from ..models import Document, Query

SPLITS = [
    "FirstAgentLowLevel",
    "FirstAgentHighLevel",
    "ThirdAgentLowLevel",
    "ThirdAgentHighLevel",
]

_SPLIT_FILES = {
    "FirstAgentLowLevel":  "FirstAgentDataLowLevel.json",
    "FirstAgentHighLevel": "FirstAgentDataHighLevel.json",
    "ThirdAgentLowLevel":  "ThirdAgentDataLowLevel.json",
    "ThirdAgentHighLevel": "ThirdAgentDataHighLevel.json",
}


class MemBenchDataset(Dataset):
    """
    MemBench dataset (https://github.com/import-myself/Membench).

    Data must be downloaded from Google Drive and placed locally.
    Set MEMBENCH_DATA_PATH to point at the directory containing
    FirstAgentDataLowLevel.json, ThirdAgentDataHighLevel.json, etc.
    Defaults to ./MemData.
    """

    name = "membench"
    description = "Agent memory at different abstraction levels and perspectives."
    splits = SPLITS
    task_type = "mcq"
    links = [
        {"label": "GitHub", "url": "https://github.com/import-myself/Membench"},
    ]

    def __init__(self) -> None:
        self.data_path = Path(os.environ.get("MEMBENCH_DATA_PATH", "./MemData"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trajectories(self, split: str) -> list[dict]:
        path = self.data_path / _SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"MemBench data not found at {path}. "
                f"Download the dataset and set MEMBENCH_DATA_PATH."
            )
        with open(path) as f:
            data = json.load(f)

        # File layout: {question_type: {scenario: [traj, ...]}}
        trajectories: list[dict] = []
        for question_type, scenarios in data.items():
            for traj in scenarios if isinstance(scenarios, list) else (
                item for sublist in scenarios.values() for item in sublist
            ):
                traj = dict(traj)
                traj.setdefault("_question_type", question_type)
                trajectories.append(traj)
        return trajectories

    @staticmethod
    def _format_message(msg) -> str:
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            if "user" in msg:
                # FirstAgent post-noise dialogue turn
                return f"User: {msg['user']}\nAgent: {msg['agent']}"
            if "message" in msg:
                # ThirdAgent observation message
                parts = [msg["message"]]
                if msg.get("time"):
                    parts.append(f"(time: {msg['time']})")
                if msg.get("place"):
                    parts.append(f"(place: {msg['place']})")
                return " ".join(parts)
        return str(msg)

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
            # QA may be a single dict or the first entry of question_list
            qa = traj.get("QA")
            if qa is None:
                ql = traj.get("question_list")
                qa = ql[0] if ql else None
            if not qa:
                continue

            tid = traj["tid"]
            choices: dict[str, str] = qa.get("choices") or {}
            ground_truth: str = qa.get("ground_truth", "")
            full_answer: str = choices.get(ground_truth, qa.get("answer", ""))

            # Include the multiple-choice options in the query text so the
            # memory provider and LLM see the full question as presented to
            # participants in the original benchmark.
            choice_lines = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
            query_text = f"{qa['question']}\n\n{choice_lines}" if choice_lines else qa["question"]

            # target_step_id references 0-based message indices within this trajectory
            gold_ids = [f"{tid}_{step}" for step in qa.get("target_step_id", [])]

            # Accept either the full answer text or just the letter
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
            for i, msg in enumerate(traj.get("message_list", [])):
                doc_id = f"{tid}_{i}"
                if ids is not None and doc_id not in ids:
                    continue
                documents.append(Document(id=doc_id, content=self._format_message(msg)))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="MemBench dataset stats")
        table.add_column("Domain",     style="bold")
        table.add_column("Queries",    justify="right")
        table.add_column("Docs",       justify="right")
        table.add_column("Avg msgs/traj", justify="right")

        total_q = total_d = 0
        for domain in SPLITS:
            try:
                trajs = self._load_trajectories(domain)
            except FileNotFoundError:
                table.add_row(domain, "n/a", "n/a", "n/a")
                continue

            n_queries = sum(1 for t in trajs if t.get("QA") or t.get("question_list"))
            n_docs    = sum(len(t.get("message_list", [])) for t in trajs)
            avg_msgs  = int(n_docs / len(trajs)) if trajs else 0

            total_q += n_queries
            total_d += n_docs
            table.add_row(domain, f"{n_queries:,}", f"{n_docs:,}", f"~{avg_msgs}")

        table.add_section()
        table.add_row("TOTAL", f"{total_q:,}", f"{total_d:,}", "", style="bold")
        console.print(table)
