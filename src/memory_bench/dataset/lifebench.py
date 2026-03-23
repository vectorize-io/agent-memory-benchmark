"""
LifeBench dataset (https://github.com/1754955896/LifeBench).
Paper: https://arxiv.org/abs/2603.03781 (ICLR 2026 Memory Agent Workshop)

A benchmark for long-horizon multi-source personalized memory evaluation.
Simulates a full year of daily life for 10 fictional users with rich
multi-modal digital traces (SMS, calls, photos, calendar, fitness, etc.)
and 2,003 English QA pairs derived from them.

Data is auto-downloaded from GitHub on first use. You can also set
LIFEBENCH_DATA_PATH to point at a local copy of our_en.json.

Structure
---------
10 users × ~200 QA pairs = 2,003 queries. Single "en" split.

The data uses a LoCoMo-compatible format:
  - sample_id: user name
  - conversation: dict with session_1..session_364 (one per calendar day),
                  each with session_N_date_time timestamp
  - qa: list of {question, answer, evidence, category}

Documents  = one per session per user
             ID = "{sample_id}_{session_key}" (e.g. "alice_session_1")

Queries    = one per QA pair per user
             gold_ids = sessions referenced in evidence

Categories (query-level, by category string):
  "0" → information-extraction   (direct fact retrieval)
  "1" → multi-hop                (reasoning over multiple sources)
  "2" → temporal-updating        (temporal reasoning, evolving info)
  "3" → nondeclarative           (habits, preferences, emotions)
  "4" → unanswerable             (questions with no answer in data)
"""
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

_DATA_URL = (
    "https://raw.githubusercontent.com/1754955896/LifeBench/main"
    "/life_bench_data/locomo_format/our_en.json"
)

SPLITS = ["en"]

_CATEGORY_NAMES: dict[str, str] = {
    "0": "information-extraction",
    "1": "multi-hop",
    "2": "temporal-updating",
    "3": "nondeclarative",
    "4": "unanswerable",
}

_CATEGORY_LABELS: dict[str, str] = {
    "information-extraction": "Information Extraction",
    "multi-hop": "Multi-hop Reasoning",
    "temporal-updating": "Temporal & Knowledge Updating",
    "nondeclarative": "Nondeclarative Memory",
    "unanswerable": "Unanswerable",
}


class LifeBenchDataset(Dataset):
    """
    LifeBench — long-horizon multi-source personalized memory benchmark.

    Data is auto-downloaded from GitHub on first use.
    Set LIFEBENCH_DATA_PATH to point at a local our_en.json to skip download.
    """

    name = "lifebench"
    published = True
    description = "Long-horizon multi-source personalized memory benchmark across 10 users."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "user"
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2603.03781"},
        {"label": "GitHub", "url": "https://github.com/1754955896/LifeBench"},
    ]

    def __init__(self) -> None:
        import os
        env = os.environ.get("LIFEBENCH_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_path(self) -> Path:
        if self._local_path:
            return self._local_path
        cache = dataset_cache_dir("lifebench")
        path = cache / "our_en.json"
        if not path.exists():
            print("Downloading LifeBench dataset…")
            urllib.request.urlretrieve(_DATA_URL, path)
        return path

    def _load_raw(self) -> list[dict]:
        with open(self._data_path(), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _session_keys(conv: dict) -> list[str]:
        """Return sorted session keys (e.g. ['session_1', 'session_2', ...])."""
        return sorted(
            (k for k in conv
             if k.startswith("session_") and not k.endswith("_date_time")
             and isinstance(conv[k], list)),
            key=lambda k: int(k.split("_", 1)[1]),
        )

    @staticmethod
    def _parse_date(date_string: str | None) -> str | None:
        """Parse various date formats → ISO-8601 UTC string."""
        if not date_string:
            return None
        for fmt in [
            "%I:%M %p on %d %B, %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ]:
            try:
                dt = datetime.strptime(date_string, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _build_dia_to_session(conv: dict, session_keys: list[str]) -> dict[str, str]:
        """Return {dia_id: session_key} for all turns in a conversation."""
        mapping: dict[str, str] = {}
        for key in session_keys:
            for turn in conv.get(key, []):
                if isinstance(turn, dict) and "dia_id" in turn:
                    mapping[str(turn["dia_id"])] = key
        return mapping

    @staticmethod
    def _session_content(turns: list) -> str:
        return json.dumps(turns)

    def _user_ids(self) -> list[str]:
        return [item["sample_id"] for item in self._load_raw()]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def categories(self, split: str) -> list[str] | None:
        # Doc categories: one per user (isolates document space)
        # Query categories: question types (same docs, different query filter)
        return self._user_ids() + list(_CATEGORY_NAMES.values())

    def category_type(self, split: str, category: str) -> str:
        return "query" if category in _CATEGORY_NAMES.values() else "doc"

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        if meta.get("sample_id"):
            axes["User"] = [meta["sample_id"]]
        if meta.get("category"):
            axes["Question Type"] = [meta["category"]]
        return axes

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        task_type: str,
        split: str,
        category: str | None = None,
        meta: dict | None = None,
    ) -> str:
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        date_str = ""
        if query_timestamp:
            date_str = f"\n# CURRENT DATE:\nThe question is being asked on: {query_timestamp} UTC\n"
        raw = meta.get("_raw_response")
        ctx = json.dumps(raw) if raw else context
        return f"""\
# CONTEXT:
You have access to facts and entities from a user's personal life data.
{date_str}
# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information or multiple instances of an event, say them all
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

Context:

{ctx}

Question: {query}
"""

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        gold_str = gold_answers[0] if gold_answers else ""
        return f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question about a user's personal information and life events,
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The gold answer will usually be a concise answer that includes the key fact.
The generated answer might be much longer, but you should be generous with your grading — as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For time-related questions, the gold answer will be a specific date or period. The generated answer is CORRECT if it refers to the same date or period, even if formatted differently.
If the gold answer indicates the question is unanswerable from the data, and the generated answer also says it cannot be answered, count it as CORRECT.

Question: {query}
Gold answer: {gold_str}
Generated answer: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.
"""

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        data = self._load_raw()

        # Determine filter type
        cat_filter: str | None = None    # question-type filter (category name)
        user_filter: str | None = None   # user filter (sample_id)
        if category:
            if category in _CATEGORY_NAMES.values():
                cat_filter = category
            else:
                user_filter = category

        queries: list[Query] = []
        for item in data:
            sample_id = item["sample_id"]
            if user_filter is not None and sample_id != user_filter:
                continue

            conv = item["conversation"]
            session_keys = self._session_keys(conv)
            dia_to_session = self._build_dia_to_session(conv, session_keys)

            # Query timestamp = date of the last session
            last_session_ts: str | None = None
            for sk in reversed(session_keys):
                last_session_ts = self._parse_date(conv.get(f"{sk}_date_time"))
                if last_session_ts:
                    break

            for qi, qa in enumerate(item.get("qa", [])):
                cat_str = str(qa.get("category", ""))
                cat_name = _CATEGORY_NAMES.get(cat_str, cat_str)
                if cat_filter is not None and cat_name != cat_filter:
                    continue

                question = qa.get("question", "")
                answer = qa.get("answer", "")
                evidence = qa.get("evidence") or []

                # Map evidence dia_ids → unique session doc IDs
                gold_session_keys = {
                    dia_to_session[str(eid)]
                    for eid in evidence
                    if str(eid) in dia_to_session
                }
                gold_ids = [f"{sample_id}_{sk}" for sk in sorted(gold_session_keys)]

                queries.append(Query(
                    id=f"{sample_id}_q{qi}",
                    query=question,
                    gold_ids=gold_ids,
                    gold_answers=[answer],
                    user_id=sample_id,
                    meta={
                        "sample_id": sample_id,
                        "category": cat_name,
                        **({"query_timestamp": last_session_ts} if last_session_ts else {}),
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
        data = self._load_raw()
        documents: list[Document] = []

        # Doc-category filter: restrict to a single user
        user_filter = category if category and category not in _CATEGORY_NAMES.values() else None

        for item in data:
            sample_id = item["sample_id"]
            if user_filter is not None and sample_id != user_filter:
                continue
            if user_ids is not None and sample_id not in user_ids:
                continue

            conv = item["conversation"]
            session_keys = self._session_keys(conv)

            for sk in session_keys:
                doc_id = f"{sample_id}_{sk}"
                if ids is not None and doc_id not in ids:
                    continue
                turns = conv.get(sk, [])
                if not turns:
                    continue
                timestamp = self._parse_date(conv.get(f"{sk}_date_time"))
                content = self._session_content(turns)
                date_display = timestamp or "unknown date"
                ctx = f"User '{sample_id}' — {sk} ({date_display})"
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    user_id=sample_id,
                    timestamp=timestamp,
                    context=ctx,
                ))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        data = self._load_raw()
        table = Table(title="LifeBench dataset stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        from collections import Counter
        cat_counts: Counter = Counter()
        n_sessions = 0
        for item in data:
            n_sessions += len(self._session_keys(item["conversation"]))
            for qa in item.get("qa", []):
                cat_str = str(qa.get("category", ""))
                cat_name = _CATEGORY_NAMES.get(cat_str, cat_str)
                cat_counts[cat_name] += 1

        n_qa = sum(cat_counts.values())
        table.add_row("Users", str(len(data)))
        table.add_row("Sessions (docs)", str(n_sessions))
        table.add_row("QA pairs", str(n_qa))
        for cat_name in _CATEGORY_NAMES.values():
            label = _CATEGORY_LABELS.get(cat_name, cat_name)
            table.add_row(f"  {label}", str(cat_counts.get(cat_name, 0)))
        console.print(table)
