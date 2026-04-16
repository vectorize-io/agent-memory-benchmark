"""
LoComo dataset (https://github.com/snap-research/locomo).

A long-term conversation memory benchmark with 10 multi-session conversations
and open-ended QA pairs over personal conversation history.

Data is auto-downloaded from the public GitHub repo on first use and cached
locally. You can also set LOCOMO_DATA_PATH to point at a local copy.

Structure
---------
Each item has a sample_id, conversation (multi-session), and qa list.
Conversation sessions are keyed as session_1, session_2, ... with associated
session_<n>_date_time timestamps.

Documents  = one per session per conversation
             ID = "{sample_id}_{session_key}" (e.g. "abc_session_1")

Queries    = one per QA pair per conversation
             gold_ids = sessions containing evidence turns

Categories (query-level, by QA category int):
  1 → single-hop   3 → multi-hop
  2 → temporal     4 → open-domain    5 → adversarial
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
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

SPLITS = ["locomo10"]

_CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
}

_SKIP_CATEGORIES = {5}  # adversarial — excluded from benchmark


class LoComoDataset(Dataset):
    """
    LoComo benchmark — long-term multi-session conversation memory.

    Data is auto-downloaded from the snap-research/locomo GitHub repo on first
    use. Set LOCOMO_DATA_PATH to point at a local JSON file to skip download.
    """

    name = "locomo"
    published = True
    description = "Multi-session long-term conversations with 1,986 QA pairs."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "conversation"
    links = [
        {"label": "GitHub", "url": "https://github.com/snap-research/locomo"},
    ]

    def __init__(self) -> None:
        import os
        env = os.environ.get("LOCOMO_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_path(self) -> Path:
        if self._local_path:
            return self._local_path
        cache = dataset_cache_dir("locomo")
        path = cache / "locomo10.json"
        if not path.exists():
            urllib.request.urlretrieve(_DATA_URL, path)
        return path

    def _load_raw(self) -> list[dict]:
        with open(self._data_path(), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _session_keys(conv: dict) -> list[str]:
        """Return sorted session keys (e.g. ['session_1', 'session_2', ...])."""
        return sorted(
            k for k in conv
            if k.startswith("session_") and not k.endswith("_date_time")
            and isinstance(conv[k], list)
        )

    @staticmethod
    def _parse_date(date_string: str | None) -> str | None:
        """Parse '1:56 pm on 8 May, 2023' → ISO-8601 UTC string."""
        if not date_string:
            return None
        try:
            dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _session_content(turns: list[dict]) -> str:
        """Serialize session turns as JSON (matches Hindsight reference implementation)."""
        return json.dumps(turns)

    @staticmethod
    def _build_dia_to_session(conv: dict, session_keys: list[str]) -> dict[str, str]:
        """Return {dia_id: session_key} for all turns in a conversation."""
        mapping: dict[str, str] = {}
        for key in session_keys:
            for turn in conv.get(key, []):
                if isinstance(turn, dict) and "dia_id" in turn:
                    mapping[turn["dia_id"]] = key
        return mapping

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def _conv_ids(self) -> list[str]:
        return [item["sample_id"] for item in self._load_raw()]

    def build_rag_prompt(self, query: str, context: str, task_type: str, split: str, category: str | None = None, meta: dict | None = None) -> str:
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        date_str = ""
        if query_timestamp:
            date_str = f"\n# CURRENT DATE:\nThe question is being asked on: {query_timestamp} UTC\n"
        # Use full recall result JSON (includes entities) when available, matching the reference impl
        raw = meta.get("_raw_response")
        ctx = json.dumps(raw) if raw else context
        return f"""\
# CONTEXT:
You have access to facts and entities from a conversation.
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
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.

Question: {query}
Gold answer: {gold_str}
Generated answer: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.
"""

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes = {}
        if meta.get("sample_id"):
            axes["Conversation"] = [meta["sample_id"]]
        if meta.get("category"):
            axes["Question Type"] = [meta["category"]]
        return axes

    def categories(self, split: str) -> list[str] | None:
        # Doc categories: one per conversation (isolates document space)
        # Query categories: question types (same docs, different query filter)
        return self._conv_ids() + list(_CATEGORY_NAMES.values())

    def category_type(self, split: str, category: str) -> str:
        return "query" if category in _CATEGORY_NAMES.values() else "doc"

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        data = self._load_raw()

        # Determine filter type
        category_int: int | None = None      # question-type filter
        conv_filter: str | None = None       # conversation filter
        if category:
            if category in _CATEGORY_NAMES.values():
                for k, v in _CATEGORY_NAMES.items():
                    if v == category:
                        category_int = k
                        break
            else:
                conv_filter = category

        queries: list[Query] = []
        for item in data:
            sample_id = item["sample_id"]
            if conv_filter is not None and sample_id != conv_filter:
                continue
            conv = item["conversation"]
            speaker_a = conv.get("speaker_a", "A")
            speaker_b = conv.get("speaker_b", "B")
            session_keys = self._session_keys(conv)
            dia_to_session = self._build_dia_to_session(conv, session_keys)

            # Query timestamp = date of the last session (questions asked after all sessions)
            last_session_ts = None
            for sk in reversed(session_keys):
                last_session_ts = self._parse_date(conv.get(f"{sk}_date_time"))
                if last_session_ts:
                    break

            for qi, qa in enumerate(item.get("qa", [])):
                cat_int = qa.get("category")
                if cat_int in _SKIP_CATEGORIES:
                    continue
                if category_int is not None and cat_int != category_int:
                    continue

                question = qa.get("question", "")
                answer = qa.get("answer", "")
                evidence = qa.get("evidence") or []

                # Map evidence dia_ids → unique session doc IDs
                gold_session_keys = {
                    dia_to_session[eid]
                    for eid in evidence
                    if eid in dia_to_session
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
                        "category": _CATEGORY_NAMES.get(cat_int, str(cat_int)),
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
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

        # Doc-category filter: restrict to a single conversation
        conv_filter = category if category and category not in _CATEGORY_NAMES.values() else None

        for item in data:
            sample_id = item["sample_id"]
            if user_ids is not None and sample_id not in user_ids:
                continue
            if conv_filter is not None and sample_id != conv_filter:
                continue
            conv = item["conversation"]
            speaker_a = conv.get("speaker_a", "A")
            speaker_b = conv.get("speaker_b", "B")
            session_keys = self._session_keys(conv)

            for sk in session_keys:
                doc_id = f"{sample_id}_{sk}"
                if ids is not None and doc_id not in ids:
                    continue
                turns = conv[sk]
                if not turns:
                    continue
                date_key = f"{sk}_date_time"
                timestamp = self._parse_date(conv.get(date_key))
                content = self._session_content(turns)
                ctx = f"Conversation between {speaker_a} and {speaker_b} ({sk} of {sample_id})"
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
        table = Table(title="LoComo dataset stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        n_convs = len(data)
        n_sessions = sum(
            len(self._session_keys(item["conversation"])) for item in data
        )
        n_qa = sum(len(item.get("qa", [])) for item in data)

        cat_counts: dict[str, int] = {v: 0 for v in _CATEGORY_NAMES.values()}
        for item in data:
            for qa in item.get("qa", []):
                cat_int = qa.get("category")
                if cat_int in _SKIP_CATEGORIES:
                    continue
                cat = _CATEGORY_NAMES.get(cat_int, "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        table.add_row("Conversations", str(n_convs))
        table.add_row("Sessions (docs)", str(n_sessions))
        table.add_row("QA pairs", str(n_qa))
        for cat, count in cat_counts.items():
            table.add_row(f"  {cat}", str(count))
        console.print(table)
