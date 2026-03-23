"""
LongMemEval dataset (https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

A benchmark for long-term memory in LLM-based chat assistants with ~500 questions
across 6 question types, each with its own haystack of conversation sessions.

Data is auto-downloaded from HuggingFace on first use. You can also set
LONGMEMEVAL_DATA_PATH to point at a local copy.

Structure
---------
Each item has a question_id, question, answer, question_type, question_date,
haystack_sessions (list of sessions), haystack_dates, haystack_session_ids.

Documents  = one per session per question
             ID = "{question_id}_{session_id}"
             Isolation unit = question_id (each question gets its own bank)

Queries    = one per item (each item has exactly one QA pair)

Categories (query-level, by question_type):
  single-session-user        single-session-assistant
  multi-session              temporal-reasoning
  knowledge-update           single-session-preference
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
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_s_cleaned.json"
)

SPLITS = ["s"]

_QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
]


class LongMemEvalDataset(Dataset):
    """
    LongMemEval benchmark — long-term memory in LLM-based chat assistants.

    Data is auto-downloaded from HuggingFace on first use.
    Set LONGMEMEVAL_DATA_PATH to point at a local JSON file to skip download.
    """

    name = "longmemeval"
    published = True
    description = "Long-term memory evaluation in LLM-based chat assistants."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "question"
    links = [
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"},
    ]

    def __init__(self) -> None:
        import os
        env = os.environ.get("LONGMEMEVAL_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_path(self) -> Path:
        if self._local_path:
            return self._local_path
        cache = dataset_cache_dir("longmemeval")
        path = cache / "longmemeval_s_cleaned.json"
        if not path.exists():
            print("Downloading LongMemEval dataset (~200MB)…")
            urllib.request.urlretrieve(_DATA_URL, path)
        return path

    def _load_raw(self) -> list[dict]:
        with open(self._data_path(), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Parse LongMemEval date format to UTC datetime."""
        if not date_str:
            return None
        try:
            # Strip day-of-week: "2023/05/20 (Sat) 02:21" → "2023/05/20 02:21"
            cleaned = date_str.split("(")[0].strip() if "(" in date_str else date_str
            for fmt in ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _parse_date_iso(date_str: str) -> str | None:
        """Return ISO-8601 string or None."""
        dt = LongMemEvalDataset._parse_date(date_str)
        return dt.isoformat() if dt else None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def categories(self, split: str) -> list[str] | None:
        return _QUESTION_TYPES

    def category_type(self, split: str, category: str) -> str:
        return "query"  # all categories are query-level filters

    def build_rag_prompt(self, query: str, context: str, task_type: str, split: str, category: str | None = None, meta: dict | None = None) -> str:
        meta = meta or {}
        raw = meta.get("_raw_response")
        ctx = json.dumps(raw) if raw else context

        question_date = meta.get("query_timestamp")
        formatted_date = question_date if question_date else "Not specified"

        return f"""You are a helpful assistant that must answer user questions based on the previous conversations.

**Answer Guidelines:**
1. Start by scanning retrieved context to understand the facts and events that happened and the timeline.
2. Reason about all the memories and find the right answer, considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

In general the answer must be comprehensive and plenty of details from the retrieved context.

For quantitative/counting questions ("how many..."): First list each unique item in your reasoning (1. X, 2. Y, 3. Z...), scanning ALL facts, then count them for your answer.
If questions asks a location (where...?) make sure to include the location name.
For recommendation questions ("can you recommend...", "suggest...", "any tips..."): DO NOT give actual recommendations. Instead, describe what KIND the user would prefer based on their context. Example answer format: "The user would prefer recommendations for [category] that focus on [their interest]. They would not prefer [what to avoid based on context]."
For questions asking for help or instructions, consider the users' recent memories and previous interactions with the assistant to understand their current situation better (recent purchases, specific product models used..)
For specific number/value questions, use the context to understand what is the most up-to-date number based on recency, but also include the reasoning (in the answer) on previous possible values and why you think are less relevant.
For open questions, include as much details as possible from different sources that are relevant.
For questions where a specific entity/role is mentioned and it's different from your memory, just say the truth, don't make up anything just to fulfill the question. For example, if the question is about a specific sport, you should consider if the memories and the question are about the same sport. (e.g. american football vs soccer, shows vs podcasts)
For comparative questions, say you don't know the answer if you don't have information about both sides. (or more sides)
For questions related to time/date, carefully review the question date and the memories date to correctly answer the question.
For questions related to time/date calculation (e.g. How many days passed between X and Y?), carefully review the memories date to correctly answer the question and only provide an answer if you have information about both X and Y, otherwise say it's not possible to calculate and why.

Consider assistant's previous actions (e.g., bookings, reminders) as impactful to the user experiences.


Question: {query}
Question Date: {formatted_date}

Retrieved Context:
{ctx}


Answer:
"""

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        # We don't know the category here directly, so we use meta indirection.
        # This gets called with the raw args; category is embedded in gold_answers context.
        # The runner passes category via meta — but build_judge_prompt doesn't receive meta.
        # We store category in the closure via get_judge_prompt_fn instead.
        return None  # overridden per-query via get_judge_prompt_fn

    def get_judge_prompt_fn(self, category: str | None, meta: dict | None = None):
        """Return a judge prompt function bound to the given question_type category."""
        def _judge(query: str, gold_answers: list[str], answer: str) -> str:
            gold_str = gold_answers[0] if gold_answers else ""
            if category in ("single-session-user", "single-session-assistant", "multi-session"):
                prompt_content = f"""Evaluate if the model response contains the correct answer to the question.

I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=no.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, set correct=false

Question: {query}

Correct Answer: {gold_str}

Model Response: {answer}

Evaluation criteria:
- Set correct=true if the response contains the correct answer
- Set correct=true if the response is equivalent to the correct answer or contains intermediate steps
- Set correct=false if the response is incorrect or missing key information

Provide your evaluation as JSON with:
- reasoning: One sentence explanation
- correct: true or false"""

            elif category == "temporal-reasoning":
                prompt_content = f"""I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, answer correct=false.
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {query}
Correct Answer: {gold_str}
Model Response: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true."""

            elif category == "knowledge-update":
                prompt_content = f"""I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {query}
Correct Answer: {gold_str}
Model Response: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true."""

            elif category == "single-session-preference":
                prompt_content = f"""I will give you a question, a answer for desired personalized response, and a response from a model.
Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false.
The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {query}
Gold answer: {gold_str}
Generated answer: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true."""

            else:
                # Default: locomo-style generous judge
                prompt_content = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic.
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

Question: {query}
Gold answer: {gold_str}
Generated answer: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true."""

            return prompt_content
        return _judge

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes = {}
        if meta.get("question_type"):
            axes["Question Type"] = [meta["question_type"]]
        return axes

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        data = self._load_raw()
        queries: list[Query] = []

        for item in data:
            qtype = item.get("question_type", "unknown")
            if category is not None and qtype != category:
                continue

            question_id = item.get("question_id", "unknown")
            question = item.get("question", "")
            answer = item.get("answer", "")

            qdate_dt = self._parse_date(item.get("question_date", ""))
            query_timestamp = qdate_dt.isoformat() if qdate_dt else None

            # Gold sessions = sessions containing at least one turn with has_answer=True
            sessions    = item.get("haystack_sessions", [])
            session_ids = item.get("haystack_session_ids", [])
            gold_ids = [
                f"{question_id}_{sid}"
                for sid, turns in zip(session_ids, sessions)
                if any(isinstance(t, dict) and t.get("has_answer") for t in turns)
            ]

            queries.append(Query(
                id=question_id,
                query=question,
                gold_ids=gold_ids,
                gold_answers=[answer],
                user_id=question_id,
                meta={
                    "question_type": qtype,
                    **({"query_timestamp": query_timestamp} if query_timestamp else {}),
                },
            ))

        if limit:
            queries = queries[:limit]
        return queries

    def load_documents(self, split: str, category: str | None = None, limit: int | None = None, ids: set[str] | None = None, user_ids: set[str] | None = None) -> list[Document]:
        data = self._load_raw()
        documents: list[Document] = []

        for item in data:
            qtype = item.get("question_type", "unknown")
            if category is not None and qtype != category:
                continue

            question_id = item.get("question_id", "unknown")
            if user_ids is not None and question_id not in user_ids:
                continue

            sessions = item.get("haystack_sessions", [])
            dates = item.get("haystack_dates", [])
            session_ids = item.get("haystack_session_ids", [])

            # Align lengths
            min_len = min(len(sessions), len(dates), len(session_ids))
            sessions, dates, session_ids = sessions[:min_len], dates[:min_len], session_ids[:min_len]

            for session_turns, date_str, session_id in zip(sessions, dates, session_ids):
                doc_id = f"{question_id}_{session_id}"
                if ids is not None and doc_id not in ids:
                    continue

                # Clean turns (remove has_answer key)
                cleaned = [
                    {k: v for k, v in t.items() if k != "has_answer"} if isinstance(t, dict) else t
                    for t in session_turns
                ]

                dt = self._parse_date(date_str)
                timestamp = dt.isoformat() if dt else None
                date_display = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "unknown"
                ctx = f"Session {doc_id} - you are the assistant in this conversation - happened on {date_display} UTC."

                documents.append(Document(
                    id=doc_id,
                    content=json.dumps(cleaned),
                    user_id=question_id,
                    timestamp=timestamp,
                    context=ctx,
                ))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        data = self._load_raw()
        table = Table(title="LongMemEval dataset stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        from collections import Counter
        cat_counts: Counter = Counter()
        total_sessions = 0
        for item in data:
            cat_counts[item.get("question_type", "unknown")] += 1
            total_sessions += len(item.get("haystack_sessions", []))

        table.add_row("Questions", str(len(data)))
        table.add_row("Total sessions (docs)", str(total_sessions))
        for cat in _QUESTION_TYPES:
            table.add_row(f"  {cat}", str(cat_counts.get(cat, 0)))
        console.print(table)
