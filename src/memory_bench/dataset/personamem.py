"""
PersonaMem dataset (https://github.com/bowen-upenn/PersonaMem).
HuggingFace: bowen-upenn/PersonaMem, config "benchmark".

Domains (splits): 32k | 128k | 1M  (context-window sizes)

Structure
---------
Each shared context is a flat list of {role, content} turns.
"system" turns (a few per context) mark session boundaries — each system
turn sets the persona / time-period frame for the following dialogue.

Documents = one per session (system turn + all user/assistant turns until
            the next system turn).
            Document ID = "{shared_context_id}_{session_index}"

Queries   = one per row in the questions CSV.
            gold_ids = all session doc IDs whose start turn index falls
                       strictly before end_index_in_shared_context.

The task is response selection: given long personal conversation history,
choose the best assistant reply to a new user message from 4 options (a-d).
"""
import ast
import json
import re
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

SPLITS = ["32k", "128k", "1M"]

_CONTEXT_FILES = {
    "32k":  "shared_contexts_32k.jsonl",
    "128k": "shared_contexts_128k.jsonl",
    "1M":   "shared_contexts_1M.jsonl",
}


class PersonaMemDataset(Dataset):
    """
    PersonaMem benchmark — long-horizon personal preference tracking.

    Questions and contexts are loaded directly from HuggingFace
    (bowen-upenn/PersonaMem). HuggingFace handles local caching.
    """

    name = "personamem"
    published = True
    description = "Long-horizon personal preference tracking across conversation sessions."
    splits = SPLITS
    task_type = "mcq"
    links = [
        {"label": "GitHub", "url": "https://github.com/bowen-upenn/PersonaMem"},
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/bowen-upenn/PersonaMem"},
    ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw_contexts(self, domain: str) -> dict[str, list[dict]]:
        """Return {shared_context_id: [turn, ...]} (flat turn list)."""
        cache = dataset_cache_dir("personamem")
        path = cache / _CONTEXT_FILES[domain]
        if not path.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="bowen-upenn/PersonaMem",
                filename=_CONTEXT_FILES[domain],
                repo_type="dataset",
                local_dir=str(cache),
            )
        contexts: dict[str, list[dict]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Each line: {context_id: [turn, ...]}
                ctx_id, turns = next(iter(entry.items()))
                contexts[ctx_id] = turns
        return contexts

    def _split_into_sessions(self, turns: list[dict]) -> list[dict]:
        """
        Split a flat turn list into sessions at each system turn.
        Returns a list of {start_turn_idx, turns} dicts.
        """
        sessions: list[dict] = []
        current: list[dict] = []
        start_idx = 0

        for i, turn in enumerate(turns):
            if turn.get("role") == "system" and current:
                sessions.append({"start_turn_idx": start_idx, "turns": current})
                current = []
                start_idx = i
            current.append(turn)

        if current:
            sessions.append({"start_turn_idx": start_idx, "turns": current})

        return sessions

    @staticmethod
    def _extract_timestamp(session: dict) -> str | None:
        """Extract an ISO-8601 timestamp from date mentions in a session's text."""
        _MONTHS = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        text = " ".join(t.get("content", "") for t in session["turns"])
        # Match "Month DD, YYYY" or "Month D, YYYY"
        m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", text, re.IGNORECASE)
        if m:
            month = _MONTHS[m.group(1).lower()]
            day = int(m.group(2))
            year = int(m.group(3))
            try:
                return datetime(year, month, day, tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        # Match YYYY-MM-DD
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        return None

    @staticmethod
    def _format_session(session: dict) -> str:
        parts: list[str] = []
        for turn in session["turns"]:
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"[{role.upper()}] {content}")
        return "\n\n".join(parts)

    def _load_sessions(self, domain: str) -> dict[str, list[dict]]:
        """Return {ctx_id: [session, ...]} where each session has start_turn_idx + turns."""
        raw = self._load_raw_contexts(domain)
        return {ctx_id: self._split_into_sessions(turns) for ctx_id, turns in raw.items()}

    @staticmethod
    def _persona_name(sessions: list[dict]) -> str:
        """Extract just the persona name from the first system turn, e.g. 'Kanoa Manu'."""
        for session in sessions:
            for turn in session["turns"]:
                if (turn.get("role") or "").lower() != "system":
                    continue
                for line in (turn.get("content") or "").splitlines():
                    if "Name:" in line:
                        return line.split("Name:", 1)[-1].strip()
        return ""

    def _load_questions(self, split: str) -> list[dict]:
        from datasets import load_dataset
        cache = dataset_cache_dir("personamem")
        ds = load_dataset("bowen-upenn/PersonaMem", "benchmark", split=split,
                          cache_dir=str(cache))
        return list(ds)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        c = meta.get("question_type")
        return {"Question Type": [c]} if c else {}

    def categories(self, split: str) -> list[str] | None:
        rows = self._load_questions(split)
        return sorted({r["question_type"] for r in rows if r.get("question_type")})

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        rows = self._load_questions(split)
        if category:
            rows = [r for r in rows if r.get("question_type") == category]
        sessions_by_ctx = self._load_sessions(split)
        queries: list[Query] = []

        for row in rows:
            ctx_id  = row["shared_context_id"]
            end_idx = int(row["end_index_in_shared_context"])
            correct = (row["correct_answer"] or "").strip()   # e.g. "(c)"
            letter  = correct.strip("() ")                    # e.g. "c"

            # Parse 4 options (stored as Python-literal list string)
            raw_opts = row.get("all_options") or "[]"
            try:
                options: list[str] = ast.literal_eval(raw_opts)
            except Exception:
                try:
                    options = json.loads(raw_opts)
                except Exception:
                    options = []

            # Query: persona name (to scope retrieval to the right person) +
            # user message + options. Name only — no preference descriptions.
            user_msg   = row["user_question_or_message"] or ""
            name       = self._persona_name(sessions_by_ctx.get(ctx_id, []))
            prefix     = f"User: {name}" if name else ""
            opt_lines  = "\n".join(options)
            query_text = "\n\n".join(filter(None, [prefix, user_msg, opt_lines]))
            # Retrieval query excludes options (cleaner signal for memory lookup)
            retrieval_query = "\n\n".join(filter(None, [prefix, user_msg]))

            # gold_ids = sessions whose start turn falls before end_idx
            sessions = sessions_by_ctx.get(ctx_id, [])
            gold_ids = [
                f"{ctx_id}_{i}"
                for i, s in enumerate(sessions)
                if s["start_turn_idx"] < end_idx
            ]

            # Correct option full text
            correct_text = next(
                (o for o in options if o.lower().startswith(f"({letter})")),
                correct,
            )
            gold_answers = [correct_text]
            if letter and letter not in gold_answers:
                gold_answers.append(letter)

            # Extract query timestamp from the last gold session
            query_timestamp = None
            for i in sorted(
                [i for i, s in enumerate(sessions) if s["start_turn_idx"] < end_idx],
                reverse=True,
            ):
                query_timestamp = self._extract_timestamp(sessions[i])
                if query_timestamp:
                    break

            queries.append(Query(
                id=row["question_id"],
                query=query_text,
                gold_ids=gold_ids,
                gold_answers=gold_answers,
                user_id=ctx_id,
                meta={
                    "persona_id": str(row["persona_id"]),
                    "question_type": row["question_type"],
                    "topic": row["topic"],
                    "retrieval_query": retrieval_query,
                    **({"query_timestamp": query_timestamp} if query_timestamp else {}),
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
    ) -> list[Document]:
        sessions_by_ctx = self._load_sessions(split)
        documents: list[Document] = []

        for ctx_id, sessions in sessions_by_ctx.items():
            for i, session in enumerate(sessions):
                doc_id = f"{ctx_id}_{i}"
                if ids is not None and doc_id not in ids:
                    continue
                content = self._format_session(session)
                if content:
                    # Pass raw turns so Mem0 can ingest a proper conversation
                    # rather than a single wall-of-text user message.
                    turns = [
                        {"role": t["role"], "content": t["content"]}
                        for t in session["turns"]
                        if t.get("content", "").strip()
                    ]
                    documents.append(Document(
                        id=doc_id, content=content, user_id=ctx_id, messages=turns,
                        timestamp=self._extract_timestamp(session),
                    ))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="PersonaMem dataset stats")
        table.add_column("Domain",      style="bold")
        table.add_column("Questions",   justify="right")
        table.add_column("Personas",    justify="right")
        table.add_column("Q-types",     justify="right")
        table.add_column("Sessions",    justify="right")

        total_q = total_s = 0
        for domain in SPLITS:
            try:
                rows     = self._load_questions(domain)
                sessions = self._load_sessions(domain)
            except Exception as e:
                table.add_row(domain, f"err: {e}", "", "", "")
                continue

            n_q       = len(rows)
            n_persona = len({r["persona_id"] for r in rows})
            n_qtypes  = len({r["question_type"] for r in rows})
            n_sess    = sum(len(s) for s in sessions.values())
            total_q  += n_q
            total_s  += n_sess
            table.add_row(domain, f"{n_q:,}", str(n_persona), str(n_qtypes), f"{n_sess:,}")

        table.add_section()
        table.add_row("TOTAL", f"{total_q:,}", "", "", f"{total_s:,}", style="bold")
        console.print(table)
