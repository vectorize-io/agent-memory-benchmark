"""supportbench — a support agent that remembers each customer and reads the product's manuals.

A customer asks the support agent a question. The answer can need what the agent remembers about
that customer (past support chats), what the product's documentation says (the real Nextcloud 32
user and administration manuals, 837 PDF pages with screenshots), or both. The question never says
which source to use: deciding that is the point.

Four question groups (``category``):

- ``docs_only``   — the manuals answer it; the customer's history is irrelevant.
- ``needs_both``  — the answer combines a fact from the customer's chats with a rule from the docs.
- ``conflict``    — a past chat states something the docs contradict (sometimes a past support
                    reply); the docs are right.
- ``exception``   — the same question asked by two customers. One of them has an approved exception
                    on file that changes the answer; the other must get the documented default. An
                    exception leaking to the wrong customer fails the second one.

The documents, per kind (the memory providers decide where each goes — see memory/hindsight_support.py):

- the manuals: ``Document.path`` set, no ``user_id``, shared by every customer;
- chats: ``Document.messages`` set, ``user_id`` = the customer;
- approved exceptions: text, ``user_id`` = the customer, no messages.

Every document's ``context`` ends with its id in brackets, so the evidence a memory came from can
be traced back (memories carry their context, not their document id). An exception note's context
never names the customer: the knowledge bank stores no context, and the memory arm must not get a
hint the others lack.

Each answer is graded twice in one judge call: does it meet the gold answer, and does it assert the
question's ``must_not_claim`` (the trap — an outdated or wrong statement the corpus contains). A trap
fails the answer even if the rest is right; the judge's reason starts with ``TRAP`` so
``summary_metrics`` can count them.

The scenario (users, chats, exceptions, questions) and the two manuals live in the
``hindsight-datasets`` submodule at ``datasets/supportbench/`` (``git submodule update --init``). The manuals are © Nextcloud GmbH and
contributors, licensed CC BY 3.0 (https://github.com/nextcloud/documentation).
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

from ..models import Document, Query
from .base import Dataset

_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "supportbench"
_SCENARIO = _ROOT / "supportbench.json"
CATEGORIES = ("docs_only", "needs_both", "conflict", "exception")

_JUDGE_PROMPT = """\
You are a strict evaluator grading a support agent's answer to a customer.

Customer question:
{query}

Gold answer (the essential facts the answer must contain):
{gold}

Trap (a wrong or outdated statement present in the agent's sources):
The answer asserts, as true for this customer now, that {trap}.

Agent's answer:
{answer}

First decide whether the answer asserts the trap. Mentioning it only to correct it, or to say it is
outdated, is not asserting it.
- If it asserts the trap: correct=false, and the reason MUST start with "TRAP:".
- Otherwise: correct=true only if the answer captures the essential facts of the gold answer.
  Refusing, saying it does not know, or giving a vague answer is correct=false. Extra correct
  detail and paraphrasing are fine.\
"""


@cache
def scenario() -> dict:
    if not _SCENARIO.exists():
        raise FileNotFoundError(f"{_SCENARIO} is missing: run `git submodule update --init datasets`")
    return json.loads(_SCENARIO.read_text())


def manual_paths() -> list[Path]:
    """The manuals, from the dataset repo; each is checked against the sha256 its page references
    were written for."""
    paths = []
    for m in scenario()["manuals"]:
        path = _ROOT / m["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != m["sha256"]:
            raise ValueError(f"supportbench: {path} does not match its pinned sha256")
        paths.append(path)
    return paths


def _messages(turns: list[list[str]]) -> list[dict[str, str]]:
    """A chat as chat messages: the customer is the user, the support agent the assistant."""
    return [{"role": "user" if role == "customer" else "assistant", "content": text} for role, text in turns]


class SupportBenchDataset(Dataset):
    name = "supportbench"
    description = (
        "Does a support agent use the product docs and each customer's history correctly? Real Nextcloud 32 "
        "manuals (837 PDF pages) plus per-customer support chats and approved exceptions."
    )
    splits = ["nextcloud"]
    task_type = "open"
    # No isolation unit: the manuals are shared by every customer, so the providers keep one store
    # and scope each customer by tag (see memory/hindsight_support.py).
    isolation_unit = None
    published = False
    links = [{"label": "Nextcloud docs (CC BY 3.0)", "url": "https://docs.nextcloud.com/server/32/user_manual/en/"}]

    def categories(self, split: str) -> list[str]:
        return list(CATEGORIES)

    def category_type(self, split: str, category: str):
        return "query"

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        s = scenario()
        queries = [
            Query(
                id=q["id"],
                query=q["question"],
                gold_ids=[],
                gold_answers=[q["gold"]],
                user_id=q["user"],
                meta={
                    "category": q["category"],
                    "pair": q.get("pair"),
                    "must_not_claim": q["must_not_claim"],
                    "sources": q["sources"],
                    "query_timestamp": s["asked_at"],
                    "customer": s["users"][q["user"]]["name"],
                },
            )
            for q in s["questions"]
            if category is None or q["category"] == category
        ]
        return queries[:limit] if limit else queries

    def load_documents(self, split: str, category: str | None = None, limit: int | None = None,
                       ids: set[str] | None = None, user_ids: set[str] | None = None) -> list[Document]:
        s = scenario()
        docs = [
            Document(id=m["id"], content="", path=str(path), filename=m["filename"],
                     context=f"{s['product']} documentation [{m['id']}]")
            for m, path in zip(s["manuals"], manual_paths())
        ]
        for uid, user in s["users"].items():
            for chat in user["chats"]:
                docs.append(Document(
                    # Stored as the JSON list of messages, in both kinds of bank.
                    id=chat["id"], content=json.dumps(_messages(chat["turns"]), ensure_ascii=False), user_id=uid,
                    timestamp=chat["at"], messages=_messages(chat["turns"]),
                    context=f"support chat with the customer {user['name']} [{chat['id']}]",
                ))
        for exc in s["exceptions"]:
            docs.append(Document(id=exc["id"], content=exc["text"], user_id=exc["user"], filename=f"{exc['id']}.md",
                                 context=f"approved exception note [{exc['id']}]"))
        # The manuals belong to everyone: a user filter keeps them.
        if user_ids is not None:
            docs = [d for d in docs if d.user_id is None or d.user_id in user_ids]
        if ids is not None:
            docs = [d for d in docs if d.id in ids]
        return docs[:limit] if limit else docs

    def get_judge_prompt_fn(self, category: str | None, meta: dict | None = None):
        trap = (meta or {})["must_not_claim"]
        return lambda query, gold, answer: _JUDGE_PROMPT.format(query=query, gold="\n".join(gold), trap=trap, answer=answer)

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        return {"Category": [meta["category"]], "Customer": [meta["customer"]]}

    def supports_oracle(self) -> bool:
        return False

    def summary_metrics(self, results: list) -> dict:
        """The trap rate next to accuracy (a wrong answer that repeats the outdated claim is the
        failure this dataset exists for), and what the answers cost and took."""
        if not results:
            return {}
        metas = [r.meta for r in results]
        usd = [m.get("query_usd") for m in metas]
        out = {
            "trap_rate": round(sum(r.judge_reason.startswith("TRAP") for r in results) / len(results), 3),
            "avg_answer_s": round(sum(r.retrieve_time_ms for r in results) / len(results) / 1000, 2),
            "avg_query_usd": None if None in usd else round(sum(usd) / len(usd), 5),
            "ingest_usd": metas[0].get("ingest_usd"),
            "ingest_s": metas[0].get("ingest_s"),
        }
        for cat in CATEGORIES:
            rows = [r for r in results if r.meta.get("category") == cat]
            if rows:
                out[f"accuracy_{cat}"] = round(sum(r.correct for r in rows) / len(rows), 3)
        return out

    def dataset_stats(self, console) -> None:
        s = scenario()
        qs = s["questions"]
        console.print(f"[bold]supportbench[/bold] — {s['product']}: {len(s['manuals'])} manuals, {len(s['users'])} customers, "
                      f"{sum(len(u['chats']) for u in s['users'].values())} chats, {len(s['exceptions'])} exceptions")
        for cat in CATEGORIES:
            console.print(f"  {cat}: {sum(q['category'] == cat for q in qs)} questions")
