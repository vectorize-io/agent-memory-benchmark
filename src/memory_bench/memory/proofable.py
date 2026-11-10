"""Proofable memory provider.

Proofable keeps a person's memory as private, typed records: one dated statement per
subject, where a newer record on the same subject supersedes the older one and only the
current records are read back. This provider puts the hosted service
(https://api.proofable.me) behind the benchmark's ingest/retrieve interface.

Ingest
    Conversation documents (LongMemEval, LoCoMo): a model distils each session into the
    durable facts the participants state, each under a stable dotted subject key such as
    ``user.car``. Every fact becomes one record tagged with the benchmark user, the source
    document, the subject and the session date. A fact on a subject that already has a
    record supersedes it (the product's ``supersedes:`` and ``superseded-by:`` tags), so
    the vault holds one current record per subject.

    Belief documents (PrecisionMemBench, or any document that arrives with dataset tags):
    stored verbatim as one record each, carrying the dataset's tags for exact-tag filtering.

Retrieve
    The user's current records are read with the product's by-wallet read, which never
    returns a superseded record. When the dataset supplies a tag filter it is applied first
    as exact tag matching. The same model then picks the subjects (or beliefs) that answer
    the question, and may pick none; their current records come back newest first.

Environment
    PROOFABLE_ACCESS_KEY         access key of the profile whose vault holds the memory
    PROOFABLE_API_URL            defaults to https://api.proofable.me
    PROOFABLE_MEMORY_MODEL       Gemini model for distillation and selection
                                 (default gemini-3.1-flash-lite, through the same
                                 google-genai client configuration the benchmark uses)
    PROOFABLE_TOP_K              records returned when the mode passes no limit (20)
    PROOFABLE_UNIT_CONCURRENCY   isolation units ingested in parallel (4)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)

BENCH_TAG = "bench:amb"
USER_TAG = "amb:user:"
DOC_TAG = "amb:doc:"
SUBJECT_TAG = "amb:subject:"
DATE_TAG = "amb:date:"
BELIEF_TAG = "amb:belief:"
SUPERSEDES_TAG = "supersedes:"
SUPERSEDED_BY_TAG = "superseded-by:"
CHAIN_ID = 8453
ASSISTANT_TURN_CHARS = 300

EXTRACT_SYSTEM = """You maintain long-term memory as typed records. Read one conversation session and extract the durable facts the participants state about themselves, their lives, possessions, habits, relationships, plans, numbers and events. In a chat between a user and an assistant, ignore the assistant's advice and general knowledge. Ignore turns that reveal nothing lasting about a participant.

Return JSON only: {"facts":[{"subject":"<key>","kind":"state" or "event","statement":"<one short sentence>"}]}.
- kind "state" is a lasting attribute whose current value can change later: where someone lives, their job, what they own, what they prefer, a relationship, a plan, an opinion. A later state on the same subject replaces the earlier one, so one state record per subject per session: merge several details about one subject into one statement.
- kind "event" is something that happened or will happen at a time: a visit, a purchase, a trip, a conversation, a milestone, an activity. Events are never replaced, so record each distinct event separately and put its time inside the statement whenever the session gives one, absolute or relative to the session date ("on 7 May 2023", "last weekend", "next month").
- subject is a stable dotted key naming the thing the fact is about. It starts with the person: "user" in a user-assistant chat, otherwise the speaker's lowercase first name. States name the attribute: "user.car", "user.job.employer", "caroline.friend.rachel.home". Events name the kind of event: "caroline.event.support_group_visit", "melanie.event.painting". Reuse a key from the known list when a state is about the same thing, even if the value changed. Never encode the value in the key.
- statement is one short sentence in the third person naming the person ("The user ...", "Caroline ...") with every concrete value kept exactly (numbers, names, places, dates, counts, durations).
- Return {"facts":[]} when the session has no durable facts."""

SELECT_SYSTEM = """You select memory records for answering a question. You are given the question, the date it is asked, and an index of a person's current memory records: one line per record in the form "subject | recorded <date> | statement". Return JSON only: {"subjects":["<subject key>", ...]} listing the records needed to answer the question, most relevant first, at most %d.
- Include a record when it could hold the answer or a value the answer depends on (a date, a count, a name, a place), and every record on the question's topic: the events, plans and states about the same thing, since the answer may need to be assembled or dated from several of them.
- Read past wording: a question about what someone researched, chose, bought or did is answered by records about applications, visits, purchases, plans or activities on that topic.
- When in doubt, include the record: an extra record costs little, a missing one costs the answer. Use the full allowance when the topic has that many records.
- Return {"subjects":[]} only when nothing in the index concerns the question."""

BELIEF_SELECT_SYSTEM = """You are a precise retriever over a person's stored beliefs. The candidates already passed an exact filter on scope and on the names the query uses. Given the query and the candidates (id, title, names, content), return JSON only: {"beliefs":["<id>", ...]} with every candidate the query names, refers to, or depends on, most relevant first, at most %d. Keep a candidate when the query mentions one of its names, even when the query rejects or negates that thing (a belief about the replacement still answers it) and even when the query is about several things at once. Drop a candidate only when the query does not concern it at all. Return {"beliefs":[]} when none applies."""


LABEL_SYSTEM = """You classify one stored memory so it can be filtered by exact tags. Read the memory text and return JSON only with one key per label below. A "value" label takes exactly one of its listed values. A "multi-text" label takes a list of strings, each written in lowercase as words separated by single spaces with no punctuation; return [] when nothing applies.

Labels:
%s"""

_WORD_RE = re.compile(r"[a-z0-9]+")


def _name_form(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").lower()))


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _similar(a: str, b: str, threshold: float = 0.45) -> bool:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _safe_tag(value: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _safe_subject(value: str) -> str:
    key = re.sub(r"[^a-z0-9._-]+", "_", str(value or "").strip().lower()).strip("_")
    return key[:80]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    match = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:.*?(\d{1,2}):(\d{2}))?", text)
    if match:
        y, mo, d, h, mi = match.groups()
        return datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), tzinfo=timezone.utc)
    return None


def _json_object(text: str) -> dict:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tags_of(proof: dict) -> list[str]:
    out: list[str] = []
    for holder in (proof.get("meta") or {}, (proof.get("options") or {}).get("meta") or {}):
        tags = holder.get("tags")
        if isinstance(tags, list):
            out.extend(str(t).strip() for t in tags if str(t).strip())
    seen: set[str] = set()
    unique = []
    for tag in out:
        low = tag.lower()
        if low not in seen:
            seen.add(low)
            unique.append(tag)
    return unique


def _content_of(proof: dict) -> str:
    candidates = [
        (proof.get("publicContent") or {}).get("content"),
        (proof.get("resolvedContent") or {}).get("value"),
        proof.get("content"),
        (proof.get("data") or {}).get("content"),
        (proof.get("privateContent") or {}).get("content"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    for verifier in proof.get("verifiedVerifiers") or []:
        value = ((verifier or {}).get("data") or {}).get("content")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class _Record:
    __slots__ = ("qhash", "subject", "date", "doc_id", "belief", "statement", "content", "tags", "created_at", "title", "resolved")

    def __init__(self, proof: dict):
        tags = _tags_of(proof)
        lower = {t.lower(): t for t in tags}

        def tag_value(prefix: str) -> str:
            for low, original in lower.items():
                if low.startswith(prefix):
                    return original[len(prefix) :]
            return ""

        self.qhash = str(proof.get("qHash") or "").lower()
        self.subject = tag_value(SUBJECT_TAG)
        self.date = tag_value(DATE_TAG)
        self.doc_id = tag_value(DOC_TAG)
        self.belief = tag_value(BELIEF_TAG)
        self.tags = tags
        self.content = _content_of(proof)
        self.title = str((proof.get("meta") or {}).get("title") or proof.get("title") or "")
        self.resolved = bool(self.content)
        if not self.content and self.title:
            # Content resolves shortly after the write; the title carries the statement until then.
            self.content = f"[{self.date or 'undated'}] {self.title}" if self.subject else self.title
        self.statement = re.sub(r"^\[[^\]]*\]\s*[^:]*:\s*", "", self.content) if self.subject else self.content
        self.created_at = proof.get("createdAt") or 0


class ProofableMemoryProvider(MemoryProvider):
    name = "proofable"
    description = (
        "Proofable typed memory (hosted). Each session is distilled into dated, subject-keyed private "
        "records; a newer record on a subject supersedes the older one and only current records are read. "
        "Retrieval applies the dataset's exact tag filter, then a model selects the subjects for the "
        "question and may select none. k=20."
    )
    kind = "cloud"
    provider = "proofable"
    variant = None
    link = "https://proofable.me"
    logo = "https://proofable.me/favicon.ico"
    concurrency = 4
    supports_filters = True

    def __init__(self) -> None:
        self.k = int(os.environ.get("PROOFABLE_TOP_K", "20"))
        self.unit_concurrency = int(os.environ.get("PROOFABLE_UNIT_CONCURRENCY", "4"))
        self.model = os.environ.get("PROOFABLE_MEMORY_MODEL", "gemini-3.1-flash-lite")
        self._api = os.environ.get("PROOFABLE_API_URL", "https://api.proofable.me").rstrip("/")
        self._http: httpx.Client | None = None
        self._wallet: str | None = None
        self._genai = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._budgets: dict[str, dict] = {}
        self._records: dict[str, list[_Record]] = {}
        self._writes = 0
        self._labels: list[dict] | None = None

    # ------------------------------------------------------------------ connections

    def _client(self) -> httpx.Client:
        if self._http is None:
            key = os.environ.get("PROOFABLE_ACCESS_KEY", "").strip()
            if not key:
                raise RuntimeError("PROOFABLE_ACCESS_KEY is not set")
            self._http = httpx.Client(
                base_url=self._api,
                headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                timeout=httpx.Timeout(180.0, connect=30.0),
            )
        return self._http

    def _wallet_address(self) -> str:
        if self._wallet is None:
            data = self._request("GET", "/api/v1/auth/me").get("data") or {}
            wallet = str(data.get("address") or data.get("primaryAccount") or "").lower()
            if not re.fullmatch(r"0x[0-9a-f]{40}", wallet):
                raise RuntimeError("the access key resolved to no wallet")
            self._wallet = wallet
        return self._wallet

    def _gemini(self):
        if self._genai is None:
            from google import genai

            self._genai = genai.Client()
        return self._genai

    def _request(self, method: str, path: str, body: dict | None = None, *, write: bool = False) -> dict:
        client = self._client()
        bucket = "write" if write else "api"
        delay = 2.0
        for attempt in range(8):
            self._pace(bucket)
            try:
                response = client.request(method, path, json=body)
            except httpx.HTTPError as error:
                if attempt == 7:
                    raise
                logger.warning("proofable %s %s: %s (retry %d)", method, path, error, attempt + 1)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            self._observe(bucket, response.headers)
            if response.status_code == 429:
                wait = float(response.headers.get("retry-after") or 0) or self._seconds_to_reset(bucket) or delay
                logger.info("proofable rate limit on %s; waiting %.0fs", path, wait)
                time.sleep(min(max(wait, 1.0), 120.0))
                continue
            if response.status_code >= 500:
                if attempt == 7:
                    response.raise_for_status()
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code >= 400 or payload.get("success") is False:
                raise RuntimeError(f"{method} {path} -> {response.status_code} {json.dumps(payload.get('error') or payload)[:300]}")
            return payload
        raise RuntimeError(f"{method} {path}: gave up after retries")

    # ------------------------------------------------------------------ write budget

    # Two budgets, read from the IETF ratelimit headers on every response: new-proof writes
    # (POST /verification) and every other API call. Each is a per-minute allowance on the
    # paying account; a request waits for the window to reset rather than drawing a 429.
    def _observe(self, bucket: str, headers: httpx.Headers) -> None:
        header = headers.get("ratelimit") or ""
        remaining = re.search(r"remaining=(\d+)", header)
        reset = re.search(r"reset=(\d+)", header)
        with self._lock:
            state = self._budgets.setdefault(bucket, {"remaining": None, "reset_at": 0.0})
            if remaining:
                state["remaining"] = int(remaining.group(1))
            if reset:
                state["reset_at"] = time.time() + int(reset.group(1))

    def _seconds_to_reset(self, bucket: str) -> float:
        with self._lock:
            state = self._budgets.get(bucket) or {}
            return max(0.0, float(state.get("reset_at") or 0.0) - time.time())

    def _pace(self, bucket: str) -> None:
        while True:
            with self._lock:
                state = self._budgets.setdefault(bucket, {"remaining": None, "reset_at": 0.0})
                remaining, reset_at = state["remaining"], state["reset_at"]
                if remaining is None or remaining > 0 or time.time() >= reset_at:
                    if remaining is not None and remaining > 0:
                        state["remaining"] = remaining - 1
                    return
                wait = reset_at - time.time()
            time.sleep(min(max(wait, 0.5), 65.0))

    # ------------------------------------------------------------------ records

    def _write(self, content: str, title: str, tags: list[str]) -> str:
        wallet = self._wallet_address()
        body = {
            "walletAddress": wallet,
            "signature": "",
            "signatureMethod": "eip191",
            "signedTimestamp": int(time.time() * 1000),
            "chainId": CHAIN_ID,
            "verifierIds": ["ownership-basic"],
            "data": {"owner": wallet, "content": content, "contentType": "text/plain"},
            "options": {
                "privacyLevel": "private",
                "storeOriginalContent": True,
                "meta": {"title": title[:120], "tags": tags, "source": "agent-memory-benchmark"},
            },
        }
        payload = self._request("POST", "/api/v1/verification", body, write=True)
        qhash = str((payload.get("data") or {}).get("qHash") or "").lower()
        if not re.fullmatch(r"0x[0-9a-f]{64}", qhash):
            raise RuntimeError(f"verification returned no qHash: {json.dumps(payload)[:200]}")
        with self._lock:
            self._writes += 1
            if self._writes % 100 == 0:
                logger.info("proofable: %d records written", self._writes)
        return qhash

    def _mark_superseded(self, old: str, new: str) -> None:
        self._request(
            "PATCH",
            f"/api/v1/proofs/{old}/metadata",
            {
                "addTags": [f"{SUPERSEDED_BY_TAG}{new}"],
                "removeTags": [],
                "ownerWallet": self._wallet_address(),
                "signedTimestamp": int(time.time() * 1000),
                "signature": "",
                "signatureMethod": "eip191",
            },
        )

    def _revoke(self, qhash: str) -> None:
        self._request("POST", f"/api/v1/proofs/revoke-self/{qhash}", {})

    def _list(self, user_id: str) -> list[_Record]:
        wallet = self._wallet_address()
        tag = f"{USER_TAG}{user_id}"
        out: list[_Record] = []
        offset = 0
        while True:
            payload = self._request("GET", f"/api/v1/proofs/by-wallet/{wallet}?tags={httpx.QueryParams({'t': tag})['t']}&limit=1000&offset={offset}")
            data = payload.get("data") or {}
            page = data.get("proofs") or []
            out.extend(_Record(p) for p in page)
            pagination = data.get("pagination") or data
            if not pagination.get("hasMore") or not page:
                break
            offset = int(pagination.get("nextOffset") or offset + len(page))
            if offset > 50_000:
                break
        return [r for r in out if r.qhash]

    def _current(self, user_id: str) -> list[_Record]:
        with self._lock:
            cached = self._records.get(user_id)
        if cached is not None:
            return cached
        records = self._list(user_id)
        with self._lock:
            self._records[user_id] = records
        return records

    # ------------------------------------------------------------------ model

    def _llm_json(self, system: str, user: str) -> dict:
        from google.genai import types

        client = self._gemini()
        delay = 3.0
        for attempt in range(6):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0,
                        response_mime_type="application/json",
                        max_output_tokens=4096,
                    ),
                )
                return _json_object(response.text or "")
            except Exception as error:  # noqa: BLE001 - provider errors are transient here
                if attempt == 5:
                    logger.warning("proofable model call failed: %s", error)
                    return {}
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return {}

    @staticmethod
    def _turns(doc: Document) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        raw = doc.messages
        if raw is None:
            try:
                parsed = json.loads(doc.content)
                raw = parsed if isinstance(parsed, list) else None
            except (ValueError, TypeError):
                raw = None
        if not raw:
            return [("user", str(doc.content or ""))]
        for turn in raw:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or turn.get("role") or "user").strip()
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if text:
                turns.append((speaker, text))
        return turns

    def _extract(self, doc: Document, known: list[str]) -> list[dict]:
        turns = self._turns(doc)
        rendered = []
        for speaker, text in turns:
            if speaker.lower() == "assistant" and len(text) > ASSISTANT_TURN_CHARS:
                text = text[:ASSISTANT_TURN_CHARS] + " […]"
            rendered.append({"speaker": speaker, "text": text})
        known_block = ""
        if known:
            known_block = "Known subject keys so far:\n" + "\n".join(f"- {k}" for k in known) + "\n\n"
        prompt = f"{known_block}Session date: {doc.timestamp or 'unknown'}\nSession:\n{json.dumps(rendered, ensure_ascii=False)}"
        parsed = self._llm_json(EXTRACT_SYSTEM, prompt)
        facts = []
        for fact in parsed.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            subject = _safe_subject(fact.get("subject"))
            statement = re.sub(r"\s+", " ", str(fact.get("statement") or "")).strip()[:400]
            kind = "event" if str(fact.get("kind") or "").strip().lower() == "event" else "state"
            if subject and statement:
                if kind == "event":
                    # Events are never replaced: each one is its own subject.
                    digest = hashlib.sha1(f"{doc.id}|{statement}".encode("utf-8")).hexdigest()[:8]
                    subject = f"{subject}.{digest}"[:80]
                facts.append({"subject": subject, "statement": statement, "kind": kind})
        return facts

    # ------------------------------------------------------------------ labels

    def set_extraction_labels(self, labels: list[dict] | None) -> None:
        """Dataset-declared labels (PrecisionMemBench: state and name). Each becomes an exact
        tag on the stored record, so the dataset's retrieval filter runs as tag matching."""
        self._labels = [l for l in (labels or []) if isinstance(l, dict) and l.get("key")] or None

    def _label_tags(self, text: str) -> list[str]:
        if not self._labels:
            return []
        lines = []
        for label in self._labels:
            kind = label.get("type") or "value"
            desc = str(label.get("description") or "").strip()
            values = label.get("values") or []
            allowed = ", ".join(f"'{v.get('value')}' ({v.get('description')})" for v in values if isinstance(v, dict))
            lines.append(f"- {label['key']} ({kind}){': ' + desc if desc else ''}{(' Values: ' + allowed) if allowed else ''}")
        parsed = self._llm_json(
            LABEL_SYSTEM % "\n".join(lines),
            "The memory text opens with the subject's canonical name followed by its alternative names "
            "(aliases, shorthand, the names of what it replaced or rejects) and only then the description. "
            "Every word-run before the first full sentence is a name of the subject: list each one separately, "
            f"in addition to the forms the description gives.\n\nMemory:\n{text}",
        )
        tags: list[str] = []
        for label in self._labels:
            key = str(label["key"])
            value = parsed.get(key)
            if (label.get("type") or "value") == "multi-text":
                forms = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
                for form in forms:
                    normalised = _name_form(form)
                    if normalised and f"{key}:{normalised}" not in tags:
                        tags.append(f"{key}:{normalised}")
            else:
                allowed = {str(v.get("value")) for v in (label.get("values") or []) if isinstance(v, dict)}
                chosen = str(value or "").strip().lower()
                if allowed and chosen not in allowed:
                    chosen = ""
                if not chosen and label.get("optional") is False and allowed:
                    chosen = "current" if "current" in allowed else sorted(allowed)[0]
                if chosen:
                    tags.append(f"{key}:{chosen}")
        return tags

    # ------------------------------------------------------------------ interface

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._wallet_address()

    def ingest(self, documents: list[Document]) -> None:
        by_user: dict[str, list[Document]] = {}
        for doc in documents:
            by_user.setdefault(doc.user_id or "bench", []).append(doc)
        for user_id, docs in by_user.items():
            self._ingest_user(user_id, docs)

    def _ingest_user(self, user_id: str, docs: list[Document]) -> None:
        existing = self._list(user_id)
        if existing:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda r: self._revoke(r.qhash), existing))
        with self._lock:
            self._records.pop(user_id, None)

        ordered = sorted(enumerate(docs), key=lambda pair: (_parse_dt(pair[1].timestamp) or datetime.max.replace(tzinfo=timezone.utc), pair[0]))
        user_tag = f"{USER_TAG}{_safe_tag(user_id)}"
        current: dict[str, str] = {}
        known_kinds: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _, doc in ordered:
                doc_tag = f"{DOC_TAG}{_safe_tag(doc.id)}"
                date_tag = f"{DATE_TAG}{_safe_tag(doc.timestamp or '', 48)}"
                if doc.tags:
                    title = str(doc.content or "").strip().splitlines()[0][:120] if doc.content else doc.id
                    tags = ["facet:memory", BENCH_TAG, user_tag, doc_tag, date_tag, f"{BELIEF_TAG}{_safe_tag(doc.id)}"]
                    tags.extend(_safe_tag(t) for t in doc.tags)
                    tags.extend(self._label_tags(str(doc.content or "")))
                    self._write(str(doc.content or ""), title, tags)
                    continue
                facts = self._extract(doc, [s for s, k in known_kinds.items() if k == "state"])
                if not facts:
                    continue

                def write_fact(fact: dict) -> tuple[str, str, str | None]:
                    subject = fact["subject"]
                    previous = current.get(subject) if fact["kind"] == "state" else None
                    content = f"[{doc.timestamp or 'undated'}] {subject}: {fact['statement']}"
                    tags = ["facet:memory", BENCH_TAG, user_tag, doc_tag, f"{SUBJECT_TAG}{subject}", date_tag, f"amb:kind:{fact['kind']}"]
                    if previous:
                        tags.append(f"{SUPERSEDES_TAG}{previous}")
                    qhash = self._write(content, f"{subject}: {fact['statement']}", tags)
                    if previous:
                        try:
                            self._mark_superseded(previous, qhash)
                        except Exception as error:  # noqa: BLE001
                            logger.warning("proofable: could not mark %s superseded: %s", previous[:10], error)
                    return subject, qhash, previous

                for fact, (subject, qhash, _) in zip(facts, pool.map(write_fact, facts)):
                    current[subject] = qhash
                    known_kinds[subject] = fact["kind"]
        self._wait_resolved(user_id)

    def _wait_resolved(self, user_id: str, timeout: float = 900.0) -> None:
        """Block until every record of the user reads back with its content, the way the other
        hosted providers wait for indexing: the write is acknowledged at once and the stored
        content becomes readable a little later."""
        deadline = time.time() + timeout
        while True:
            records = self._list(user_id)
            pending = [r for r in records if not r.resolved]
            if not pending:
                with self._lock:
                    self._records[user_id] = records
                return
            if time.time() >= deadline:
                logger.warning("proofable: %d of %d records for %s still resolving after %.0fs", len(pending), len(records), user_id, timeout)
                with self._lock:
                    self._records[user_id] = records
                return
            time.sleep(5.0)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
        filters: dict | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or "bench"
        records = self._current(uid)
        limit = k if filters is not None else self.k
        if not records:
            return [], {"records": 0, "selected": []}

        beliefs = [r for r in records if r.belief]
        if beliefs:
            return self._retrieve_beliefs(query, beliefs, limit, filters)
        return self._retrieve_subjects(query, records, limit, query_timestamp)

    def _retrieve_subjects(self, query: str, records: list[_Record], limit: int, query_timestamp: str | None):
        by_subject: dict[str, _Record] = {}
        for record in sorted(records, key=lambda r: r.created_at):
            if record.subject:
                by_subject[record.subject] = record
        if not by_subject:
            return [], {"records": len(records), "selected": []}
        index = "\n".join(f"{s} | recorded {r.date or 'undated'} | {r.statement}" for s, r in by_subject.items())
        prompt = f"Question date: {query_timestamp or 'unknown'}\nQuestion: {query}\n\nMemory index:\n{index}"
        parsed = self._llm_json(SELECT_SYSTEM % limit, prompt)
        chosen: list[str] = []
        for key in parsed.get("subjects") or []:
            subject = _safe_subject(key)
            if subject in by_subject and subject not in chosen:
                chosen.append(subject)
        docs = []
        for subject in chosen[:limit]:
            record = by_subject[subject]
            shown_subject = re.sub(r"\.[0-9a-f]{8}$", "", subject)
            docs.append(
                Document(
                    id=record.qhash,
                    # Statement first; provenance after it, so a reader sees the fact and not a key.
                    content=f"{record.statement} (recorded {record.date or 'undated'}; about {shown_subject})",
                    source_ids=[record.doc_id] if record.doc_id else None,
                    timestamp=record.date or None,
                )
            )
        # No raw payload: the conversation datasets put a provider's raw response into the
        # answer prompt in place of the rendered memories, and the records above are the answer.
        logger.info("proofable retrieve: %d of %d records selected", min(len(chosen), limit), len(by_subject))
        return docs, None

    def _retrieve_beliefs(self, query: str, beliefs: list[_Record], limit: int, filters: dict | None):
        candidates = beliefs
        if filters:
            candidates = self._apply_filters(candidates, filters)
        if not candidates:
            return [], {"records": len(beliefs), "filtered": 0, "selected": []}
        listing = "\n\n".join(
            f"id: {b.belief}\ntitle: {b.title}\nnames: {', '.join(t[5:] for t in b.tags if t.lower().startswith('name:')) or '-'}\n{b.content[:1200]}"
            for b in candidates
        )
        parsed = self._llm_json(BELIEF_SELECT_SYSTEM % limit, f"Query: {query}\n\nCandidate beliefs:\n{listing}")
        by_id = {b.belief: b for b in candidates}
        chosen = []
        for belief_id in parsed.get("beliefs") or []:
            key = str(belief_id).strip()
            if key in by_id and key not in chosen:
                chosen.append(key)
        docs = [
            Document(
                id=by_id[key].qhash,
                content=by_id[key].content,
                source_ids=[by_id[key].doc_id] if by_id[key].doc_id else None,
                context=f"beliefId={key}",
                timestamp=by_id[key].date or None,
            )
            for key in chosen[:limit]
        ]
        return docs, {"records": len(beliefs), "filtered": len(candidates), "selected": chosen[:limit]}

    @staticmethod
    def _apply_filters(records: list[_Record], filters: dict) -> list[_Record]:
        """The dataset's provider-neutral filter as exact tag matching on the stored tags.

        any:        groups of tags; a record must carry one tag from every group
        all:        every listed tag
        none:       no listed tag
        narrow:     one of the listed tags
        narrow_any: alternative narrowings, OR-ed; each lists candidate tags and resolves
                    them exactly, or fuzzily by trigram similarity against the stored tags
        """

        def tags_of(record: _Record) -> set[str]:
            return {t.lower() for t in record.tags}

        def groups(value) -> list[list[str]]:
            if not value:
                return []
            if all(isinstance(v, str) for v in value):
                return [[str(v).lower() for v in value]]
            return [[str(t).lower() for t in group] for group in value if group]

        out = records
        for group in groups(filters.get("any")):
            out = [r for r in out if tags_of(r) & set(group)]
        if filters.get("all"):
            required = {str(t).lower() for t in filters["all"]}
            out = [r for r in out if required <= tags_of(r)]
        if filters.get("none"):
            banned = {str(t).lower() for t in filters["none"]}
            out = [r for r in out if not (tags_of(r) & banned)]
        if filters.get("narrow"):
            wanted = {str(t).lower() for t in filters["narrow"]}
            out = [r for r in out if tags_of(r) & wanted]
        specs = [s for s in (filters.get("narrow_any") or []) if isinstance(s, dict) and s.get("tags")]
        if specs:
            kept = []
            for record in out:
                stored = tags_of(record)
                hit = False
                for spec in specs:
                    wanted = [str(t).lower() for t in spec["tags"]]
                    if spec.get("resolve") == "fuzzy":
                        prefix = wanted[0].split(":", 1)[0] + ":" if wanted and ":" in wanted[0] else ""
                        stored_forms = [t.split(":", 1)[1] for t in stored if prefix and t.startswith(prefix)]
                        hit = any(_similar(w.split(":", 1)[1], s) for w in wanted for s in stored_forms)
                    else:
                        hit = bool(stored & set(wanted))
                    if hit:
                        break
                if hit:
                    kept.append(record)
            out = kept
        return out

    def cleanup(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
