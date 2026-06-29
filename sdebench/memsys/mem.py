"""sdebench local file-based memory system.

A memory system that INGESTS a project's knowledge from wherever it lives — git commit rationales
(H), CONVENTIONS.md (K), and past user conversations (F) — into a single local store of files, then
RETRIEVES the entries relevant to a task and surfaces them. The point: the agent gets the relevant
decision regardless of which source it lived in, and regardless of whether it would have thought to
look (the two failure modes the vanilla baseline shows).

Store: a JSONL file at STORE. Each entry = {project, kind, title, text}. The store is shared and
seeded with EVERY project's decisions, so retrieval must discriminate the relevant one from
distractors (other domains) — it is not hand-fed the answer.

ingest_project(repo, conversations, project) -> entries from git/doc/conversation
recall(query, k)                              -> top-k entry strings, ranked by term overlap (TF-IDF-ish)
"""
import json, re, subprocess, math
from pathlib import Path
from collections import Counter

STORE = Path("/tmp/sdebench_mem/store.jsonl")

_STOP = set("a an the of to and or for in on at by is are be with this that it its as we our you your "
            "do not no don dont should must always never use using used here there from into onto than "
            "them they i he she his her one two three when what which how why if then so but also can".split())


def _tok(text):
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if w not in _STOP and len(w) > 2]


# ── ingest ──────────────────────────────────────────────────────────────────
_NOISE_SUBJ = re.compile(r"^(scaffold|tests?|readme|changelog|release|chore|ci|style|bump|merge|"
                         r"refactor: simplify|feat: [a-z_]+$|docs: project conventions)", re.I)


def _git_decisions(repo):
    """Commits that carry a rationale body = recorded decisions (H source)."""
    out = subprocess.run(["git", "-C", str(repo), "log", "--format=%s%x1f%b%x1e"],
                         capture_output=True, text=True).stdout
    entries = []
    for chunk in out.split("\x1e"):
        if not chunk.strip():
            continue
        subj, _, body = chunk.partition("\x1f")
        subj, body = subj.strip(), body.strip()
        if body and not _NOISE_SUBJ.match(subj):
            entries.append({"kind": "decision", "title": subj, "text": (subj + " — " + body).strip()})
    return entries


def _doc_conventions(repo):
    """CONVENTIONS.md sections (K source)."""
    p = Path(repo) / "CONVENTIONS.md"
    if not p.exists():
        return []
    entries = []
    for sec in re.split(r"\n##+ ", p.read_text()):
        sec = sec.strip()
        if not sec or sec.lower().startswith("engineering conventions"):
            continue
        title, _, rest = sec.partition("\n")
        entries.append({"kind": "convention", "title": title.strip(), "text": sec.replace("\n", " ").strip()})
    return entries


def _conversation_prefs(conversations):
    """User turns that state a preference/correction (F source)."""
    entries = []
    for turn in conversations or []:
        if turn.get("role") == "user" and len(turn.get("text", "")) > 60:
            entries.append({"kind": "conversation", "title": "user preference",
                            "text": "From an earlier session, the user said: " + turn["text"].strip()})
    return entries


def ingest_project(repo, conversations, project):
    entries = _git_decisions(repo) + _doc_conventions(repo) + _conversation_prefs(conversations)
    for e in entries:
        e["project"] = project
    return entries


def write_store(entries):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    seen, uniq = set(), []
    for e in entries:
        key = (e["kind"], e["text"])
        if key not in seen:
            seen.add(key); uniq.append(e)
    STORE.write_text("\n".join(json.dumps(e) for e in uniq) + "\n")
    return uniq


def load_store():
    if not STORE.exists():
        return []
    return [json.loads(line) for line in STORE.read_text().splitlines() if line.strip()]


# ── recall ──────────────────────────────────────────────────────────────────
def recall(query, k=2):
    store = load_store()
    if not store:
        return []
    docs = [_tok(e["text"]) for e in store]
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    n = len(docs)
    idf = {w: math.log(1 + n / df[w]) for w in df}
    q = Counter(_tok(query))
    scored = []
    for e, d in zip(store, docs):
        tf = Counter(d)
        score = sum(q[w] * tf[w] * idf.get(w, 0) ** 2 for w in q)
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [f"[{e['kind']}] {e['text']}" for s, e in scored[:k] if s > 0]
