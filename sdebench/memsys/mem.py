"""sdebench local file-based memory system.

A memory system that INGESTS a project's knowledge from wherever it lives — git commit rationales
(H) and past user conversations (F) — into a single local store of files, then
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


# Identifiers used as calls `name(` / defs, members `.name`, or assignments `name =`. These forms
# hold across Python/JS/Go/Rust/Java/… so extraction is LANGUAGE-AGNOSTIC — no parser, no grammar.
_KW = set("def class function func fn return import from export public private static void const let "
          "var val if else elif for while switch case new self this true false none null nil print".split())


def _code_symbols(text):
    """Language-agnostic identifiers in any code/diff/prose text. Two lexical signals that stay
    specific (they don't fire on ordinary prose words the way `.name`/`name =` would):
      - call / definition position `name(` — captures `getlist`, `parse_flag`, `OrderedMultiDict`;
      - shape (snake_case / camelCase / CONST) — captures `parse_flag`, `under2camel`, `MAX_ATTEMPTS`.
    A change to `__setitem__` still surfaces `getlist` because `getlist(` appears in the same file."""
    text = text or ""
    syms = set(re.findall(r"([A-Za-z_]\w{2,})\s*\(", text))       # call or definition: name(
    syms |= set(_text_symbols(text))                              # snake / camel / CONST shape
    return {s.lower() for s in syms if s.lower() not in _STOP and s.lower() not in _KW}


def _diff_symbols(repo, sha):
    """Identifiers in the commit's DIFF hunks only — scoped to what the commit actually changed, in a
    single git call regardless of file count (a 62-file commit costs one `show`, not 62). No file cap,
    no whole-file scrape: uniform and cheap for every commit in a real history."""
    diff = subprocess.run(["git", "-C", str(repo), "show", "-U3", "--format=", sha],
                          capture_output=True, text=True, errors="replace").stdout
    return _code_symbols(diff)


def _note_symbols(text, repo_syms):
    """Symbols a note references: identifiers used in the text (code-shaped or call position) PLUS any
    bare token that is a known repo symbol (catches `slugify` mentioned in prose)."""
    syms = set(_text_symbols(text)) | _code_symbols(text)
    if repo_syms:
        syms |= {t for t in _tok(text) if t in repo_syms}
    return syms


def _git_note(repo, sha, subj, body, repo_syms):
    """One note per commit, ingested IDENTICALLY whether it's upstream history or a task's own commit:
    the message text + the identifiers in its diff. No commit is privileged or pre-selected."""
    text = (subj + " — " + body).strip() if body else subj
    syms = _note_symbols(text, repo_syms) | _diff_symbols(repo, sha)
    return {"kind": "decision", "source": "git", "title": subj, "text": text, "symbols": sorted(syms)}


def _all_commits(repo, base_ref=None, head="HEAD"):
    """Every non-trivial commit (base_ref..head, or all of head) as (sha, subj, body)."""
    rng = f"{base_ref}..{head}" if base_ref else head
    out = subprocess.run(["git", "-C", str(repo), "log", rng, "--format=%H%x1f%s%x1f%b%x1e"],
                         capture_output=True, text=True, errors="replace").stdout
    commits = []
    for chunk in out.split("\x1e"):
        if not chunk.strip():
            continue
        parts = (chunk.split("\x1f") + ["", "", ""])[:3]
        sha, subj, body = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if subj and not _NOISE_SUBJ.match(subj):
            commits.append((sha, subj, body))
    return commits


def ingest_history_noise(repo, repo_syms=None, head="HEAD"):
    """Inherited upstream history — EVERY commit ingested identically to a task's own commit (message +
    diff symbols), no selection, no file cap. Seeded once for the shared store."""
    if repo_syms is None:
        repo_syms = set(_repo_symbols(repo))
    return [_git_note(repo, sha, subj, body, repo_syms)
            for sha, subj, body in _all_commits(repo, None, head)]




def _gemini_key():
    import os
    k = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        envf = root / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.strip().startswith(("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY")) and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')
    return None


_SUM_CACHE = Path("/tmp/sdebench_mem/summary_cache.json")


def _llm_summarize(session_text):
    """Process a verbose session into a concise feedback DECISION NOTE (the 'user-feedback story').
    Cached by content hash; returns None on any failure so the caller can fall back to raw text."""
    import hashlib, json as _json, urllib.request
    h = hashlib.sha1(session_text.encode()).hexdigest()
    cache = {}
    if _SUM_CACHE.exists():
        cache = _json.loads(_SUM_CACHE.read_text())
    if h in cache:
        return cache[h]
    key = _gemini_key()
    if not key:
        return None
    prompt = (
        "You are a long-term memory system for a coding team. Summarize the following work session "
        "as a concise DECISION NOTE for future reference. Capture: what the user wanted, what was "
        "TRIED and REJECTED (and why), and the FINAL decision/rule that was settled on. Be specific — "
        "preserve any exact values, identifiers, modes, or thresholds that were decided, quoting them "
        "verbatim from the session rather than inventing or recomputing them. 1-3 sentences, no "
        "preamble, no transcript.\n\nSESSION:\n" + session_text)
    url = ("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + key)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        req = urllib.request.Request(url, data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = _json.load(urllib.request.urlopen(req, timeout=60))
        out = r["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None
    cache[h] = out
    _SUM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SUM_CACHE.write_text(_json.dumps(cache))
    return out


def _conversation_story(conversations, repo_syms=None):
    """Ingest a past session as ONE summarized feedback STORY, not scattered turns.

    Real sessions are verbose (the assistant does the work inline; the decision is buried in a long
    turn or a terse user reply). We render the whole exchange, then LLM-summarize it into a decision
    note that captures what was tried/rejected and the rule settled on. Falls back to the raw
    exchange if no LLM is available. Symbols come from the RAW exchange (richer than the summary), so
    the note is reachable by the code entities it discusses."""
    turns = [t for t in (conversations or []) if t.get("text")]
    if not turns:
        return []
    raw = " ".join(f"{t['role'].upper()}: {t['text'].strip()}" for t in turns)
    summary = _llm_summarize(raw) or raw
    return [{"kind": "conversation", "source": "session", "title": "feedback decision note",
             "text": "Past user feedback (summarized): " + summary,
             "symbols": sorted(_note_symbols(raw, repo_syms))}]


_SRC_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cc",
            ".cpp", ".hpp", ".cs", ".kt", ".scala", ".php", ".swift", ".m", ".mm"}


def _repo_symbols(repo):
    """The project's identifier vocabulary — the entities a decision is *about* — harvested
    language-agnostically (call/member/assignment forms) from its source files, no per-language parser."""
    syms = set()
    for p in Path(repo).rglob("*"):
        if p.suffix.lower() not in _SRC_EXT or "test" in p.name.lower() or not p.is_file():
            continue
        syms |= _code_symbols(p.read_text(errors="ignore"))
    return sorted(s for s in syms if len(s) > 2)


def _text_symbols(text):
    """Identifier-like tokens in free text (snake_case / CamelCase / CONST) — for entries without a repo."""
    syms = set(re.findall(r"\b[a-z]+_[a-z_]+\b", text or ""))
    syms |= set(re.findall(r"\b[A-Z][a-z]+[A-Z]\w+\b", text or ""))
    syms |= set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text or ""))
    return sorted({s.lower() for s in syms})


def ingest_project(repo, conversations, project, base_ref=None):
    """A task's OWN contribution to the store: its commits (base_ref..HEAD) — ingested with the SAME
    _git_note path as all upstream history, no privileged enrichment — plus its past conversation."""
    repo_syms = set(_repo_symbols(repo))
    notes = [_git_note(repo, sha, subj, body, repo_syms)
             for sha, subj, body in _all_commits(repo, base_ref)]
    notes += _conversation_story(conversations, repo_syms)
    for e in notes:
        e["project"] = project
    return notes


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
    # symbol signal: which defined code symbols does the bug report name?
    all_syms = set().union(*[set(e.get("symbols", [])) for e in store]) if store else set()
    q_syms = {w for w in q if w in all_syms}
    SYMBOL_BOOST = 1000.0   # a named symbol match dominates generic term overlap
    scored = []
    for e, d in zip(store, docs):
        tf = Counter(d)
        score = sum(q[w] * tf[w] * idf.get(w, 0) ** 2 for w in q)
        score += SYMBOL_BOOST * len(q_syms & set(e.get("symbols", [])))
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [f"[{e['kind']}] {e['text']}" for s, e in scored[:k] if s > 0]
