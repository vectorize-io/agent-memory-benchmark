"""
PrecisionMemBench (https://github.com/tenurehq/precisionmembench).
Paper: https://arxiv.org/abs/2605.11325

A multi-dimensional *retrieval* benchmark for memory systems. Unlike every other
dataset in this repo, it does not ask an LLM to answer a question and judge the
answer — it asserts, per case, exactly which belief IDs a memory system must
return and which it must not. Noise is a hard failure, not an invisible cost.

Structure
---------
Seed corpus = 35 beliefs (34 for `test-user`, 1 for `other-user`) spanning two
domain scopes, a supersession chain, and a cross-user leak fixture.
Cases       = 77 single-turn retrieval cases across 13 categories.

Documents  = one per belief. ID = belief `_id` (e.g. "b-db-decision").
             content = upstream `beliefToText()` (canonical_name + aliases +
             content + why_it_matters), byte-identical to what upstream seeds.
Queries     = one per case. ID = `caseId`.

Scoring (ported from upstream `src/retrieval.external.eval.test.ts`, via the
vendored de-ava'd `runCases.ts` in garrytan/gbrain-evals)
--------------------------------------------------------------------------
Upstream splits the retrieved context into four tiers, and only ONE of them is
provider-dependent:

  personaPrelude  — a fixed constant in the harness
  pinnedFacts     — derived locally from the seed corpus (pinned + in scope)
  openQuestions   — derived locally from the seed corpus (type=open_question)
  relevantBeliefs — the memory provider's search results  ← the actual measurement

That is why upstream reports "structural passes" separately: a provider that
returns nothing still passes cases whose assertions only touch the local tiers.
This port reproduces that split exactly, so `active_pass` in each result's meta
is the number that is comparable to upstream's "Active passes" column.

Adaptation deltas (see ATTRIBUTION below)
-----------------------------------------
1. Belief-ID resolution. Upstream's provider contract requires `/search` to
   echo back the belief ID it was seeded with. amb providers return their own
   memory IDs (Hindsight returns fact IDs, mem0 returns mem IDs), so retrieved
   documents are mapped back to beliefs by a layered resolver — see
   `_resolve_belief_ids`. Each result records which resolution path fired
   (`meta["resolution"]`) so a run can be audited for resolver artifacts.
2. The `maxBeliefs` budget is applied after resolution, not as the provider's
   result limit. Providers that store extracted facts return several memories per
   belief, so truncating their raw results to 20 would cut the belief list far
   below the budget upstream intends; resolving first and then keeping the top 20
   *distinct* beliefs is the closest equivalent — and the more generous reading.
3. `scope` is passed to the provider only as part of the case's user scoping;
   amb's retrieve() has no scope parameter. Upstream sends it in the `/search`
   body, where most reference wrappers (including its own Hindsight wrapper)
   ignore it too.

Upstream: tenurehq/precisionmembench @ b95d6ab (MIT), fixtures fetched pinned
to that commit. Reference port: garrytan/gbrain-evals `eval/precisionmembench/`.
"""
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

_COMMIT = "b95d6abb471c0d591c440172283dd74e7af000df"
_RAW = f"https://raw.githubusercontent.com/tenurehq/precisionmembench/{_COMMIT}/fixtures"
_FIXTURES = {
    "beliefs.seed.json": f"{_RAW}/beliefs.seed.json",
    "retrieval.cases.json": f"{_RAW}/retrieval.cases.json",
}

SPLITS = ["single-turn"]

DEFAULT_USER_ID = "test-user"

# Default context budget — verbatim from upstream baseAdapter.ts DEFAULT_BUDGET.
_DEFAULT_BUDGET = {"maxBeliefs": 20, "maxPinnedFacts": 10, "maxQuestions": 15}

# Verbatim from upstream retrieval.external.eval.test.ts EVAL_PERSONA.
_EVAL_PERSONA_UNIVERSAL = (
    "You prefer direct answers without preamble. You push back when plans have "
    "problems rather than defaulting to agreement. You edit AI output; you do not "
    "let AI edit your prose."
)

_BELIEF_ID_RE = re.compile(r"\bb-[a-z0-9][a-z0-9-]*\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Categories whose assertions can be satisfied without any query-dependent
# retrieval are still reported, but `active_pass` (below) is what compares to
# upstream's "Active passes" column.


# ── tag vocabulary ────────────────────────────────────────────────────────────
# Beliefs are ingested carrying their own identity, scope and (LLM-classified) state as
# tags, so recall filters them server-side instead of the harness post-filtering.

_TAG_SAFE = re.compile(r"[^a-z0-9]+")


def _tag(value: str) -> str:
    return _TAG_SAFE.sub("-", value.lower()).strip("-")


def _name_forms(b: dict) -> list[str]:
    """Every surface form the user might call this belief by.

    Upstream's `/add` contract passes `aliases` as a first-class field precisely so a provider
    can index identity separately from content; these are that field plus the canonical name."""
    return [(b.get("canonical_name") or "").replace("_", " "), *(b.get("aliases") or [])]


# Open-vocabulary: the names cannot be enumerated in advance, so the extractor reads them
# off each belief's own text rather than being handed a list.
_NAME_LABEL = {
    "key": "name",
    "type": "multi-text",
    "optional": True,
    "tag": True,
    "description": (
        "Every name the SUBJECT of this fact is known by — the thing the fact is about. "
        "Include its canonical name plus every abbreviation, acronym, short form, alternative "
        "spelling and shorthand someone might type for it (for example 'Kubernetes' is also "
        "'k8s' and 'kube'), and any alternative name the text itself gives. List the surface "
        "forms themselves, not descriptions of them.\n"
        "Name only the fact's own subject. Do NOT list names of other things the fact merely "
        "mentions, refers to, or depends on — those are separate memories with their own "
        "names. A fact about how one component depends on another is named after the "
        "dependency it describes, not after either component.\n"
        "A name must pick this memory out on its own. If a candidate would equally name a "
        "different memory — because it names a component this fact merely involves — it is a "
        "mention, not a name. Prefer the longer compound form over a fragment of it.\n"
        "Write every name in lowercase, as words separated by single spaces, with no "
        "punctuation — 'CI/CD' becomes 'ci cd', 'error_handling' becomes 'error handling', "
        "'@typescript-eslint' becomes 'typescript eslint'. A caller looking a name up will "
        "normalise the same way, and the lookup is an exact match. Leave empty when the fact "
        "is not about a nameable thing."
    ),
}

def _name_candidates(query: str, max_words: int = 3, min_word: int = 1) -> list[str]:
    """The query's words and short word-runs, as candidate identity tags.

    Stateless and vocabulary-free: whatever the query says might be a name, normalised the
    same way the extractor is told to write names, so the lookup is exact array containment.
    A run of up to `max_words` covers multi-word names like "session backend"."""
    words = _WORD_RE.findall(query.lower())
    out: list[str] = []
    for n in range(1, max_words + 1):
        for i in range(len(words) - n + 1):
            run = words[i:i + n]
            if min(len(w) for w in run) < min_word:
                continue
            tag = "name:" + " ".join(run)
            if tag not in out:
                out.append(tag)
    # A tag no memory carries — a null byte would be rejected by the API.
    return out or ["name:--no-candidate--"]


_STATE_LABEL = {
    "key": "state",
    "type": "value",
    "optional": False,
    "tag": True,
    "description": (
        "Whether this belief is current guidance. Use 'historical' when the record says it is "
        "a historical record, was previously used, or has been replaced or superseded. Use "
        "'open_question' when it describes an undecided or unresolved question. Otherwise "
        "'current'."
    ),
    "values": [
        {"value": "current", "description": "Current, active guidance that should shape answers now"},
        {"value": "historical", "description": "Superseded, replaced or previously used; historical record only"},
        {"value": "open_question", "description": "An undecided or unresolved open question"},
    ],
}


def _belief_to_text(b: dict) -> str:
    """Verbatim port of upstream BaseAdapter.beliefToText (canonical_name_aliases mode)."""
    parts = [b.get("canonical_name"), *(b.get("aliases") or []), b.get("content"), b.get("why_it_matters")]
    return " ".join(p for p in parts if p)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


class PrecisionMemBenchDataset(Dataset):
    """
    PrecisionMemBench — belief-ID-level retrieval precision benchmark.

    Fixtures are auto-downloaded from GitHub (pinned commit) on first use.
    Set PRECISIONMEMBENCH_DATA_PATH to a directory holding beliefs.seed.json and
    retrieval.cases.json to use a local copy instead.
    """

    name = "precisionmembench"
    published = True
    description = "Belief-level retrieval precision: 77 cases asserting exactly which memories must and must not surface."
    splits = SPLITS
    task_type = "retrieval"
    isolation_unit = None  # per-user scoping is done by the provider's user_id filter
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2605.11325"},
        {"label": "GitHub", "url": "https://github.com/tenurehq/precisionmembench"},
        {"label": "Leaderboard", "url": "https://huggingface.co/spaces/tenurehq/precisionmembench"},
    ]

    def __init__(self):
        self._cache: dict[str, list] = {}

    # ── fixture loading ───────────────────────────────────────────────────────

    def _fixture(self, filename: str) -> list:
        if filename in self._cache:
            return self._cache[filename]
        env = os.environ.get("PRECISIONMEMBENCH_DATA_PATH")
        path = Path(env) / filename if env else dataset_cache_dir(self.name) / filename
        if not path.exists():
            if env:
                raise FileNotFoundError(f"PRECISIONMEMBENCH_DATA_PATH set but {path} is missing")
            Console().print(f"[dim]Downloading {filename} from upstream @ {_COMMIT[:7]}...[/dim]")
            urllib.request.urlretrieve(_FIXTURES[filename], path)
        data = json.loads(path.read_text())
        self._cache[filename] = data
        return data

    @property
    def _beliefs(self) -> list[dict]:
        return self._fixture("beliefs.seed.json")

    @property
    def _by_id(self) -> dict[str, dict]:
        key = "_by_id"
        if key not in self._cache:
            self._cache[key] = {b["_id"]: b for b in self._beliefs}
        return self._cache[key]

    @property
    def _cases(self) -> list[dict]:
        return self._fixture("retrieval.cases.json")

    # ── dataset interface ─────────────────────────────────────────────────────

    def categories(self, split: str) -> list[str] | None:
        seen: list[str] = []
        for c in self._cases:
            if c["category"] not in seen:
                seen.append(c["category"])
        return seen

    def category_type(self, split: str, category: str) -> str:
        # Every case queries the same 35-belief corpus; categories partition
        # queries, not documents.
        return "query"

    def supports_oracle(self) -> bool:
        # Oracle mode would drop the very beliefs the `mustExclude` assertions
        # are checking for, which would silently turn failures into passes.
        return False

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        docs = []
        for b in self._beliefs:
            bid = b["_id"]
            if ids is not None and bid not in ids:
                continue
            if user_ids is not None and b["user_id"] not in user_ids:
                continue
            docs.append(
                Document(
                    id=bid,
                    content=_belief_to_text(b),
                    user_id=b["user_id"],
                    timestamp=b.get("created_at"),
                    # Only scope, which is genuinely caller data (upstream sends it in
                    # `/add`). Identity is NOT supplied: the belief's own names are stated in
                    # its text, and the extraction LLM derives them — see extraction_labels().
                    tags=[f"scope:{_tag(sc)}" for sc in b.get("scope", [])],
                    # Upstream seeds `{beliefId, scope}` as provider metadata; keeping it
                    # here gives the resolver a marker path on providers that echo context.
                    context=f"beliefId={bid} scope={(b.get('scope') or [''])[0]}",
                )
            )
        return docs[:limit] if limit else docs

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        queries = []
        for c in self._cases:
            if category and c["category"] != category:
                continue
            expect = c.get("expect", {})
            rb = expect.get("relevantBeliefs", {}) or {}
            pf = expect.get("pinnedFacts", {}) or {}
            gold = list(dict.fromkeys([*(rb.get("shouldOnlyInclude") or []),
                                       *(rb.get("mustInclude") or []),
                                       *(pf.get("mustInclude") or [])]))
            queries.append(
                Query(
                    id=c["caseId"],
                    query=c["query"],
                    gold_ids=gold,
                    gold_answers=gold,
                    user_id=c.get("userId") or DEFAULT_USER_ID,
                    meta={
                        "category": c["category"],
                        "description": c.get("description", ""),
                        "scope": c["scope"],
                        "budget": c.get("budget") or {},
                        "expect": expect,
                        "retrieval_limit": {**_DEFAULT_BUDGET, **(c.get("budget") or {})}["maxBeliefs"],
                    },
                )
            )
        return queries[:limit] if limit else queries

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        if meta.get("category"):
            axes["Category"] = [meta["category"]]
        if meta.get("pass_type"):
            axes["Pass type"] = [meta["pass_type"]]
        return axes

    def extraction_labels(self) -> list[dict] | None:
        """Ask the provider's extractor to classify each belief's currency.

        Supersession/resolution is stated in the belief text itself ("Historical record only",
        "Stack replaced by MongoDB", "Not decided") — it is not a hidden schema field — so a
        provider can classify it from what every comparison system is given."""
        return [_STATE_LABEL, _NAME_LABEL]

    def retrieval_filter(self, query: Query) -> dict | None:
        """Hard filter for this case: right scope, current only, and — when the query names a
        belief — only beliefs carrying that identity.

        Every clause is decided by tags the provider stored at ingest, so this becomes a SQL
        WHERE inside the provider rather than a post-filter here."""
        scope_tags = [f"scope:{_tag(sc)}" for sc in query.meta.get("scope", [])]
        return {
            "any": [scope_tags] if scope_tags else [],
            # Restrict to the beliefs the query names. The words are sent as candidate tags
            # and resolved against the bank's own tag vocabulary by trigram similarity, so a
            # misspelled name ("typsecript", "eror") still reaches its memory. A query naming
            # nothing recognisable matches nothing, which is the answer for 34 of the cases.
            # Two ways in, OR-ed. Exact covers every candidate, including short names a
            # query spells correctly (k8s, ts, gha). Fuzzy covers only candidates whose
            # words are long enough to carry signal: trigram similarity on a 2-3 letter
            # token is noise — "a" resolves to auth/fp/ts/dlq, "my" to mongo — so short
            # words must match literally or not at all.
            "narrow_any": [
                {"tags": _name_candidates(query.query), "resolve": "exact"},
                {"tags": _name_candidates(query.query, min_word=4), "resolve": "fuzzy"},
            ],
            "none": ["state:historical", "state:open_question"],
            # Which belief the query is about. The query's own words and word-runs are sent
            # as candidate identity tags and matched by the existing tag filter — an indexed
            # array containment, no vocabulary to load. Abstention falls out of it: a query
            # naming nothing stored matches nothing (34 of the 77 cases expect exactly that).
        }

    # ── local (provider-independent) context tiers ────────────────────────────
    # Verbatim ports of BaseAdapter.listPinnedFacts / listPinnedOpenQuestions /
    # expandRelationParticipants. Upstream derives these from its own seed index,
    # not from the provider, and so do we.

    def _pinned_facts(self, user_id: str, scope: list[str]) -> list[dict]:
        return [
            b for b in self._beliefs
            if b["user_id"] == user_id
            and b.get("pinned") is True
            and b.get("type") != "open_question"
            and not b.get("superseded_by")
            and not b.get("resolved_at")
            and any(s in scope for s in b.get("scope", []))
        ]

    def _pinned_open_questions(self, user_id: str, scope: list[str]) -> list[dict]:
        return [
            b for b in self._beliefs
            if b["user_id"] == user_id
            and b.get("type") == "open_question"
            and b.get("pinned") is True
            and not b.get("resolved_at")
            and any(s in scope for s in b.get("scope", []))
        ]

    def _expand_relation_participants(
        self, user_id: str, relation_ids: list[str], scope: list[str], exclude: set[str]
    ) -> list[str]:
        out: list[str] = []
        for rid in relation_ids:
            rel = self._by_id.get(rid)
            if not rel or rel.get("type") != "relation":
                continue
            for pid in rel.get("participants") or []:
                if pid in exclude:
                    continue
                b = self._by_id.get(pid)
                if not b or b["user_id"] != user_id:
                    continue
                if b.get("type") == "open_question":
                    # Open-question participants belong in the openQuestions tier, not
                    # relevantBeliefs — see the `relation-type-expands-participants`
                    # case description. The vendored adapter omits this filter, but it
                    # only ever fires for a provider that actually surfaces the relation
                    # belief, which no upstream comparison system does; without it a
                    # perfect provider scores 73/77 instead of upstream's 77/77.
                    continue
                if scope and not any(s in scope for s in b.get("scope", [])):
                    continue
                out.append(pid)
        return out

    # ── belief-ID resolution ──────────────────────────────────────────────────

    def _lexical_match(self, text: str) -> str | None:
        """Last-resort attribution for providers that rewrite facts and drop all IDs.

        Matches a retrieved memory to the belief whose canonical name or alias it
        names outright, else to the belief with the strongest token overlap above
        a conservative threshold. Returns None rather than guessing weakly.
        """
        low = (text or "").lower()
        toks = _tokens(text)
        if not toks:
            return None
        best, best_score = None, 0.0
        for b in self._beliefs:
            name = (b.get("canonical_name") or "").replace("_", " ").lower()
            names = [name, *[a.lower() for a in (b.get("aliases") or [])]]
            hit = any(n and n in low for n in names)
            b_toks = _tokens(b.get("content", ""))
            jac = len(b_toks & toks) / len(b_toks | toks) if b_toks else 0.0
            score = (1.0 if hit else 0.0) + jac
            if score > best_score:
                best, best_score = b["_id"], score
        # Require either an outright name/alias hit, or a strong content overlap.
        return best if best_score >= 1.0 or best_score >= 0.6 else None

    def _resolve_belief_ids(self, docs: list[Document]) -> tuple[list[str], dict[str, int]]:
        """Map provider-returned documents onto seed belief IDs, order preserved."""
        known = self._by_id
        resolved: list[str] = []
        seen: set[str] = set()
        how: Counter = Counter()
        for d in docs:
            bid = method = None
            for sid in (getattr(d, "source_ids", None) or []):
                if sid in known:
                    bid, method = sid, "source_id"
                    break
            if bid is None and d.id in known:
                bid, method = d.id, "doc_id"
            if bid is None:
                blob = f"{d.context or ''}\n{d.content or ''}"
                for cand in _BELIEF_ID_RE.findall(blob):
                    if cand in known:
                        bid, method = cand, "marker"
                        break
            if bid is None:
                bid = self._lexical_match(d.content)
                method = "lexical" if bid else None
            if bid is None:
                how["unresolved"] += 1
                continue
            how[method] += 1
            if bid not in seen:
                seen.add(bid)
                resolved.append(bid)
        return resolved, dict(how)

    # ── scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pass_type(expect: dict, expected_relevant: set[str]) -> str:
        """Upstream's three-way pass classification (43 active / 25 structural / 9 empty).

        active          — satisfying the case requires query-dependent belief IDs.
        trivially-empty — the case expects an empty relevantBeliefs tier and asserts
                          nothing else, so returning nothing passes it.
        structural      — expects nothing relevant but still asserts something the
                          local tiers decide (scope isolation, supersession, routing).
        """
        if expected_relevant:
            return "active"
        rb = expect.get("relevantBeliefs") or {}
        pf = expect.get("pinnedFacts") or {}
        other = (bool(rb.get("mustExclude")) or bool(expect.get("openQuestions"))
                 or bool(pf.get("mustInclude")) or bool(pf.get("mustExclude"))
                 or bool(expect.get("personaPrelude")))
        return "structural" if other else "trivially-empty"

    def score_retrieval(self, query: Query, retrieved: list[Document]) -> tuple[bool, str]:
        """Score one case. Returns (passed, reason); writes metrics into query.meta.

        Assertion order and precision/recall math are a direct port of upstream's
        `scoreCases` (see module docstring).
        """
        meta = query.meta
        scope: list[str] = meta["scope"]
        expect: dict = meta["expect"]
        user_id = query.user_id or DEFAULT_USER_ID
        budget = {**_DEFAULT_BUDGET, **(meta.get("budget") or {})}
        cap = budget["maxBeliefs"]

        pinned = self._pinned_facts(user_id, scope)
        questions = self._pinned_open_questions(user_id, scope)
        pinned_ids_all = {b["_id"] for b in pinned}

        # buildContext: no search when the query is blank or the budget is zero.
        if query.query.strip() and cap > 0:
            searched, resolution = self._resolve_belief_ids(retrieved)
        else:
            searched, resolution = [], {}

        # Optional client-side return cap: hand back only the top N distinct beliefs,
        # the way gbrain's `adaptiveReturn` gate does (its headline 0.582 row is N=1).
        # Applied to the provider's raw ranked list, before pinned beliefs are split
        # out, because that is what "return only the top result" actually means.
        return_cap = os.environ.get("AMB_PMB_RETURN_CAP")
        if return_cap:
            searched = searched[: int(return_cap)]
            meta["return_cap"] = int(return_cap)
        raw_results = [b for b in searched if b not in pinned_ids_all][:cap]
        expansions = self._expand_relation_participants(
            user_id, raw_results, scope, exclude=pinned_ids_all | set(raw_results)
        ) if raw_results else []

        all_relevant = raw_results + expansions
        capped_pinned = pinned[:cap]
        capped_relevant = all_relevant[: max(0, cap - len(capped_pinned))]

        pinned_ids = {b["_id"] for b in capped_pinned}
        relevant_ids = list(dict.fromkeys(capped_relevant))
        relevant_set = set(relevant_ids)
        question_ids = {q["_id"] for q in questions[: budget["maxQuestions"]]}
        union_ids = pinned_ids | relevant_set
        prelude = _EVAL_PERSONA_UNIVERSAL if user_id == DEFAULT_USER_ID else ""

        failures: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                failures.append(msg)

        rb = expect.get("relevantBeliefs") or {}
        for bid in rb.get("mustInclude") or []:
            check(bid in union_ids, f"missing expected belief: {bid}")
        for bid in rb.get("mustExclude") or []:
            check(bid not in union_ids, f"forbidden belief surfaced: {bid}")
        for bid in rb.get("shouldInclude") or []:
            check(bid in union_ids, f"expected belief missing (shouldInclude): {bid}")

        only_expected = rb.get("shouldOnlyInclude")
        if only_expected is not None:
            expected_set = set(only_expected)
            for bid in relevant_ids:
                check(bid in expected_set, f"unexpected belief in relevantBeliefs: {bid}")
            for bid in expected_set:
                check(bid in relevant_set, f"missing expected belief: {bid}")

        if rb.get("maxCount") is not None:
            check(len(relevant_set) <= rb["maxCount"],
                  f"relevantBeliefs count {len(relevant_set)} > maxCount {rb['maxCount']}")
        if rb.get("minCount") is not None:
            check(len(relevant_set) >= rb["minCount"],
                  f"relevantBeliefs count {len(relevant_set)} < minCount {rb['minCount']}")

        for a, b in rb.get("orderedBefore") or []:
            idx_a = relevant_ids.index(a) if a in relevant_ids else -1
            idx_b = relevant_ids.index(b) if b in relevant_ids else -1
            check(idx_a != -1, f"orderedBefore: {a} not in relevantBeliefs")
            check(idx_b != -1, f"orderedBefore: {b} not in relevantBeliefs")
            if idx_a != -1 and idx_b != -1:
                check(idx_a < idx_b, f"ranking: {a} (idx {idx_a}) should precede {b} (idx {idx_b})")

        pf = expect.get("pinnedFacts") or {}
        for bid in pf.get("mustInclude") or []:
            check(bid in pinned_ids, f"missing pinned belief: {bid}")
        for bid in pf.get("mustExclude") or []:
            check(bid not in pinned_ids, f"forbidden belief in pinnedFacts: {bid}")

        oq = expect.get("openQuestions") or {}
        for bid in oq.get("mustInclude") or []:
            check(bid in question_ids, f"missing expected question: {bid}")
        for bid in oq.get("mustExclude") or []:
            check(bid not in question_ids, f"forbidden question surfaced: {bid}")

        pp = expect.get("personaPrelude") or {}
        if pp.get("nonEmpty"):
            check(len(prelude) > 0, "personaPrelude empty")
        if pp.get("isNull"):
            check(prelude == "", "personaPrelude not empty")
        for s in pp.get("contains") or []:
            check(s in prelude, f'personaPrelude missing "{s}"')
        for s in pp.get("mustNotContain") or []:
            check(s not in prelude, f'personaPrelude contains "{s}"')

        # Precision / recall over the relevantBeliefs tier only (upstream math).
        pinned_in_seed = {b["_id"] for b in self._beliefs
                          if b.get("pinned") is True and b["user_id"] == DEFAULT_USER_ID}
        if only_expected is not None:
            expected_relevant = set(only_expected)
        else:
            expected_relevant = {bid for bid in (rb.get("mustInclude") or []) if bid not in pinned_in_seed}

        hits = len(expected_relevant & relevant_set)
        if len(relevant_set) == 0 and not expected_relevant:
            precision = None
        elif len(relevant_set) == 0:
            precision = 0.0
        else:
            precision = hits / len(relevant_set)
        recall = hits / len(expected_relevant) if expected_relevant else None

        expected_pinned = set(pf.get("mustInclude") or [])
        pinned_coverage = (len(expected_pinned & pinned_ids) / len(expected_pinned)
                           if expected_pinned else None)

        passed = not failures
        pass_type = self._pass_type(expect, expected_relevant)

        meta.update({
            "pass_type": pass_type,
            "active_pass": bool(passed and pass_type == "active"),
            "relevant_beliefs": relevant_ids,
            "pinned_beliefs": sorted(pinned_ids),
            "retrieved_questions": sorted(question_ids),
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "pinned_coverage": pinned_coverage,
            "failures": failures,
            "resolution": resolution,
            "retrieved_count": len(retrieved),
        })
        reason = "pass" if passed else "; ".join(failures[:4])
        return passed, reason

    # ── stats ─────────────────────────────────────────────────────────────────

    def summary_metrics(self, results: list) -> dict:
        """Precision and recall over the relevantBeliefs tier, plus the active-pass count.

        Accuracy alone is misleading here: 34 of the 77 cases expect an empty result, so a
        provider that returns nothing scores 44%. `active_passes` counts only the cases that
        cannot be satisfied without retrieving the right belief."""
        def m(r, key):
            meta = r.meta if hasattr(r, "meta") else r.get("meta", {})
            return meta.get(key)

        precisions = [v for r in results if (v := m(r, "retrieval_precision")) is not None]
        recalls = [v for r in results if (v := m(r, "retrieval_recall")) is not None]
        active = [r for r in results if m(r, "pass_type") == "active"]
        passed = [r for r in active if (r.correct if hasattr(r, "correct") else r.get("correct"))]
        return {
            "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "active_passes": len(passed),
            "active_total": len(active),
        }

    def summarize_run(self, results: list, console: Console) -> None:
        """Report in upstream's shape: active/structural/empty passes, precision, recall.

        `Active passes` is the comparable number — it counts only the cases that
        cannot be satisfied without query-dependent belief IDs coming back from
        the provider. Total passes includes structural and trivially-empty cases.
        """
        def m(r, key):
            meta = r.meta if hasattr(r, "meta") else r.get("meta", {})
            return meta.get(key)

        by_type = Counter(m(r, "pass_type") for r in results)
        passed = [r for r in results if (r.correct if hasattr(r, "correct") else r.get("correct"))]
        passed_by_type = Counter(m(r, "pass_type") for r in passed)
        prec = [v for r in results if (v := m(r, "retrieval_precision")) is not None]
        rec = [v for r in results if (v := m(r, "retrieval_recall")) is not None]
        resolution: Counter = Counter()
        for r in results:
            resolution.update(m(r, "resolution") or {})

        t = Table(title="PrecisionMemBench")
        t.add_column("Metric", style="bold"); t.add_column("Value", justify="right")
        t.add_row("Active passes", f"{passed_by_type['active']}/{by_type['active']}")
        t.add_row("Structural passes", f"{passed_by_type['structural']}/{by_type['structural']}")
        t.add_row("Trivially-empty passes", f"{passed_by_type['trivially-empty']}/{by_type['trivially-empty']}")
        t.add_row("Total passes", f"{len(passed)}/{len(results)}")
        t.add_row("Mean precision", f"{sum(prec) / len(prec):.2f}" if prec else "—")
        t.add_row("Mean recall", f"{sum(rec) / len(rec):.2f}" if rec else "—")
        if resolution:
            t.add_row("ID resolution", ", ".join(f"{k}={v}" for k, v in resolution.most_common()))
        console.print(t)

    def dataset_stats(self, console: Console, sample_size: int = 200) -> None:
        beliefs, cases = self._beliefs, self._cases
        t = Table(title="PrecisionMemBench — seed corpus")
        t.add_column("Metric"); t.add_column("Value", justify="right")
        t.add_row("Beliefs", str(len(beliefs)))
        t.add_row("Users", ", ".join(sorted({b["user_id"] for b in beliefs})))
        t.add_row("Pinned beliefs", str(sum(1 for b in beliefs if b.get("pinned"))))
        t.add_row("Open questions", str(sum(1 for b in beliefs if b.get("type") == "open_question")))
        t.add_row("Cases", str(len(cases)))
        console.print(t)

        ct = Table(title="Cases by category")
        ct.add_column("Category"); ct.add_column("Cases", justify="right")
        for cat, n in Counter(c["category"] for c in cases).most_common():
            ct.add_row(cat, str(n))
        console.print(ct)
