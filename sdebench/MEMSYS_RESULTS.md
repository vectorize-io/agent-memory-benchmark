# memsys vs vanilla — a local file-based memory beats the baseline

**TL;DR.** A purely local, file-based memory system (`sdebench/memsys/`) that ingests a project's
knowledge from wherever it lives — git rationale commits (H) and past user conversations (F) — and
retrieves the relevant decision for the task, **eliminates every intervention in the baseline
(12 → 0) while cutting cost ~31% and turns ~31%.** The win is real retrieval (verified against 39
distractors), not answer-injection (verified: hidden test values never reach the prompt).

## Setup

- Agent: opencode + gemini-3.5-flash. Metric: **interventions** (the harness feeds back the failing
  hidden test and resumes, cap 5) — lower is better. Plus cost ($/run) and turns.
- Dataset: 8 auto-validated generated tasks = 4 traps × 2 **realistic** sources (`sdebench/gen/`).
  Each trap is a non-default policy; the repro is policy-ambiguous so a naive guess passes it but
  **fails the hidden test**; the answer lives only in its source.
  - **H** = git history (a rationale commit, later dropped by a regression).
  - **F** = a past user conversation (never written to the repo).
  - (A `CONVENTIONS.md`-based "K" source was removed — code decisions don't live in convention docs,
    and most repos don't have one. The two realistic out-of-code homes for a decision are H and F.
    Cross-feature sibling code "X" is future work.)
- `memsys`: ingest H + F across **all** projects into one shared local JSONL store (cross-project
  accumulated memory), seeded with **39 distractor decisions** from unrelated domains → TF-IDF +
  **symbol-aware** recall → surface top-2 to the agent.

## Result

| arm      | runs | interventions | avg turns | avg $/run |
|----------|-----:|--------------:|----------:|----------:|
| baseline (vanilla full git) | 16 (n=2) | **12** | 24.2 | 0.446 |
| memsys   | 40 (n=5) | **0** | 16.6 | 0.308 |

**memsys vs baseline: interventions 12 → 0 · turns −31% · cost −31%.** 40/40 solved; no task ever
non-zero across n=5.

Per-task interventions (baseline total → memsys total over n=5):

```
              H      F            why baseline pays
rounding     0  →0   2 →0        F: conversation unreachable
budget       0  →0   2 →0        F: unreachable
listmerge    2  →0   2 →0        H: agent didn't think to consult git; F: unreachable
slugify      2  →0   2 →0        H: didn't consult; F: unreachable
```

The baseline's 12 interventions split into exactly the failure modes the design targets:
- **F = 8** — structurally unreachable. Vanilla operates on a single repo whose git/code never
  contains the fact (it was only ever said in conversation), so every F needs the human.
- **listmerge/slugify H = 4** — the decision is right there in git, but the agent **doesn't think to
  look** for a behavioral/policy decision.
- **rounding/budget H = 0** — git-natural / value questions; the agent consults on its own.

memsys closes both: it ingests F's conversation (now in memory) and pushes the H decision the agent
would otherwise skip.

## Why it's legitimate (verified)

- **Dataset integrity:** `gen/validate.py` ALL VALID for all 8 (existing green@HEAD, repro+hidden
  red@HEAD, correct fix→green, every naive guess passes repro but fails hidden, answer only in its source).
- **No answer-leak:** memsys surfaces the *decision/rationale* (e.g. "round half-cents down to match
  the ledger"), never the hidden test's literal values. Checked across all 40 runs: hidden values
  (`2.135`, `tom-and-jerry`) appear **only in the agent's own reasoning/testing, never in the injected
  prompt**. The agent still translates the decision into code.
- **Real retrieval, not a toy store:** with 39 distractor decisions from other domains, the correct
  decision still ranks **#1 for all 8 tasks**. An earlier stress test exposed a miss (a `truncate
  slugs` distractor beat the ampersand decision on the generic word "slug"); fixed by linking each
  decision to the **code symbols** it concerns (`def`/`class`/`CONST`), so a bug report that names a
  symbol boosts entries about it.
- **Solves are clean:** all 40 memsys patches touch only the source module (never tests); final pytest
  green; 0 real legitimacy problems.

## What this shows

The bottleneck for "does memory help" is **not** whether the knowledge exists — it's (a) whether the
agent can *reach* it (F: no — it was never written to the repo) and (b) whether it *thinks to look*
(behavioral H: often no). A memory system earns its value by doing both: ingest from wherever
knowledge lives (including conversations git can't hold), and surface it proactively. Pull does not
work here — an earlier `recall_conversations` *skill* was invoked **0/3 times**; the system must
decide relevance and push. Surfacing *knowledge* (a past decision) is legitimate memory; it is
distinct from steering *behavior* ("stop exploring"), which we do not do.

## Honest caveats

- One model (gemini-3.5-flash); synthetic (generated) tasks; n=5 (the 0-result is saturated but the
  suite is small). Real-repo validation is future work.
- The store is **cross-project accumulated memory** — a trap's fact can be reached via any project
  that recorded it (so the F result shows "memory has the fact from conversation," not strictly
  "the conversation alone was recalled"). Per-source attribution would need isolated stores; the
  point here is the *system-level* win: memory makes reachable+reliable what single-repo vanilla misses.
- On the 4/8 cells vanilla already aces (rounding/budget H), memsys is 0 → 0 — neutral, as it should be.
- Retrieval is TF-IDF + a symbol boost; a much larger/real store may need stronger retrieval.
