# memsys vs vanilla — a local file-based memory beats the baseline

**TL;DR.** A purely local, file-based memory system (`sdebench/memsys/`) that ingests a project's
knowledge from wherever it lives — git rationale commits, `CONVENTIONS.md`, and past user
conversations — and retrieves the relevant decision for the task, **eliminates every intervention
in the baseline (15 → 0) while cutting cost ~24% and turns ~26%.** The win is real retrieval
(verified against 39 distractors), not answer-injection (verified: no test values reach the prompt).

## Setup

- Agent: opencode + gemini-3.5-flash. Metric: **interventions** (the harness feeds back the failing
  hidden test and resumes, cap 5) — lower is better. Plus cost ($/run) and turns.
- Dataset: 12 auto-validated generated tasks = 4 traps × 3 sources (`sdebench/gen/`). Each trap is a
  non-default policy; the repro is policy-ambiguous so a naive guess passes it but **fails the hidden
  test**; the answer lives only in its source (H = git history, K = `CONVENTIONS.md`, F = conversation).
- `memsys`: ingest all three source kinds → one local JSONL store (seeded with **39 distractor
  decisions** from unrelated domains) → TF-IDF + **symbol-aware** recall → surface top-2 to the agent.

## Result

| arm      | runs | interventions | avg turns | avg $/run |
|----------|-----:|--------------:|----------:|----------:|
| baseline (vanilla full git) | 24 | **15** | 23.3 | 0.428 |
| memsys   | 60 (n=5) | **0** | 17.2 | 0.323 |

**memsys vs baseline: interventions 15 → 0 · turns −26% · cost −24%.** 60/60 solved; no task ever
non-zero across n=5.

Per-task interventions (baseline total → memsys total over n=5):

```
              H      K      F            why baseline pays
rounding     0  →0   0  →0   2 →0        F: conversation unreachable
budget       0  →0   0  →0   2 →0        F: unreachable
listmerge    2  →0   2  →0   2 →0        H/K: agent didn't think to consult; F: unreachable
slugify      2  →0   1  →0   2 →0        H/K: didn't consult; F: unreachable
```

The baseline's 15 interventions split into exactly the failure modes the design targets:
- **F = 8** — structurally unreachable. Vanilla can't see the conversation, so every F needs the human.
- **listmerge/slugify H/K = 7** — the decision is right there in git/the doc, but the agent **doesn't
  think to look** for a behavioral/policy decision.
- **rounding/budget H/K = 0** — git-natural / value questions; the agent consults on its own.

memsys closes all three: it ingests F's conversation (now reachable) and pushes the H/K decision the
agent would have skipped.

## Why it's legitimate (verified)

- **Dataset integrity:** `gen/validate.py` ALL VALID (existing green@HEAD, repro+hidden red@HEAD,
  correct fix→green, every naive guess passes repro but fails hidden, answer only in its source).
- **No answer-leak:** memsys surfaces the *decision/rationale* (e.g. "round half-cents down to match
  the ledger"), never the hidden test's literal values (checked: `2.135`, `tom-and-jerry` never reach
  the prompt). The agent still has to translate the decision into code.
- **Real retrieval, not a toy store:** with 39 distractor decisions from other domains (52 total), the
  correct decision still ranks **#1 for all 4 domains**. A stress test first exposed a miss (a
  `truncate slugs` distractor beat the ampersand decision on the generic word "slug"); fixed by
  linking each decision to the **code symbols** it concerns (`def`/`class`/`CONST`), so a bug report
  that names a symbol boosts entries about it.
- **Solves are clean:** all 60 memsys patches touch only the source module (never tests); final pytest
  green; 0 legitimacy problems.

## What this shows

The bottleneck for "does memory help" is **not** whether the knowledge exists — it's (a) whether the
agent can *reach* it (F: no), and (b) whether it *thinks to look* (behavioral H/K: often no). A memory
system earns its value by doing both for the agent: ingest from wherever knowledge lives, and surface
it proactively. Pull does not work here — an earlier `recall_conversations` *skill* was invoked **0/3
times**; the system must decide relevance and push. Surfacing *knowledge* (a past decision) is
legitimate memory; it is distinct from steering *behavior* ("stop exploring"), which we do not do.

## Honest caveats

- One model (gemini-3.5-flash); synthetic (generated) tasks; n=5 (the 0-result is saturated but the
  suite is small). Real-repo validation is future work.
- The result is "memory makes reachable+reliable what vanilla misses." On the 4/12 cells vanilla
  already aces (rounding/budget H/K), memsys neither helps nor hurts (0 → 0) — as it should.
- Retrieval is TF-IDF + a symbol boost; a much larger/real store may need stronger retrieval, but the
  symbol signal already resolves the obvious same-domain confusion.
