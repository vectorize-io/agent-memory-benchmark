# sdebench-boltons — datasheet

A benchmark for **whether a coding agent, working on a real codebase, benefits from a memory system**
that has ingested the project's git history and past developer conversations.

Each task is a bug-fix whose correct solution hinges on a **non-guessable, project-specific decision**.
The obvious fix passes the visible repro but fails a held-out test. Where that decision *lives* is the
independent variable — and what each agent can *reach* is the whole point.

## What each arm can access (this is the experiment)

| | plain agent (`full`) | agent + memory (`memsys`) |
|---|---|---|
| the repo + its full git history | ✅ | ✅ |
| a memory store of ingested git rationales + past chats, retrieved & surfaced | ❌ | ✅ |

So the comparison is **"no memory system" vs "memory system"**, holding the agent fixed. What that
means per task depends on **where the decision lives**:

## Axis 1 — source (where the decision lives)

- **H (git history)** — the decision is a real commit's rationale in the repo's history. The plain
  agent *can* reach it (`git log`/`blame`) but often doesn't think to; memory surfaces it. **The fair
  test: both arms can reach it; memory is about reliability, not access.**
- **F (conversation)** — the decision was made in a past developer chat, never written into the repo.
  The plain agent *cannot* reach it at all; only a memory system that captured the chat can. **The
  "memory is necessary" test.**
- *(K = conventions doc: dropped — code decisions don't live in a CONVENTIONS.md, and most repos have
  none. X = cross-feature sibling pattern: not yet built.)*

## Axis 2 — tier (how the task is hosted)

- **real-function** — the policy is planted on an *untested edge* of a **real boltons function**,
  graded against boltons' **real test suite** (a fix must keep the library green). Authentic, harder.
- **planted** — a validated trap as a **new small module inside the real boltons repo** (so it still
  gets the real git-history noise + real-repo navigation), graded against its own tests. Controlled.

## Host & noise

[boltons](https://github.com/mahmoud/boltons) (BSD, ~1600 commits, pinned at `979fa9b`) is the host,
used as a fixture (cloned, not vendored). Its **~1500 real commit subjects + rationale bodies** seed
the memory store as retrieval noise, plus decoy chats — so surfacing the right decision is a real
ranking problem, not a lookup.

## Tasks (10)

| task | source | tier | function | non-guessable policy |
|---|---|---|---|---|
| omdset | **H** | real-function | `dictutils.OrderedMultiDict.__setitem__` | find-the-bug: a `perf` commit rewrote `__setitem__` to reuse `add()` (appends to a stale value list) — must reset to `[v]`; symptom is a stale `getlist` two modules away, all 153 real tests pass; git blame/log is the shortcut |
| slugify | F | real-function | `strutils.slugify` | symbol map `&→and, $→usd, %→pct` (not dollar/percent) |
| pluralize | F | real-function | `strutils.pluralize` | `person→persons, index→indexes, matrix→matrixes` (not people/indices/matrices) |
| under2camel | F | real-function | `strutils.under2camel` | acronym set `{HTTP,API,SKU,GDPR}` — incl. domain SKU/GDPR, **excludes** common db/url |
| findhashtags | F | real-function | `strutils.find_hashtags` | drop all-numeric tags **except 4-digit years** (`#2024` stays) |
| rounding | F | planted | `round_cents` | round half-cents **DOWN** (not banker's/half-up) |
| listmerge | F | planted | `apply_updates` | **union** list values, deduped, base order (not replace/append) |
| budget | F | planted | `MAX_ATTEMPTS` | exactly **7** (measured, not round) |
| discount | F | planted | `apply_discounts` | **percent before fixed** stacking |
| parseflag | F | planted | `parse_flag` | truthy set exactly `{"true","on"}` (not 1/yes) |

Per-task metadata is in each `task.json` (`source`, `tier`, `module`, `function`, `policy`,
`non_guessable`); summary in `datasets/MANIFEST.json`.

## Grading & metric

Grading (Docker, pristine test copies): `pass_to_pass` (existing behaviour) + the repro
(`FAIL_TO_PASS`, red at HEAD) + a **held-out hidden test** (`HIDDEN_TO_PASS`). Every task is verified:
`HEAD` fails repro+hidden, the correct fix passes all, the *naive* fix passes repro but **fails hidden**
(non-guessable). Empirically confirmed too: the plain agent needs interventions on every task.

Primary metric: **interventions** — on a failing grade the harness feeds back the failing-test output
and resumes (cap 5). 0 = solved first try, no human help. Also: turns, cost, solve rate.

## Memory system under test (`memsys`)

Local, file-based (`sdebench/memsys/`): **ingest** git rationale commits + past chats (each verbose
chat LLM-summarized into a decision note — "what was tried, rejected, decided") → **retrieve** by
TF-IDF + a code-symbol boost → **surface** the top entries in the prompt. Not answer-injection: it
surfaces the *decision* (verified no hidden-test values leak), and the agent still writes+tests the fix.

## Results

**n=5, 84 runs, all solved, 0 legitimacy problems:** across the F suite, memory takes the plain
agent's **56 interventions → 1**, cutting turns **46%**, on the real codebase with ~1500 real commits
as noise. Every task discriminates (plain-agent 0.8–2.0 interventions/run). By category:
`real-function 1.53 → 0.05` interventions/run, `planted 1.17 → 0.00`. Full table: `BOLTONS_SUITE.md`.
(The H task's H-vs-F contrast — where the plain agent *can* reach the decision — is being measured.)

## Reproduce

```
git clone https://github.com/mahmoud/boltons ~/dev/_sdebench_hosts/boltons   # host fixture, ref 979fa9b
uv run python sdebench/memsys/seed.py                                        # seed memory
uv run python sdebench/harness/run.py --task sdebench/datasets/boltons-<name>/tasks/main/task.json \
    --history {full|memsys} --run-id <id>
uv run python sdebench/harness/aggregate.py                                  # results table
uv run python sdebench/validate_dataset.py                                   # structural integrity
```

boltons is © Mahmoud Hashemi, BSD — used unmodified as a fixture. Traps, planted modules, tests,
chats, harness, and memory system are this project's.
