# sdebench-boltons — a memory benchmark on a real codebase

**What it measures.** Whether a coding agent, working on a *real* codebase, benefits from a memory
system that has ingested the project's git history and past developer conversations. Each task is a
fix whose *correct* solution depends on a **non-guessable, project-specific decision** that lives
only in memory (a past chat) — not in the code, and not inferable from the bug report. A plain agent
must guess (and burns human interventions); an agent with memory should solve it directly.

## Host codebase

[boltons](https://github.com/mahmoud/boltons) — a widely-used pure-Python utility library (1600+
commits, 2013–present, BSD-licensed). It provides:

- a **real git history** (~1500 commits) ingested as **retrieval noise** — the memory system must
  surface the relevant decision out of a realistic, large store;
- a **real test suite** used as `pass_to_pass` for the real-function tasks (a fix must not break it);
- real functions with **untested edges** where a non-guessable policy can be planted.

The host is pinned at ref `979fa9b` and used as a fixture (not vendored — see Reproducibility).

## Task design

Every task is a **regression-fix**: the repo ships a failing repro (`FAIL_TO_PASS`), the agent fixes
the source, and grading runs `pass_to_pass` (existing behaviour) + the repro + a **held-out hidden
test** (`HIDDEN_TO_PASS`). The hidden test is the linchpin: the *obvious* fix passes the repro but
**fails hidden**, so the agent genuinely needs the non-guessable decision.

- **Source = F (conversation).** The load-bearing decision lives only in a past developer chat. The
  plain agent can't reach it; the memory system ingested and summarized it.
- **Non-guessable.** The repro is policy-ambiguous; a naive guess (the natural/default choice) passes
  it but fails hidden. Verified per task: `HEAD` fails repro+hidden, the correct fix passes all, the
  naive fix passes repro but fails hidden.

### Two tiers

- **real-function** (4 tasks) — the policy is planted on an *untested edge* of a **real boltons
  function**, graded against boltons' **real `test_strutils.py`**. Highest realism.
- **planted** (5 tasks) — a validated trap planted as a small module *inside* the real boltons repo
  (so it still gets the real git-history noise and a real-repo navigation), graded against its own tests.

## Tasks

| task | tier | function | policy (non-guessable) |
|---|---|---|---|
| boltons-slugify | real-function | `strutils.slugify` | symbol map `&→and, $→usd, %→pct` (not dollar/percent) |
| boltons-pluralize | real-function | `strutils.pluralize` | formal/DB plurals `persons/indexes/matrixes` (not people/indices/matrices) |
| boltons-under2camel | real-function | `strutils.under2camel` | restore acronyms `{HTTP,HTTPS,URL,API,IO,ID}` uppercase |
| boltons-findhashtags | real-function | `strutils.find_hashtags` | drop **all-numeric** tags (`#42` out, `#2nd` stays) |
| boltons-rounding | planted | `round_cents` | round half-cents **DOWN** (not banker's/half-up) |
| boltons-listmerge | planted | `apply_updates` | **union** list values, deduped, base order (not replace/append) |
| boltons-budget | planted | `MAX_ATTEMPTS` | exactly **7** (measured, not a round number) |
| boltons-discount | planted | `apply_discounts` | **percent before fixed**-amount stacking |
| boltons-parseflag | planted | `parse_flag` | truthy set is exactly `{"true","on"}` (not `1`/`yes`) |

Full metadata per task is in each `task.json` (`tier`, `module`, `function`, `policy`,
`non_guessable`) and summarized in `datasets/MANIFEST.json`.

## Metric

Primary: **interventions** — on a failing grade the harness feeds back the failing-test output and
resumes the agent (cap 5). Lower is better; 0 = solved first try with no human help. Also reported:
**turns**, **wall-clock**, **cost**, and **solve rate**.

## The memory system under test (`memsys`)

A purely **local, file-based** memory (`sdebench/memsys/`):
- **ingest** — git commit rationales + past chats into one JSONL store; each verbose chat session is
  **LLM-summarized into a feedback decision note** ("what was tried, what was rejected, the rule
  settled on") rather than stored as scattered turns;
- **retrieve** — TF-IDF + a code-symbol boost against the bug report;
- **surface** — the top entries are pushed into the prompt ("project memory").
The store is seeded with the host's **real commit subjects as noise** (~1500) plus decoy chats, so
retrieval is a real ranking problem.

## Reproducibility

1. Clone the host fixture: `git clone https://github.com/mahmoud/boltons ~/dev/_sdebench_hosts/boltons`
   (the build scripts copy from there at ref `979fa9b`).
2. Seed memory: `uv run python sdebench/memsys/seed.py`.
3. Run a task: `uv run python sdebench/harness/run.py --task sdebench/datasets/boltons-<name>/tasks/main/task.json --history {full|memsys} --run-id <id>`.

Grading runs in Docker (`sdebench-base`) on pristine test copies, so agent edits to tests are ignored.

## Licensing

boltons is BSD-licensed (© Mahmoud Hashemi). It is used unmodified as a host fixture (cloned, not
redistributed here). The planted modules, traps, tests, chats, and harness are this project's.

## Results

See `BOLTONS_SUITE.md` for the current vanilla-vs-memsys results (n=5).
