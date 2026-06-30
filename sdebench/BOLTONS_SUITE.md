# boltons real-codebase host suite

A scale-up of sdebench onto a **real codebase** (boltons, 1622 commits, 2013–2026) as the host:
real git history is the retrieval noise, a real test suite is `pass_to_pass`, and the agent
navigates a real repo. The question: does the local file-based memory still help when the store is
flooded with real commits and the chats are realistic verbose sessions?

## Result (n=1, vanilla vs memsys)

```
task        vanilla            memsys
slugify     interv=1 turns=33   interv=0 turns=19
rounding    interv=1 turns=35   interv=0 turns=13
listmerge   interv=1 turns=27   interv=0 turns=18
budget      interv=2 turns=42   interv=0 turns=8
discount    interv=1 turns=16   interv=0 turns=10
parseflag   interv=2 turns=32   interv=0 turns=16
TOTAL       8 interv / 185      0 interv / 84 (-55% turns)
```

**All 6 tasks discriminate: vanilla needs interventions (8 total), memsys solves every one at 0** —
and turns drop 55%. Against a store of **1486 entries** (1471 real boltons commit subjects + 15 chats
incl. 8 decoy sessions) the right memory still ranks **top-2** for every task.

(parseflag was initially guessable — vanilla inferred the strict `"true"` policy at 0 — so it was
sharpened to an *arbitrary* truthy set `{"true","on"}` (not `"1"`/`"yes"`), which vanilla can't guess;
it now needs 2 interventions.)

## The tasks

- **slugify** — a *real-function* trap on `boltons.strutils.slugify`, graded against boltons' **real**
  `test_strutils.py`. Non-guessable policy: a project symbol map `& -> and, $ -> usd, % -> pct` (NOT
  the natural words "dollar"/"percent"). The repro only reveals `&`; the hidden enforces `usd`/`pct`.
  The decision lives only in a chat (F).
- **rounding, listmerge, budget, discount, parseflag** — validated generator traps *planted* into the
  real boltons repo as modules (so they get the real history noise + a real-repo navigation), each
  with a realistic chat as the decision source.

## How memory is built (unchanged from the core memsys)

- **git** → commit rationale bodies (thin on real repos — boltons commits are terse one-liners, so
  the *subjects* are ingested as retrieval noise; the real decision is in chat).
- **chats** → each realistic verbose session (6–10k chars) is LLM-summarized into a feedback DECISION
  NOTE ("what was tried, what was rejected, the rule settled on"), then retrieved by TF-IDF + a
  code-symbol boost and surfaced. This is what carries the non-guessable value.

## Honest caveats

- **n=1** — boltons runs are slow (full-repo copy + Docker grade per run), so this is a single pass.
  The 5/6 signal is clear (vanilla >0, memsys 0) but not yet replicated.
- **5/6 are planted modules, 1 is a real function.** The strongest realism is the real-function
  slugify (real test suite as pass_to_pass). Converting more traps to real boltons functions (find
  untested edges) is the next push.
- **The memory system was left untouched** for this scale-up (per instruction) — retrieval is still
  TF-IDF + symbol-boost, which held at ~1500 entries but thinned the margin (right memory dropped to
  #2 behind a real commit on one task); a much larger store would likely want a reranker/embeddings.

## Setup note

The host is a fixture, not vendored: clone boltons to `~/dev/_sdebench_hosts/boltons` (the build
scripts copy from there, frozen at ref `979fa9b`).
