# sdebench — Software-Development Engineer Benchmark

A benchmark for coding agents where **git history is load-bearing**. Each task is a
**regression fix** on a synthetic repo whose history we engineer: a bug is *bundled
inside an otherwise-legitimate commit*, so finding and fixing it rewards reading the
history (`git log`/`blame`/`bisect`) and the commit messages (which encode intent).

This is the opposite of SWE-Bench-CL, where tasks don't recur and history is incidental.
Here history *should* help — and the harness measures whether the agent exploits it.

## Why it's designed this way
- **Bundled regression** — the breaking commit also makes a wanted change (with its own
  test), so a lazy `git revert` fails `PASS_TO_PASS`. Forces a *surgical* fix.
- **Intent lives in history** — the guarantee broken by the regression was established in
  an earlier commit whose message states it; the breaking commit's message claims only a
  perf tweak. Diagnosing it cleanly needs the history, not just the code.
- **Deterministic grading** — injected-clock tests (no wall-clock flakiness).

## Grading (a task is solved iff)
1. `FAIL_TO_PASS` — the regression repro (shipped with the bug report) now passes.
2. `PASS_TO_PASS` — the pre-existing suite still passes (no new breakage; no lazy revert).
3. `HIDDEN_TO_PASS` — held-out tests for the same behaviour with different inputs
   (defeats overfitting to the visible repro). Graded from a pristine copy so test edits
   are ignored. Resolution is binary.

## A/B: does history help?
The same task is run with `full` history vs a `squashed` single-commit repo (identical
file tree, no commit trail). The only variable is history availability.

## Metrics
`resolution` (binary), `cost` (input+output tokens × model price), `speed` (wall-clock;
tool-turns secondary). Agent: opencode + gemini-3.5-flash, in a prebuilt Docker image.
Primary comparison metric: **interventions** — on a failing grade the harness feeds the failing test
back and resumes (cap 5); 0 = solved first try.

## Running

The dataset lives in the [sde-bench](https://github.com/vectorize-io/sde-bench) submodule at
`sdebench/datasets` (10 boltons-hosted tasks; see its `DATASET.md` / `GENERATING.md`). There are two
front doors:

**Via the OMB runner** (integrated: results land in the OMB `outputs/` + viewer, alongside the other
benchmarks). `task_type="coding"` — the runner grades by tests, not a judge. **AMB does zero memory
work** — memory is entirely the plugin's domain:
```bash
uv run omb run --dataset sdebench --split boltons --mode coding --memory none              # vanilla baseline
SDE_HINDSIGHT_URL=http://localhost:8899 \
  uv run omb run --dataset sdebench --split boltons --mode coding --memory hscoding          # agent + plugin memory
uv run omb run --dataset sdebench --split boltons --mode coding --memory none -q 1          # one task
```
`--memory none` = vanilla. `--memory hscoding` = the mode (a) builds the task repo, (b) **triggers the
plugin's own backfill** (`hindsight-coding-backfill`) over that repo + the task's conversations — the
**plugin** decides what/how to ingest (extraction, strategies, git scope, pages) — then (c) runs
opencode + the plugin, which does reflect+inject. AMB never calls Hindsight retain *or* reflect. Env:
`SDE_HINDSIGHT_URL` (server), `SDE_HSCODING_PLUGIN_DIR` (the plugin dir with `dist/backfill.js`),
`SDE_HSCODING_GIT_LIMIT` (optional git scope; unset ⇒ the plugin decides).

**Standalone harness** (direct, more arms/flags):
```bash
uv run python sdebench/harness/run.py --task sdebench/datasets/boltons-<name>/tasks/main/task.json \
    --history {full|hscoding|oracle} --run-id <id>
```

## Layout
```
sdebench/
  datasets/            # -> sde-bench submodule: the 10 boltons tasks + generator (gen/) + datasheet
  harness/run.py       # the coding engine: build repo -> agent -> interventions -> pytest grade
                       #   (the OMB `coding` mode shells out to this; run.py is load-bearing)
  Dockerfile           # prebuilt grading env (python + pytest + git)
  FINDINGS.md          # results write-up
```

## Tasks & design
The tasks now live in the [sde-bench](https://github.com/vectorize-io/sde-bench) submodule — 10
bug-fix tasks hosted in the real boltons library, each hinging on a **non-guessable, project-specific
decision** (the obvious fix passes the visible repro but fails a held-out hidden test). Axes: **source**
(H git history / F past conversation), **tier** (real-function / planted), **category** (the kind of
decision). See the submodule's `DATASET.md` (datasheet) and `GENERATING.md` (how tasks are built and
how to add one).

Design rule: the decision must be **non-guessable** — a conventional value/rule the agent guesses
without memory won't discriminate the with-memory vs without-memory arms.
