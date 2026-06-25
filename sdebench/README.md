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

## Layout
```
sdebench/
  datasets/<repo>/build.py            # builds the repo with engineered git history
  datasets/<repo>/regression_test.py  # FAIL_TO_PASS (shipped to the agent, red at HEAD)
  datasets/<repo>/hidden_test.py      # HIDDEN_TO_PASS (held out)
  datasets/<repo>/task.json           # task definition + test sets
  harness/                            # runner (full vs squashed), grading, metrics
  Dockerfile                          # prebuilt env (python + pytest + git)
```

## Tasks
- `ratelimiter-regression-001` — token-bucket limiter; a `perf:` commit floors partial
  token refill (`int(elapsed*rate)`) while adding `available()`. Fix = drop the floor.
