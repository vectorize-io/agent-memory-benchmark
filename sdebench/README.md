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

## Codebases & tasks
**Small / easy** (one short module, one regression):
- `ttlcache-regression-001` — a refactor changed `DEFAULT_TTL` 287→600 and dropped the
  rationale comment; `287` (a non-round "measured" value) lives only in git history.
- `ledger-regression-001` — a refactor changed `round_cents` to half-up; the real rule is
  round-half-DOWN ("match legacy billing") — non-guessable (agents default to banker's/half-up).

**`billing`** (4 modules, ~18-commit noisy history — the "medium" reference codebase):
- `billing-rounding-001` — same half-down rounding rule, now buried in noise.
- `billing-taxbase-001` — tax charged on the discounted subtotal (2019 policy); navigate noise.

**`minicalc`** (9 modules, ~22-commit noisy history — the "hard / bigger" codebase, a
spreadsheet formula engine: tokens/nodes/parser/refs/sheet/functions/evaluator/errors/engine):
- `minicalc-erragg-001` — **bug far from its symptom**: a "centralize argument evaluation"
  refactor made the *evaluator* short-circuit on any error argument, so COUNT/AVG/MIN/MAX over
  a range containing a `#DIV/0!` return the error instead of aggregating the numbers (SUM is
  unaffected → slips past existing tests). Symptom points at `COUNT`; the bug is in `evaluator.py`.
  Underdetermined (a COUNT-only fix passes the repro, fails the AVG/MIN/MAX hidden tests); the
  "functions decide; the evaluator must not short-circuit calls" policy lives in history + `functions.py`.

Design rule (learned the hard way): the history-encoded fact must be **non-guessable** — a
conventional value/rule (e.g. TTL=300, or banker's rounding for money) the agent guesses
without history won't make the A/B diverge.
