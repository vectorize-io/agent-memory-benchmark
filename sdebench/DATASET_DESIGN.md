# sdebench — dataset generation design

How we generate tasks, and the dimensions every task is tagged on.

## Task shape (all tasks)
A regression-fix on a synthetic repo. The fix depends on a **non-guessable fact** (a value,
rule, or policy the agent cannot infer from the buggy code alone). Grading:
- **FAIL_TO_PASS** — the repro test (shipped to the agent, red at HEAD).
- **PASS_TO_PASS** — the existing suite (green at HEAD; the regression slipped past it).
- **HIDDEN_TO_PASS** — held-out variants (defeat overfitting / underdetermined fixes).
Run from pristine test copies in Docker. On a failing grade the harness feeds back the failing
output (not the fix) and resumes the agent; metric = number of such **interventions** (cap 5),
plus **cost** (token split → USD) and **speed** (turns, wall).

**Non-guessability is mandatory.** If the agent can guess the fact from convention (e.g. TTL=300,
banker's rounding), no source of memory matters and the task doesn't discriminate.

## Three independent axes (tag every task)

### Axis 1 — SOURCE: where the load-bearing fact lives
| | source | lives in | static/dynamic |
|---|---|---|---|
| **H** | History | git commits / diffs / blame | static (in repo) |
| **K** | Conventions / Knowledge | docs, ADRs, `CONVENTIONS.md`, `CLAUDE.md`, policy comments | static (in repo) |
| **F** | Feedback / Preferences | retained user/team corrections & way-of-working | dynamic* |
| **X** | Cross-task experience | the agent's own past solves (patterns) | dynamic* |

\* **Dynamic sources only exist if something happened before** (a prior correction, a prior solve).
For the static single-task harness they are **pre-seeded as memory files** in the repo (e.g.
`.agent/preferences.md` for F, `.agent/past_fixes.md` for X). The *genuine* dynamic version
requires a multi-task **stream** where task N's feedback/solve becomes task N+1's memory — built
later, not in the static set.

### Axis 2 — DELIVERY: how memory reaches the agent
`full` (git + tools, the **vanilla** harness) · `squashed` (none) · `memtool` (pull tool) ·
`inject` (push into prompt) · `oracle` (known fact pushed, upper bound).

### Axis 3 — BOTTLENECK: what the task stresses
- **knowledge-missing** — fact not in the code → memory is decisive (squashed fails).
- **exploration-heavy** — bug findable, agent over-digs → behavioral constraint helps, memory cosmetic.
- **multi-hop** — cause far from / split across the symptom → needs tracing; resolution discriminator.

## Generation rules
1. Each task is **decisive on exactly one SOURCE** (the fact lives in only that source). Tag it.
2. The fact is **non-guessable** (validate: a conventional guess passes the repro but fails HIDDEN).
3. **Isolate by ablation** — the *same* bug can be re-emitted with the fact moved between sources
   (H-only / K-only / F-only / X-only / none), to measure which source the agent actually uses.
4. Existing suite green at HEAD; a conventional/single-file fix passes the repro but fails HIDDEN.
5. Record per task: `source`, `bottleneck`, repo size (modules / commits).

## Status
- Built (mixed H+K, mostly history): ttlcache, ledger, billing-rounding/taxbase, minicalc-erragg,
  minicalc-rangemf (multi-hop). See FINDINGS.md.
- **Next**: one clean task per SOURCE (H, K, F, X) reusing the proven half-down rounding fact, so
  SOURCE becomes a measured category. Then the dynamic F/X **stream**.
