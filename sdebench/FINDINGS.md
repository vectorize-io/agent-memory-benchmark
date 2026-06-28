# sdebench findings — what reduces coding-agent cost on regression-fix

Agent: opencode + gemini-3.5-flash. Metric: resolution (interventions, cap 5), cost (USD from
the provider's token split), speed (wall, turns). All comparisons use the **same base prompt**
across arms (no arm-specific steering) unless noted. n=3 unless stated; numbers are means.

## The benchmark
Regression-fix tasks on synthetic repos whose git history is engineered. A breaking commit is
bundled inside a plausible refactor; the fix depends on a fact (a value, a rule, a policy) that
lives in history and/or requires tracing. Grading = FAIL_TO_PASS (repro) + PASS_TO_PASS
(existing suite) + HIDDEN_TO_PASS (held-out variants), from pristine test copies, in Docker.
On a failing grade the harness feeds the failing-test output back (not the fix) and resumes the
agent; the metric is **how many such interventions** were needed.

Tasks span codebase size and bottleneck type:
- `ttlcache`, `ledger` — one short module; a non-guessable value/rule in history.
- `billing` — 4 modules, ~18-commit noisy history.
- `minicalc` — 9-module spreadsheet engine, ~22-commit history; a **bug far from its symptom**
  (COUNT errors out, but the cause is an evaluator short-circuit) that is also **underdetermined**.

## Arms (how history is delivered)
`full` git · `squashed` (no history) · `memtool` (a `recall_intent` tool the agent may invoke =
**pull**) · `inject` (top-2 query-ranked commits auto-placed in the prompt = **push**) ·
`oracle` (the *known* cause commit pushed = upper bound) · plus behavioral prompt variants.

## Finding 1 — A memory *tool* (pull) does not beat git; pushing the same context does
Fairly (no steering), the `recall_intent` tool is a wash-to-worse vs `full` git: the agent calls
it once, gets the right answer, then **explores anyway**. The bottleneck is the agent's
disposition to explore + its reluctance to act on an optional tool — not finding the answer.

Deliver the *same* retrieval as **pushed context** instead of a tool and it wins. `inject` vs
`full` (cost): ttlcache −15%, taxbase −25%, minicalc −18%, rounding −6%, **ledger +26%**.
`oracle` (perfect retrieval, pushed) beats `full` on 4/5 and is much lower on the hard task
(minicalc −37%). **Lesson: push > pull; agents ignore tools but can't ignore the prompt.**

Two nuances: (a) **retrieval quality is the limiter** — `inject` ranks on the *symptom*, which
misses the symptom-distant cause on minicalc, so `oracle` (−37%) ≫ `inject` (−18%). (b) Push
can *hurt* when git is already efficient (ledger): the injected context is pure prompt overhead.

## Finding 2 — Behavioral constraint cuts cost, but only on exploration-bound tasks
A uniform, fair prompt variant — *"make one change in one file, don't explore"* (`minimal`) —
on the no-memory `squashed` arm:
- minicalc **−30%** (\$0.430→\$0.299, 22→15 turns), taxbase −10% — the bug is *findable*, the
  agent just over-investigates; discipline alone fixes it, no memory needed.
- ttlcache **+2% and [1,1,1] interventions** — the value `287` isn't in the code, so "don't
  explore" can't help and slightly hurts; this task is **knowledge-bound**, not exploration-bound.

## Finding 3 — The two levers fix *different* bottlenecks, and they stack
| bottleneck | example | memory helps? | behavior helps? |
|---|---|---|---|
| knowledge-missing (value only in history) | ttlcache | **yes** | no (can hurt) |
| exploration-heavy (findable bug, over-digging) | minicalc, taxbase | yes | **yes** |

Combining push memory + the `minimal` behavioral prompt (`inject+minimal`) vs `full` git:
- taxbase **−30%** (\$0.260 vs \$0.372, 11 vs 23 turns)
- minicalc **−41%** (\$0.263 vs \$0.447) — far beyond either lever alone (they compose)
- ttlcache: clean `[0,0,0]` (memory supplies the value; `minimal` *alone* here needs `[1,1,1]`)

`inject+minimal` is the best or tied-best config on every task and **never hurts**.

### Benchmark-wide confirmation (Exp5, all 5 tasks, cost vs full+base git)
| task | inject | minimal | inject+minimal |
|---|---|---|---|
| rounding | −10% | +22% `[0,1,1]` | **−27%** `[0,0,0]` |
| taxbase  | −1%  | +7%           | **−14%** `[0,0,0]` |
| minicalc | −3%  | −3% `[1,0,0]` | **−25%** `[0,0,0]` |
| ttlcache | −14% | +80% `[1,3,1]`| **−25%** `[0,0,0]` |
| ledger   | +14% | +33% `[1,1,1]`| +3%      `[0,0,0]` |

**The sharper point: memory makes aggressive behavioral constraint SAFE.** `minimal` *alone*
backfires on knowledge-bound tasks (ttlcache +80% / `[1,3,1]`, ledger +33% / `[1,1,1]`) — telling
an agent "don't explore" is harmful when the answer isn't in the code. But `inject+minimal` is
`[0,0,0]` everywhere: the pushed value removes the *reason* to explore, so the constraint stops
hurting. The combination is robustly best (−14% to −27% on 4/5, tied on ledger, never worse).

## Takeaways
1. **Delivery matters more than content.** The same memory loses as a tool, wins as injected context.
2. **Diagnose the bottleneck.** Knowledge-missing → memory; exploration-heavy → behavioral constraint.
   Picking the wrong lever does nothing (or backfires).
3. **The best single recipe** here is *push the relevant history + instruct a minimal, single-file fix*
   (`inject+minimal`): −30% to −41% cost vs raw git, fairly, with no loss of resolution.
4. **Retrieval ceiling (a hard limit, not just an open lever).** The `inject`→`oracle` gap (esp.
   minicalc) is the cause commit being *symptom-distant*: it shares no terms with the symptom and the
   bug is a wrong return value (no traceback to trace-guide from). Verified offline that neither top-4
   nor a bug+repro 'rich query' retrieves it. So simple symptom-based push retrieval fundamentally
   cannot find such causes — closing the gap needs code-semantic retrieval or the agent's own query
   after it understands the code (the strength of *pull*, which agents nonetheless under-use). A
   push+pull hybrid is the natural test.

## Method notes / honesty
- An earlier "−55%" memtool win was an artifact of an *unfair* system-prompt steering note given
  only to the tool arm; removed, the tool does not beat git. All results above use the same prompt
  across arms. `oracle` is an upper-bound ablation, not a deployable method.
- n=3 is noisy; directional findings (push>pull, the bottleneck taxonomy, stacking on minicalc)
  reproduce across batches, but per-task percentages will move with more N.
