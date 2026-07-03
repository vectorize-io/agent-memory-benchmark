# Hardening journey — 2026-07-03 → 2026-07-05

Mission: decontaminate the memory plugin, harden + extend the task suite (harder tasks, both
sources), improve Hindsight where evals expose weaknesses, re-run legitimately (opencode for
iteration, claude-code once solid), update the customer doc with clean numbers. Never force
memory to win.

## 2026-07-03 — contamination fix (pre-registered before any re-run)

A fairness audit found three benchmark-specific strings in the plugin:

1. `missions.ts` CHAT_CUSTOM_INSTRUCTIONS example was the literal answer to graded task
   `boltons-rounding` ("round_cents uses ROUND_HALF_DOWN … legacy ledger") — replaced with a
   fictional non-benchmark example (API version pinning).
2. `inject.ts` told the model "the hidden tests depend on those exact choices" — benchmark
   grading knowledge the vanilla arm never gets; reworded benchmark-agnostic.
3. REFLECT_MISSION examples shape-matched to slugify/budget ("words a symbol maps to",
   "the exact number") — neutralized to "the actual decided value".

Fixed in BOTH copies: the hindsight monorepo package (`hindsight-integrations/
hindsight-coding-agents`, pushed to PR #2522 as 4edad3b0b) and the standalone
`~/dev/hindsight-coding-opencode` package the harness actually mounts (not a git repo —
src+dist rebuilt in place). Verified by grep: no benchmark strings in src or dist of either.

Everything downstream of here runs on the decontaminated plugin with FRESH banks (old banks
were built with the contaminated extraction prompt — must not be reused).

Prediction to falsify: the honest effect should mostly hold, since the decision chats genuinely
contain the decisions; expected impact is largest on `boltons-rounding` (its answer was in the
extraction prompt) and a possible small drop on oc-hs overall (inject wording no longer names
hidden tests).

## Harder-task design (Task #3 plan)

Hardness levers (all legitimate — they make the DECISION harder to guess/converge on, not the
retrieval artificially easier): (L1) multi-part policies — 3+ interacting constraints so naive
fixes satisfy subsets and each feedback round only surfaces part of the rule; (L2) symptom-distant
vocabulary — bug report shares no keywords with the decision text, so retrieval must reason;
(L3) cross-module consistency — the policy spans two files that must agree; (L4) wide hidden tests
(parametrized, 10+ cases) so assertion leakage per round stays partial under the 2500-char feedback
tail; (L5) history-hard — rationale buried in a commit whose subject looks unrelated; (L6) several
plausible naive guesses, each proven to fail hidden.

Planned traps (each emits conversation + history variants):
1. `dedupe` (collection-merge, L1+L4): merge_records key=(email lowercase, day-truncated date);
   conflict → most-filled-fields wins; tie → list_a ("CRM is source of truth"). Naive: keep-latest.
2. `redact` (filter-rule/set, L1): mask {password, token, api_key, ssn, card_number} by key SUFFIX
   match incl. nested; card keeps last4; email NOT masked (support needs it). Naive: mask email too /
   full-mask card / exact-key match only.
3. `trimstats` (numeric-policy, L2): latency aggregator drops exactly the top 2 samples per fixed
   60-sample window (hypervisor warmup spikes), not a percentile. Naive: p95 clamp / drop top 1.
4. `sched` (ordering, L1): next-job = priority desc, tie → shorter estimated_runtime, and same
   tenant never twice back-to-back. Naive: priority+FIFO.
5. `retryjitter` (set+numeric, L6): retry only 5xx plus {429, 408}; decorrelated jitter capped 30s.
   Naive: retry all 4xx / exclude 408.
6. `csvquote` (invariant, L2): exporter must emit leading-zero-preserving quoted text fields for the
   ERP import; symptom reported as "IDs corrupted in the monthly export".
NEW SOURCE TYPE (memory-shines candidate, fair): `conversation-amended` — the rule is set in chat A
and AMENDED in chat B weeks later; bank ingests both; correct answer = the amended rule. Tests
consolidation across conversations (exactly what real teams do). Vanilla opencode gets both chats
seeded, so access parity holds. 2 tasks planned (variants of dedupe + retryjitter policies).

Validation gate per task: HEAD=(pass,fail,fail), CORRECT=(pass,pass,pass), every NAIVE=(pass,pass,FAIL)
via gen validators, then structural validate.py, then an opencode sanity run (vanilla + memory, n=1).

## 2026-07-03 evening — hard tier landed (12 new tasks, 31 total)

Six new planted traps implemented (by parallel subagents, integrated + validated centrally):
dedupe, trimstats, sched, redact, retryjitter, csvquote — each with conversation AND history
variants. All 24 discrimination checks PASS on the official validators; structural validator
green after regenerating MANIFEST (31 tasks: 15 conversation / 16 history; every category now
has a hard representative). Design notes: agent A caught that the requested trimstats naive
(winsorize-at-p99) was mathematically indistinguishable from plain p95 and substituted a 5%-trim
naive — the kind of impossible-cell detection the validation gate exists for. Emitter regression
found: emit_host.py drops post-emission enrichment fields (function/policy/non_guessable/host) —
restored published task.jsons from git and enriched the new ones; emitter fix deferred.
Dataset pushed to sde-bench branch `hardening-2026-07`.

Still pending for task #3: opencode sanity runs on the 12 new tasks (waiting for the re-baseline
sweep to finish to avoid box contention), and the conversation-amended source type (2 tasks).

## 2026-07-03 ~18:50 — sweep restart, now 31 tasks

The first re-baseline sweep was killed ~50min in (1 task completed; cause unknown — harness-tracked
background job). Relaunched DETACHED (nohup+caffeinate, log /tmp/sweep-dz-oc-hs-1.log). The dataset
dir now holds 31 tasks, so this hs sweep covers old 19 (contamination comparison vs nz-oc-hs-*) AND
the 12 new hard tasks (their first real memory-arm contact). An oc-vanilla 31-task sweep follows.
Noted: the omb runner starts several tasks concurrently — relevant to wall-clock honesty; find the
concurrency knob before the final sweeps and run those serially or measure contention explicitly.

## 2026-07-03 ~20:00 — emitter idempotence + a self-inflicted git incident

Made the three emitters merge-preserve unknown task.json keys (re-emission no longer strips
enrichment) and normalized field order. In the process, a stash/re-emit sequence briefly reverted
the emitter patches and I pushed a broken dataset state (bba6413: two tasks missing `policy`).
Caught by the structural validator on the next run; restored from the stash, verified idempotence
properly (three emitters re-run → zero diffs), repaired in 9cc3d38. Lesson recorded: validate
BEFORE commit in the same shell invocation gates nothing if the chain uses `;` — gate pushes on
validator exit status.

Sweep progress at 20:00: 8/31 tasks done, all correct so far on the memory arm (incl. new
csvquote pair). Backfill of 31 fresh banks is the long pole as predicted.
