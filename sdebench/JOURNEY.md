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

## 2026-07-04 ~02:10 — clean memory-arm sweep done: contamination WAS material

oc-hs on all 31 tasks, decontaminated plugin, fresh banks: 31/31 solved. On the old 19 tasks the
clean run needed 15 total corrections vs the contaminated runs' mean 10.0 (range 9-11) — i.e. the
removed strings were worth roughly a third of the memory arm's apparent advantage. rounding/-history
(whose answer sat verbatim in the extraction-prompt example) went 0.33/0.00 -> 1/1. This validates
the decision to fix + re-run rather than ship v1's numbers to more customers. Caveat: n=1 vs n=3,
and the server branch/config differs from the overnight runs — the go-forward comparison is
clean-vs-clean, same server, same day.

Hard tier, memory arm: 13 corrections across 12 tasks (dedupe-001 worst at 3) — harder than the
legacy suite even WITH memory, as designed. Vanilla 33-task sweep launched 02:05 (no backfill
needed, should be faster). The 2 amended tasks still need a memory-arm top-up run.

## 2026-07-04 ~03:30 — MAJOR FINDING: the memory arm ran memory-BLIND all night

Vanilla sweep done (33/33 solved). Comparing arms exposed a wrong-shaped result (memory ~uniform
1 correction; WORSE than seeded vanilla on the hard tier 13 vs 11) → investigated → the server log
shows ZERO task-time reflects during either sweep window; every logged reflect is backfill page
generation. The plugin's reflect is "best-effort": on any failure it silently injects nothing — so
the "memory arm" was actually an unseeded vanilla agent for all 31 tasks. The identical harness
path reflects fine on the now-idle box (verified end-to-end incl. a harness-exact container), so
the sweep-time failure was environmental (most plausibly the reflect fetch dying under the
backfill-saturated box) — and INVISIBLE by design.

Fix shipped (product improvement, not benchmark tuning): the plugin now writes a per-session
reflect diagnostic (/tmp/hindsight-plugin.log in-container: reflect_ok/empty/failed + ms + error),
and run.py surfaces it into result.json (memory_diag) and the console with a loud warning when a
memory-arm run had no injected memory. A memory run that silently isn't one can no longer masquerade.

Implication for tonight's data: dz-oc-hs-1 is INVALID as a memory measurement (it is, ironically,
a good unseeded-vanilla replicate). dz-oc-none-1 stands. Re-running hs with REUSE_BANK on settled
banks + diagnostics; every future sweep asserts reflect_ok per task.

## 2026-07-04 ~04:40 — disk-full incident took docker down mid-sweep

hs re-sweep #2 (first with working injection — 50 task reflects confirmed live) started failing
tasks at ~200s each: the DATA VOLUME hit 100% (867/926 GiB) and the docker daemon died. Each task
workdir holds TWO full boltons clones (agent repo + pristine grading copy) and sweeps never cleaned
them — hundreds of dirs ate the disk. Fixes: (1) run.py now deletes repo+grade copies after writing
result/trace (steady-state disk ~2 tasks × concurrency); (2) cleaned /tmp/sdebench (+6 GiB);
(3) OrbStack restarted (`orb start`), hindsight-db auto-recovered, API server (embedded pg0)
unaffected — banks intact. Sweep #2 is invalid (infra); relaunched as #3: instrumented, REUSE_BANK,
disk-lean. The reflect diagnostics did their job on their first outing — the 14 pre-crash results
show injection working (e.g. dedupe 3->1 corrections vs the blind run).

## 2026-07-04 ~05:45 — first VALID clean comparison (n=1) + a real Hindsight gap found

hs sweep #3: 33/33 solved, reflect_ok verified on every task (diagnostics prove memory was
injected). Honest n=1 numbers vs seeded vanilla:
  ALL 33:  corrections 34 -> 24 (-29%) | cost $25.40 -> $20.27 (-20%) | turns 1184 -> 1027 (-13%)
  OLD 19:  corrections 21 -> 13 (-38%) | HARD 12: 11 -> 9 (-18%) | AMENDED 2: 2 -> 2 (0%)
Much smaller than v1's contaminated -58%, and with a clear frontier: memory LOSES on 7
conversation tasks (v=0, m=1-2) where seeded vanilla reads the raw chat but reflect's summary
drops a component of a multi-part policy.

The conversation-amended type caught a real defect on its first outing: reflect on the
dedupe-amended bank returns the STALE chat-A rule (keep-latest — the proven naive!) and misses
the amendment (most-filled + tie->primary) plus the day-truncated key. Cross-conversation
supersession does not happen: both chats' facts coexist and reflect prefers the wrong one.

Planned general fixes (product-level, not benchmark tuning): (1) backfill assigns per-chat
occurred_at (real session exports have timestamps; ordered synthetic dates for JSON chats) so
recency is available; (2) reflect mission: on conflicting facts the latest/superseding decision
wins and the old rule must be reported as superseded. To keep n=3 internally consistent, these
land AFTER the n=3 sweeps; the amended pair then gets a before/after case study.

Fixes shipped to PR #2522 (89b58f376): chronological session recency (was inverted!) +
supersession-aware reflect mission. Revised sweep plan so final memory numbers use banks built
by the FIXED plugin: vanilla #2, #3 (bank-independent, running/queued) -> banks v2 fresh backfill
+ hs runs A/B/C with reuse -> claude-code arms. hs sweep #3's banks (v1) stay archived as the
pre-fix point of comparison; the amended-pair before/after becomes the case study.

## 2026-07-04 ~15:30 — banks v2 run A: fixes hold up

hs v2a (fresh banks, fixed plugin): 33/33 solved, 27 corrections, reflect_ok verified on every
task. Amended pair with v2 banks: dedupe-amended 0 corrections (was 1 with v1 banks — reflect had
surfaced the superseded keep-latest rule), retryjitter-amended 1. The chronological-recency +
supersession-mission fixes moved exactly the tasks they were built from — and nothing else was
touched to get there. Run B (reuse) launched for n=3.

## 2026-07-04 ~17:35 — OPENCODE n=3 FINAL (33 tasks, decontaminated, injection-verified)

vanilla:  34, 33, 29  (mean 32.0)
memory:   27, 26, 22  (mean 25.0)   => corrections -22%
Every memory run: reflect_ok on all 33 tasks. Solve rate 100% everywhere. This is the honest
opencode story on the hardened suite: a fifth fewer human corrections, no contamination, no
silent memory loss, hard multi-part tasks included. (v1 doc claimed -58% on the old suite with
a contaminated plugin — the gap between those numbers is the price of legitimacy, documented
throughout this file.) Claude Code arms launched next (vanilla first).

## 2026-07-05 ~00:15 — CAMPAIGN COMPLETE

Final n=3, 33 tasks, decontaminated plugin, v2 banks, injection verified on every memory run:
  OpenCode+Gemini:  corrections/task 0.97 -> 0.76 (-22%) | cost -18% | turns -10% | solved 99/99 vs 99/99
  Claude+Sonnet:    corrections/task 0.89 -> 0.37 (-58%) | cost -31% | turns -21% | solved 99/99 vs 98/99
  (the 98th: findhashtags-001 hit the 5-cap in one cc memory run — reported, not rerun away)
Amended case study: v1 banks reflect returned the SUPERSEDED rule verbatim (defect reproduced live);
after the chronological-recency + supersession fixes, the stale rule no longer surfaces and the
amended pair averages 1.3 corrections/run vs 2 before. One residual quality observation: a v2
reflect sample fabricated a file path + REF-ID — logged as future work (reflect grounding).
Customer doc rewritten as v2 (supersedes v1 with an explicit "what changed and why numbers are
lower" section: contamination found+fixed, harder suite, injection verification). Charts
regenerated from dz-* outputs (corrections/cost/turns; wall+tokens dropped — not uniformly backed).

## 2026-07-05 ~00:20 — capped-task post-mortem + n-boost

findhashtags-001 (the one capped cc-memory run): reflect on that bank surfaces the FULL policy
including the 4-digit-year carve-out; the trace shows the agent implemented the general
digit-filter and then iterated on partial hidden-test feedback for 5 rounds without revisiting
the injected rule. Application variance, not retrieval failure — the other two runs solved the
same task with the same bank. No fix warranted; the cap stands in the results.
Launching cc runs 4-5 (both arms) to tighten claude's wide memory variance (14/15/8).

## 2026-07-05 ~04:30 — n=5 for Claude; campaign closed

cc-none n=5: 33,31,24,32,30 (0.91/task). cc-hs n=5: 14,15,8,9,17 (0.38/task) = -58%, unchanged
from n=3. Second capped run appeared (csvquote-history in run 5; different task than run 2's) —
solve 163/165 vs vanilla 165/165, disclosed in the doc. Doc + charts updated to n=5 for Claude.
Final deliverables: doc v2 (~/Documents), charts (corrections/cost/turns, n-averaged with error
bars), PRs: hindsight #2522 (decontamination, diagnostics, supersession fixes), benchmark #23
(harness fixes, 20 run outputs, this journal), sde-bench hardening-2026-07 (33-task dataset).

## 2026-07-23 — retired the standalone plugin copy; harness hardening

The harness now uses the PR-tracked monorepo package (hindsight-integrations/
hindsight-coding-agents, PR #2522) instead of the standalone ~/dev/hindsight-coding-opencode
copy; container mount renamed to /opt/hindsight-coding-agents, both images rebuilt. Sanity run
green: reflect_ok, budget solved 0 corrections. Two infra rots found on the way (server down,
sdebench-base image lost) — the reflect diagnostics flagged the first immediately, and grade()
now raises on a grading-container failure instead of burning the intervention cap on empty
"tests still fail" feedback.
