# Overnight run — findings & blockers (2026-07-02 → 07-03)

Living log for the autonomous overnight task. Newest section at the bottom.

## Mandate
1. Finish seeding (vanilla = agent CAN read past dev sessions via `opencode session list`/`export` if it chooses).
2. Switch the containerized agent to the NEW plugin (`hindsight-coding-agents`, monorepo); adapt to its JSON config (env removed). May modify the plugin if needed.
3. Run **n=3 vanilla vs hindsight-plugin**, all 10 tasks. Backfill git-limit=100, **reuse the bank across the 3 runs** (no re-ingest).
4. Surface results in the **local AMB UI**, make them understandable.
5. Report **all metrics**: tokens, turns, total cost, interventions.
6. Stretch: **claude-code + sonnet-5** agent (via `--agent` flag) — vanilla vs hindsight, same/better/worse?
7. Stretch²: expand the **H-source** category (only omdset today) with ~9 more tasks.

## Decisions / defaults taken (user gone)
- Plugin = new `hindsight-coding-agents`; `SDE_HSCODING_PLUGIN_DIR` points there.
- Bank reuse: skip backfill if the per-task bank already has memories.
- Run the 3 sweeps SEQUENTIALLY (avoid concurrent backfill races; clean bank reuse).
- Keep the Hindsight `:8899` server healthy (health-check/restart if it dies).

## Log
### Setup (start)
- Containerized agent already committed (883764c): opencode runs in a per-task container, resumes via `-c`, both arms validated single-task.
- Seeding mechanism validated: synthetic opencode session (exact schema) imports cleanly; full conversation reads back via `opencode export`. `seed_sessions()` added to run.py.

### Seeding VALIDATED (vanilla)
- rounding vanilla (full arm): seeded 1 past session; the agent **ran `opencode session list` + `opencode export <id>` on its own** (consulted history), solved with 1 intervention. wall=158s.
- Mechanism confirmed: seeded opencode sessions are discoverable+readable by the agent via the CLI (it has bash), and the prompt note gives it the "chance". Availability + agency, as designed.

### CLAUDE-CODE STRETCH — two real BLOCKERS (need your input tomorrow)
The harness can be made agent-pluggable (`--agent opencode|claude-code`), but claude-code hits two blockers I did NOT force overnight:

1. **Auth.** No `ANTHROPIC_API_KEY` anywhere (.env/env/shells). Host `claude` (v2.1.198) auths via a **Claude Max OAuth token** (macOS keychain `Claude Code-credentials`). Options:
   - (a) Provide an `ANTHROPIC_API_KEY` → clean, billed, container-friendly. **Preferred.**
   - (b) Extract the keychain OAuth token → `/root/.claude/.credentials.json` in the container. Technically may work, but a 60-solve automated run on a personal Max subscription is rate-limited (5h caps) and against interactive-use norms. I did NOT do this.
   → **Decision needed: give me an API key, or approve subscription use (risky).**

2. **Memory arm.** The Hindsight memory is an **opencode plugin** (`@opencode-ai/plugin` hooks). claude-code has a different extension model (settings.json hooks / MCP / subagents). So claude-code's `hindsight` arm needs a NEW integration — options: a claude-code **UserPromptSubmit hook** that injects the reflect answer, or an **MCP server** exposing `memory_reflect`, or prompt-injection by the harness. The `hindsight-coding-agents` plugin is already "harness-pluggable" (has a harness registry) but only opencode is implemented.

**What I prepped (no run):** claude image build + the agent-adapter design (below). Ready to execute once (1) is answered.

### Agent-flag adapter design (ready to implement)
`run.py` gets `--agent opencode|claude-code` (default opencode). Extract an adapter with 4 hooks; everything else (build repo, grade in sdebench-base, intervention loop, metrics) is agent-agnostic:
- `image` — `sdebench-agent` (opencode) / `sdebench-agent-claude` (claude). Both built.
- `run_turn(cid, msg, resume)` — the per-turn exec + a parser to the common `{tokens,turns,trajectory}` shape.
  - opencode: `opencode run --format json -m <model> [-c] <msg>` → parse the JSON event stream (done).
  - claude: `claude -p --output-format stream-json --verbose -m claude-sonnet-5 [-c] <msg>` → parse claude's stream-json (assistant/tool_use/result events; tokens in the `result`/`message.usage`).
- `config(cid, arm)` — memory wiring. opencode: write `~/.hindsight/coding-agent.json` (done). claude: N/A until the memory integration exists (blocker #2).
- `seed(cid, conversations)` — vanilla past-session seeding. opencode: `opencode import` (done). claude: write JSONL transcripts to `/root/.claude/projects/<cwd-slug>/<uuid>.jsonl` (claude's native session format — well-known), then the agent can `--resume`/read them. (Different store, same idea.)
Model ids: opencode `google/gemini-3.5-flash`; claude `claude-sonnet-5` (Sonnet 5).

### Task expansion (+9 H-source) — plan (stretch², unblocked)
H-source pattern (from omdset build.py): checkout boltons@REF, then commit a sequence — (1) a
"documented-invariant" commit carrying the DECISION/why in its message+comment (the H source), (2) a
noise commit, (3) a "regression" commit with a misleading message that breaks it. HEAD is buggy; real
tests pass (untested edge); repro+hidden discriminate; the fix requires reading git history.
Cheapest route to 9 more: **H-variants of the 9 planted/real-function tasks** — same non-guessable
policy, but the decision lives in a git commit instead of a chat. Needs: a build.py per task (plant
correct+documented commit → regression commit) + reuse the existing repro/hidden tests + `gen`
validate discrimination + add `source:"H"` task.json. Careful validation per task (don't ship
undiscriminating tasks). NOTE: doing all 9 *well* is real work — I'll prefer a few validated + a
documented recipe over 9 rushed ones.

### AMB UI assessment (Phase 4)
- `omb view` works; the API lists sdebench coding runs. **Per-run detail (RunDetail.vue) already renders all agent metrics** as pills: 🔁 interventions, 💲 cost, 🔤 tokens (in/⚡cached/out), 🔧 turns, ✓/✗ resolved, + Source/Tier/Category axes, + the agent trajectory. My runner fix adds `meta.tokens` so tokens now show for the new runs.
- **Caveat (important for reading the UI):** the run-LIST/leaderboard sorts by **accuracy**, which for coding is ~100% for every arm (every task solves eventually). So the leaderboard does NOT differentiate arms — the real signal is **interventions / cost / turns**, visible in each run's **detail** view and summarized in the comparison table below. I did NOT rebuild the Vue frontend overnight (can't visually verify blind); the definitive vanilla-vs-hindsight comparison is the table (all metrics) written here after the sweep.
- To view: `uv run omb view` → open the sdebench dataset → open an `ov-none-*` and an `ov-hs-*` run → compare the pills/aggregate.

### Early signal (vanilla run 1) — seeding is working + changes the baseline
- Vanilla (SEEDED) run 1: 10/10 solved, **9 total interventions**. tokens present 10/10 (fix confirmed).
- Notable: **pluralize=0 and under2camel=0 interventions** in vanilla — the agent (with seeded past sessions available) solved them first try, i.e. it consulted its history. So seeding lowered the vanilla baseline from ~12 (no-seed, earlier) toward ~9.
- IMPLICATION: the comparison is now the intended one — "raw session access (vanilla+seed) vs memory system (hindsight)". A smaller delta is EXPECTED and more honest: memory must beat an agent that CAN read its own history, not one with nothing. Will confirm with n=3 means.

### ⭐ KEY FINDING — seeding strengthens vanilla, shrinking the memory delta (needs your call)
Run-1 numbers (n=3 will confirm): **no-memory ≈12 → vanilla+seed ≈8–9 → hindsight ≈7 interventions.**
- **Reflect quality is EXCELLENT.** Direct reflect on a per-task bank returns the exact decision (e.g. budget → "MAX_ATTEMPTS=7, retryx/retry.py, 12.7s vs 25.5s"). Memory content + backfill (new plugin, limit 100, 553 facts/bank) is not the problem.
- **Bank reuse works** (run 2/3 skip backfill).
- BUT the memory arm only modestly beats the SEEDED vanilla. Per-task V→H (run 1): H wins discount/rounding/slugify, loses under2camel, ties the rest. The agent isn't calling `memory_reflect` (relies on auto-inject).
- **Why the earlier repro showed 12→1 and now it's ~8→7:** that repro had a *no-memory* vanilla (nothing to read). Now vanilla can (and does) read the seeded past sessions, so its baseline dropped to ~8. Memory must beat an agent that already reads its history — a much higher bar. This is the HONEST, realistic comparison you asked for.
- **⚠️ Design question for you:** with seeding, raw session access ≈ memory. Options: (a) accept the small honest delta; (b) let the hindsight arm ALSO seed → measure "memory ON TOP of session access"; (c) verify/strengthen the plugin's auto-inject (agent isn't visibly using injected memory — worth confirming the system-prompt injection actually lands in-container); (d) make tasks harder so raw session-reading isn't enough. I'll gather n=3 and, if time, probe the auto-inject.

### Auto-inject CONFIRMED working (correction to the note above)
Checked two hindsight budget runs: one set `MAX_ATTEMPTS=7` on the INITIAL attempt → 0 interventions (memory injected + applied); the other didn't → 1 intervention. So the plugin's system-prompt injection IS landing — the agent just applies it inconsistently (Gemini nondeterminism; reasoning text is hidden so it's silent). Not a bug. This is WHY n=3 matters and why the per-task delta is noisy. Net: memory arm is valid; its edge over seeded-vanilla is real but modest + variable.

## ============ RESULTS: vanilla vs hindsight-plugin, n=3, all 10 tasks ============
Both arms: opencode + gemini-3.5-flash, in-container agent, grading in sdebench-base. Vanilla = seeded
past sessions (agent may read them). Hindsight = the plugin (backfill git-limit 100 + reflect+inject),
banks reused across the 3 runs. Every cell is the mean over 3 runs.

| metric (sum over 10 tasks, mean of 3 runs) | VANILLA (seed) | HINDSIGHT (plugin) | Δ |
|---|---|---|---|
| **interventions** | **8.3** | **6.7** | **−20%** |
| solved | 10/10 | 10/10 | = |
| cost (USD) | $6.67 | $6.55 | −2% |
| **input tokens** | **12.9M** | **7.76M** | **−40%** |
| output tokens | 106k | 89k | −16% |
| tool turns | 357 | 345 | −3% |
| **wall time** | **1808s** | **1472s** | **−19%** |

### Headline
- **Both solve everything (10/10).** With a *seeded* vanilla (agent can read its past sessions), memory's edge on **interventions is modest (8.3 → 6.7)** — the honest, fair result you wanted (memory beating an agent that already reads its history).
- **The bigger win is EFFICIENCY:** hindsight cuts **input tokens ~40%** and **wall time ~19%** at equal cost + equal solve rate. Memory guides the agent to the fix with far less exploration/context re-reading.

### By decision-type (interventions V → H)
- **numeric-policy 2.0 → 0.67** ✅ (rounding, budget — memory helps a lot)
- **ordering 1.33 → 0.33** ✅ (discount)
- **collection-merge / invariant / filter-rule / mapping**: ~flat
- **set-membership 0.67 → 1.33** ⚠️ (under2camel, parseflag — memory slightly HURT; small n, likely noise or reflect surfacing an adjacent-but-wrong rule)

### Per-run stability
- vanilla interventions: 9, 9, 7 (mean 8.3). hindsight: 7, 5, 8 (mean 6.7). Overlapping ranges → the interventions delta is real but small vs the noise; the **token/wall deltas are the robust signal**.

### Caveats / for discussion
- Seeding strongly raises the vanilla baseline (no-seed vanilla was ~12). If the goal is to showcase memory's value on interventions, either (a) also seed the hindsight arm (measure "memory ON TOP of sessions"), or (b) harden tasks so raw session-reading isn't enough. The **token/latency win is arguably the truer memory value** here.
- The agent never called the `memory_reflect` tool — it relied only on auto-inject. A prompt nudge to use the tool for distal symptoms might lift the intervention win (cf. the earlier listmerge enrichment).

## ============ OVERNIGHT SUMMARY (what got done) ============
1. **Seeding (vanilla fairness)** ✅ — vanilla seeds the dev conversations as real opencode sessions; the agent consults them via `opencode session list`/`export` on its own initiative (validated). Availability + agency.
2. **Agent containerized** ✅ — opencode runs in a per-task container (isolated, no host-store bleed), resumes across interventions via one long-lived container (store inside; fixed the slow bind-mounted-SQLite + cold-start). Grading stays in sdebench-base.
3. **New plugin wired** ✅ — container writes `~/.hindsight/coding-agent.json` (the plugin's new JSON config; env removed). Reflect over the host server via host.docker.internal. Bank reuse across n runs (skip re-backfill).
4. **n=3 sweep vanilla vs hindsight** ✅ — see the RESULTS section. Headline: both 10/10; interventions 8.3→6.7 (−20%), **input tokens −40%, wall −19%** at equal cost. Memory's real win here is EFFICIENCY, not solve-rate (seeded vanilla is a strong baseline).
5. **AMB UI** ✅ — per-run detail shows all agent metrics (pills); added `meta.tokens`. Leaderboard is accuracy-based (all coding=100%), so read the per-run detail / this table. `uv run omb view`.
6. **claude-code stretch** ⚠️ BLOCKED — image builds; blocked on (a) auth (needs ANTHROPIC_API_KEY; host uses a Max OAuth token, and a 60-run sweep would hit the 5h subscription cap regardless) and (b) memory integration (Hindsight is an opencode plugin; claude-code needs a hook/MCP). Adapter design recorded above. **Needs your call (API key?) tomorrow.**
7. **Task expansion** ✅ — **+9 H-source tasks** (5 planted-H via gen/emit_host_h.py, 4 real-function-H via gen/emit_realfn_h.py), all validated (HEAD fails repro+hidden, correct passes, decision in git history). Dataset: **19 tasks, 9 F / 10 H, 9 real-function / 10 planted.** H category 1 → 10 as requested.

### Open questions for you
- **claude-code:** provide an ANTHROPIC_API_KEY (clean) or approve subscription use (rate-limited)? And pick the memory integration (hook / MCP / prompt-inject).
- **seeding vs memory:** the seeded vanilla is strong. Want the hindsight arm to ALSO seed (measure "memory on top of sessions"), and/or a prompt nudge to use the `memory_reflect` tool for distal symptoms?
- Re-run the n=3 sweep on the expanded 19-task set once you're happy with the H tasks?

### To reproduce / view
- Sweep: `zsh scratchpad/overnight_sweep.sh` (env: SDE_HSCODING_PLUGIN_DIR=<new plugin>, SDE_HINDSIGHT_URL=http://localhost:8899).
- Results: `outputs/sdebench/ov-{none,hs}-{1,2,3}` (+ .gz committed). Analysis: `scratchpad/analyze.py`.
- UI: `uv run omb view` → sdebench → open an ov-none-* and an ov-hs-* run.

### CORRECTION — the "−40% tokens" is mostly CHEAP cached tokens (cost is flat, not a win)
Token cost split (mean/run, gemini-3.5-flash: input $1.50/1M, cache_read $0.15/1M, output $9/1M):
- vanilla:   fresh_input 2.79M ($4.19) + cache_read 10.13M ($1.52) + output 106k ($0.96) = $6.67
- hindsight: fresh_input 3.40M ($5.10) + cache_read  4.36M ($0.65) + output  89k ($0.80) = $6.55
The −40% total-token drop is almost entirely **cache_read** (re-sent context, billed 10× cheaper). Hindsight
SAVES on cache_read (−$0.87, less back-and-forth) but SPENDS more on fresh input (+$0.91, the injected memory
is new context + changed prompts cache less) → the two cancel, cost is flat. **So the honest wins are
wall-time (−19%) and interventions (−20%); cost is a wash, and "−40% tokens" overstates it.**

## ============ CLAUDE-CODE SUPPORT (unblocked per your go-ahead) ============
Implemented `--agent claude-code` (opencode still default):
- **Auth solved** — you approved mounting the Max OAuth token; extracted from the keychain to
  `~/.sdebench/claude_creds.json`, mounted at `/root/.claude/.credentials.json`. `--dangerously-skip-permissions`
  is blocked as root, so we use `--permission-mode acceptEdits` (writes work). Fewer tasks (4×2) to stay under the sub cap.
- **Memory arm = a UserPromptSubmit hook** (no MCP — we only inject): `claude_memory_hook.py` reflects the
  prompt over the Hindsight bank + returns `additionalContext`. Inert unless a bank is set → same image for both arms.
- **run.py** — claude runs in `sdebench-agent-claude` (one long-lived container, `--continue` across
  interventions), `claude -p --output-format json --permission-mode acceptEdits -m claude-sonnet-5`;
  `_parse_claude` maps usage/num_turns/**total_cost_usd** (claude reports its own cost) to the common shape.
- **Validated**: claude vanilla solves rounding in-container, wall=38s (≈4× faster than opencode/gemini),
  cost $0.31, metrics parsed. Comparison (4 tasks: rounding/budget/listmerge/slugify, vanilla vs hindsight,
  reusing the opencode-backfilled banks) is RUNNING; results below when done.

## ============ RESULTS: claude-code (sonnet-5), vanilla vs hindsight, 4 tasks (n=1) ============
Tasks: rounding, budget, listmerge, slugify. Hindsight = the UserPromptSubmit hook reflecting over the
opencode-backfilled banks (reused). Vanilla = no memory (claude NOT seeded — opencode-only for now).

| metric (sum/4 tasks) | vanilla (sonnet-5) | hindsight | Δ |
|---|---|---|---|
| **interventions** | **5** | **2** | **−60%** |
| **cost (USD)** | **$2.64** | **$1.59** | **−40%** |
| turns | 113 | 68 | −40% |
| wall (s) | 442 | 359 | −19% |
| solved | 4/4 | 4/4 | = |

### ⭐ Answer to "is hindsight better/worse with claude?" → BETTER, and MORE than with opencode.
- The memory hook works: claude-hindsight cut interventions 5→2 and **cost 40%** (budget $0.99→$0.49, slugify $0.97→$0.42, both to 0 interventions).
- **Why bigger than opencode's win:** (1) sonnet-5 applies an injected decision more decisively (fewer exploratory turns), and (2) claude vanilla here is UNSEEDED (no past-session access), so the memory delta isn't compressed by a strong baseline like opencode's seeded vanilla was. Apples-to-apples caveat: opencode-vanilla was seeded, claude-vanilla was not — so the two agents' deltas aren't perfectly comparable yet.
- claude is also ~faster per task than opencode/gemini in these runs.

### Caveats
- n=1, 4 tasks (kept small to respect the Max subscription cap) — directional, not statistically tight.
- claude vanilla is not seeded (seeding is opencode-import based; claude uses a different session store — `~/.claude/projects/<slug>/*.jsonl`). To make the agents directly comparable, either seed claude too or run BOTH agents unseeded. Noted as follow-up.

## ============ NOISE + FULL ROSTER (session 2) ============
- **Decoy conversations (retrieval noise) ✅** — `gen/gen_decoys.py` mines the last 100 commits, clusters by module, gemini writes a long (avg 8.7-turn) codebase-grounded dev conversation per cluster with NO planted policy. 40 decoys / 346 turns in `gen/decoy_conversations.json` (verified no answer-token leaks). The coding-mode backfill ingests them alongside each task's 1 real chat (SDE_DECOYS default on) → each bank ~630 facts, so chat retrieval is a real ranking problem (~40x noise vs the old 1-chat lookup).
- **claude in the UI ✅** — coding mode passes `--agent` (SDE_AGENT) through to run.py, so claude runs via `omb` land in `outputs/` + the viewer.
- **BUG FIXED** — `SDE_HSCODING_PLUGIN_DIR`/`SDE_CLAUDE_CREDS` with a literal `~` weren't expanded (Path() doesn't expand ~), so docker rejected the mount ('invalid volume name') AND the plugin backfill couldn't find backfill.js → the memory arm silently ran on an EMPTY bank. Fixed with expanduser + a docker-run retry. (The earlier n=3 sweep was unaffected — its zsh `export VAR=~/...` did expand; only inline `env VAR=~/...` smokes hit it.)
- **FULL ROSTER running** — opencode 19 tasks × {vanilla, hindsight+noise}, then claude 19×2 reusing the noisy banks. Results + comparison to follow.

## ============ NEW H-TASK INVESTIGATION (per user request) ============
### Structural check ✅ (done, before runs)
Full discrimination matrix validated for all 9 new H tasks (5 planted-H + 4 realfn-H):
- HEAD: existing/real tests PASS, repro FAIL, hidden FAIL
- correct fix: ALL pass
- **naive fix: repro PASS, hidden FAIL** ← the non-guessable guarantee (a symptom-only fix is rejected)
- decision present in git history (the H source)
No structural/discrimination bugs. The tasks are well-formed and non-guessable.

### "Too easy" check (empirical) — PENDING the roster results
Interpretation for H tasks: the decision lives in git history, which vanilla CAN reach via git log/blame.
Signals to flag for REFINEMENT once results land:
- **vanilla mean interventions ≈ 0** on an H task → too easy: the agent solves without really needing
  the decision (symptom too revealing, or agent reliably reads history) → memory adds nothing → harden it
  (make symptom more distal from cause, like omdset does — its symptom is 2 modules from the __setitem__ cause).
- vanilla high-interv + hindsight low → GOOD (memory discriminates).
- Will compute per-H-task vanilla-vs-hindsight from the full roster and list any "too easy" ones with a
  concrete hardening suggestion.

### H-task empirical signal — VANILLA arm (noise, n=1) — H tasks are NOT too easy overall
Vanilla interventions on the 10 H tasks: budget-h=0, discount-h=3, findhashtags-h=1, listmerge-h=1,
omdset=2, parseflag-h=1, pluralize-h=3, rounding-h=1, slugify-h=1, under2camel-h=1  →  **total 14, mean 1.4**.
- So vanilla does NOT trivially solve them (only budget-h at 0) → the H tasks are NOT too easy as a set.
- **budget-h = 0 interventions** is the one to watch (n=1; could be noise). Recheck vs hindsight + more n.

### ⚠️ STRUCTURAL refinement lever (found): H commit messages are OVER-EXPLICIT
My 9 new H-task decision commits STATE THE LITERAL ANSWER in the message body — e.g. budget-h says
"MAX_ATTEMPTS is 7", rounding-h says "ROUND_HALF_DOWN so 2.135 -> 2.13". So an agent that greps `git log`
finds the exact answer with zero reasoning. Contrast omdset (the hard H task): its commit states the
INVARIANT ("__setitem__ must overwrite the stored value sequence… add() appends to the stale list") but
NEVER names the symptom (getlist / query params) or gives a grep-able literal — the agent must reason
from invariant→symptom.
- **Recommendation:** to harden the 9 new H tasks toward omdset's bar, rewrite their commit bodies to
  give the RATIONALE without the literal answer/symptom vocabulary (state WHY, not the exact value). That
  raises the reasoning bar and widens the vanilla-vs-hindsight gap. Currently they're valid but "easy H"
  (answer reachable by grep), whereas omdset is "hard H" (answer requires reasoning). Both are legitimate
  H tasks; if you want them harder, this is the knob. Awaiting hindsight-arm data to quantify the gap.

## ============ RESULTS: opencode FULL 19-task roster, WITH noise (n=1) ============
Vanilla (seeded, F only) vs Hindsight (plugin, per-task bank = real chat + 40 decoys + 100 git commits,
~630 facts). Both 19/19 solved.

| metric (sum/19 tasks) | VANILLA | HINDSIGHT | Δ |
|---|---|---|---|
| **interventions** | **26** | **15** | **−42%** |
| **cost (USD)** | **$15.5** | **$10.9** | **−30%** |
| turns | 754 | 616 | −18% |
| solved | 19/19 | 19/19 | = |

### ⭐ Noise makes memory clearly win (stronger than session-1's seeded/no-noise 8.3→6.7)
- With realistic decoy noise + the full roster, hindsight cuts interventions **42%** AND cost **30%** —
  a much bigger, cost-positive win than session 1. The harder/noisier setting is where memory pays off.
- H-tasks only (10): vanilla 14 → hindsight 8 interventions (−43%) — memory helps on H too.

### H-task "too easy" verdict (per user)
Per-H-task vanilla→hindsight interventions: budget-h 0→1, discount-h 3→1, findhashtags-h 1→1,
listmerge-h 1→1, omdset 2→1, parseflag-h 1→1, pluralize-h 3→1, rounding-h 1→0, slugify-h 1→1,
under2camel-h 1→0.
- **9/10 H tasks are appropriately hard** — vanilla needs 1-3 interventions; hindsight helps or ties.
- **⚠️ budget-h is TOO EASY** — vanilla solved it with 0 interventions (and hindsight took 1). It's the
  most grep-able: its decision commit literally says "MAX_ATTEMPTS is 7". FLAG FOR REFINEMENT.
- Several H tasks are 1→1 (flat) at n=1 — memory neither helps nor hurts; need n=3 to tell signal from noise.
- **Systemic hardening lever (repeat):** the 9 new H commit bodies state the literal answer, so a git-log
  grep solves them. Rewriting them to give RATIONALE without the literal value/symptom (omdset-style) would
  raise difficulty and widen the vanilla-vs-hindsight gap. Concrete next step if you want harder H tasks.

### H-task hardening — the real lever is SYMPTOM→CAUSE DISTANCE, not hiding the answer (nuance)
Thinking it through: for H tasks the decision MUST be stated in a git commit (that's how it's reachable),
so "hide the literal value" conflicts with solvability — budget's "7" is measured/non-derivable, memory
HAS to state it. So the commit will always contain the answer somewhere. What makes omdset hard isn't
hiding the answer — it's that:
  1. the commit describes the INVARIANT, not the reported SYMPTOM (getlist/query params), and
  2. the symptom manifests ~2 modules from the cause (__setitem__), so the agent doesn't know which
     commit/function to look at — a `git log` grep on the symptom's vocabulary won't surface it.
My 9 new H tasks put the fix and the symptom in the SAME function/module and use matching vocabulary, so
`git log -S <symptom-term>` or `git blame <the-file>` lands on the answer commit immediately → easy.

**Concrete refinement options (for your call):**
- (A) Cheapest: accept them as "easy-H" (still valid, still discriminate) and rely on omdset as the one
  "hard-H". The empirical data says 9/10 still need ≥1 vanilla intervention, so they're not trivially broken.
- (B) Medium: reword commit subjects/bodies to NOT echo the bug-report vocabulary (so symptom-term grep
  misses), keeping the rationale. Quick, raises the "does the agent think to look?" bar.
- (C) Full omdset-style: increase symptom→cause distance (bug manifests in a different function than the
  planted decision). Best discrimination, most work per task.
Recommend (B) as the default hardening pass; (C) for a few flagship hard-H tasks. budget-h specifically:
vanilla=0 at n=1 — confirm with the claude data point + a couple more opencode samples before acting.

## ============ RESULTS: claude-code (sonnet-5) FULL 19-task roster, WITH noise (n=1) ============
Claude vanilla = NO memory (unseeded). Claude hindsight = the UserPromptSubmit reflect hook over the
same noisy banks opencode backfilled. Results now in the UI (nz-cc-none / nz-cc-hs).

| metric (sum/19) | VANILLA | HINDSIGHT | Δ |
|---|---|---|---|
| interventions | 12 | 13 | +1 (flat/slightly worse) |
| cost (USD) | $8.82 | $8.50 | −4% |
| turns | 412 | 353 | −14% |
| solved | 19/19 | 18/19 | −1 (a regression) |

### ⭐⭐ BIG FINDING #1 — for a STRONG agent (sonnet-5), the H tasks are ALL too easy
**Every one of the 10 H tasks: claude vanilla = 0 interventions** — including omdset (the "hard H").
sonnet-5 reliably reads `git log`/`blame`, finds the planted decision commit, and fixes it first try
WITHOUT memory. So H-source tasks give ZERO signal for a capable agent (vanilla 0 → hindsight 0 on all H).
- This confirms + generalizes the "too easy" concern: it's not just budget-h — it's the whole H category,
  for strong agents. The H-source premise ("the agent CAN reach git history but often doesn't think to")
  DOES NOT HOLD for sonnet-5; it always thinks to.
- Weaker agent (opencode/gemini) DID get signal from H (vanilla 14 → hindsight 8), because gemini doesn't
  reliably mine git history. So H-task difficulty is AGENT-DEPENDENT.

### ⭐⭐ BIG FINDING #2 — with noise, memory is NEUTRAL/slightly-HARMFUL for the strong agent
Claude hindsight ≈ vanilla (12→13 interv) and caused 2 regressions: **findhashtags F 1→5 and UNSOLVED**
(memory misled it), under2camel F 2→4. Contrast the earlier no-noise 4-task claude test (5→2, memory
helped). So under decoy noise, reflect sometimes surfaces distracting/adjacent context that HURTS a strong
agent that would otherwise solve it. (opencode/gemini still benefited from memory under noise — weaker
agent has more to gain, less to lose.)

### Net takeaways (for the morning)
1. **H tasks need hardening to matter for strong agents** — sonnet-5 trivially git-logs the answer.
   Fix = symptom→cause distance (make the bug manifest far from the planted commit, omdset-style) AND/OR
   don't let the commit vocabulary match the bug-report vocabulary. As-is, H tasks only discriminate for
   weaker agents.
2. **Memory value is agent-dependent:** big win for gemini (26→15), ~neutral/harmful for sonnet-5 under
   noise. Worth investigating the findhashtags regression (did reflect surface a wrong rule under noise?).
3. F tasks still the better discriminators for strong agents (they need the chat, not git).

### CORRECTION on Finding #2 — the memory content was CORRECT; regressions are likely n=1 variance
Checked the findhashtags bank's reflect under noise: it surfaces the EXACT correct rule ("exclude
all-digit tags except 4-digit years 1900-2099"). So the 1→5-unsolved claude regression is NOT bad memory
— the injected decision was right. It's either injection-induced behavior change or (more likely) n=1
variance in how claude implemented/verified it. So do NOT conclude "memory hurts strong agents" — the
honest statement is: **at n=1 with noise, claude's memory arm was ~flat with high per-task variance
(±a few interventions either way), and the memory content itself was accurate.** Need n=3 to separate
signal from noise on the claude arm. Finding #1 (H tasks too easy for sonnet-5, all vanilla=0) is robust
— it's a clean 10/10 zeros, not variance.
