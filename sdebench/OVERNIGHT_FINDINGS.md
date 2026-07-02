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
