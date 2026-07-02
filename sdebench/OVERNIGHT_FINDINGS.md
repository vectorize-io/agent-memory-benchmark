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
