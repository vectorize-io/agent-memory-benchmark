# sdebench — does memory help a coding agent?

A benchmark that measures whether a coding agent, working on a **real codebase**, benefits from a
**memory system** that has ingested the project's git rationale and past developer conversations.

Each task is a bug-fix whose correct solution hinges on a **non-guessable, project-specific
decision**: the obvious fix passes the visible repro but fails a **held-out hidden test**. Where
that decision lives (git history vs a past chat vs a chat later amended) is the dataset's main
axis — and what each arm can *reach* is the experiment:

| | plain agent | agent + memory |
|---|---|---|
| the repo + full git history | ✅ | ✅ |
| past developer chats | ✅ reachable (seeded, see below) | ✅ ingested & surfaced |
| a memory system surfacing the decision | ❌ | ✅ |

The dataset lives in the [sde-bench](https://github.com/vectorize-io/sde-bench) submodule at
`sdebench/datasets` (boltons-hosted tasks; the census is its `MANIFEST.json`, the datasheet its
`DATASET.md`, authoring guide `GENERATING.md`).

## Grading (a task is solved iff)

1. `FAIL_TO_PASS` — the visible repro (shipped with the bug report) now passes.
2. `PASS_TO_PASS` — the pre-existing suite still passes (no collateral breakage).
3. `HIDDEN_TO_PASS` — the held-out policy test passes (defeats guessing the obvious fix).

Graded in Docker from a **pristine copy** with only the agent's source patch applied — agent edits
to test files are ignored. Resolution is binary.

**Primary metric: corrections (`interventions`)** — on a failing grade the harness feeds the
failing-test output back (like a reviewer would) and resumes the same session, cap 5. `0` = solved
first try with no help. Also reported: turns, cost, wall time, tokens, solve rate.

## Agents

Three agent stacks, selected with `--agent` (standalone harness) or `SDE_AGENT` (OMB path):

| `--agent` | model | image | memory delivery (memory arm) |
|---|---|---|---|
| `opencode` | `google/gemini-3.5-flash` | `sdebench-agent` | the `hindsight-coding-agents` opencode plugin, mounted + configured in-container |
| `claude-code` | `claude-sonnet-5` | `sdebench-agent-claude` | the plugin's `UserPromptSubmit` hook (`claude-hook.js`), wired exactly as the installer does (the real product path) |
| `codex` | `gpt-5.1-codex-mini` | `sdebench-agent-codex` | Codex hooks running the plugin's `codex-hook.js` (the real product path) |

Build the images from `sdebench/Dockerfile.agent*`; the grading image is `sdebench/Dockerfile`
(`sdebench-base`). Auth: opencode/codex take API keys from the environment (`OPENAI_API_KEY` is
passed through and logged in in-container for codex); claude-code mounts OAuth credentials from
`~/.sdebench/claude_creds.json`.

**Vanilla-arm fairness**: on `conversation`-source tasks the past developer chats are made
*reachable* by the plain agent — seeded as native opencode sessions for `opencode`, and as markdown
transcripts under `/root/project-history/` for `codex`/`claude-code`. The prompt does NOT point at
them (availability, not advertisement): whether the agent thinks to look for prior context is part
of what's measured. That keeps the comparison "reliable surfacing" vs "available but unprompted",
not "access" vs "no access".

## Running

**Via the OMB runner** (results land in `outputs/` + the viewer):

```bash
uv run omb run --dataset sdebench --split boltons --mode coding --memory vanilla     # no-memory baseline
SDE_HINDSIGHT_URL=http://localhost:8888 \
  uv run omb run --dataset sdebench --split boltons --mode coding --memory hindsight-coding  # agent + memory
```

Memory flows through OMB's **standard provider pipeline**: the dataset exposes each task's
knowledge corpus (`isolation_unit = "task"` — decision chat/commit + decoy conversations +
host-history noise), the runner ingests it into the selected provider, and the coding mode
dispatches:

- `--memory vanilla` — no-memory baseline (`none` is a legacy alias).
- `--memory hindsight-coding` — the Hindsight plugin (`hscoding` is a legacy alias): its provider ingests via the plugin's own **deepen
  engine** over the BUILT repo (`--git-ingest full`, + conversations) and polls `status.js` until
  `synced`; delivery is agent-side (reflect+inject inside the harness).
- `--memory <any other provider>` (bm25, mem0, …) — generic path: the runner ingests the task
  corpus, the mode calls `provider.retrieve(bug_report)` and injects the top memories into the
  task prompt (`provided` arm). Any AMB memory system can run the coding benchmark.

`--skip-ingestion` reuses existing memory state across n-runs (for `hscoding`: populated banks).

Env: `SDE_AGENT` (`opencode`|`claude-code`|`codex`), `SDE_HINDSIGHT_URL` (server),
`SDE_HSCODING_PLUGIN_DIR` (plugin checkout with `dist/` built — `hscoding` only).

Note: `-q N` selects the first N tasks **alphabetically** — a subset, not a sample.

**Standalone harness** (direct, more arms/flags):

```bash
uv run python sdebench/harness/run.py \
    --task sdebench/datasets/boltons-<name>/tasks/main/task.json \
    --agent {opencode|claude-code|codex} --history {full|hscoding} --run-id <id>
```

`--history full` = vanilla (full git history); `--history hscoding` = the memory arm. Other arms
(`oracle`, `inject`, `index`, …) are research modes — see `harness/run.py`.

**Charts**: `uv run --with matplotlib python scripts/sdebench_charts.py` renders
corrections/cost/turns per task per agent (globs n>1 reruns, error bars; agents without runs are
dropped).

## Layout

```
sdebench/
  datasets/               # -> sde-bench submodule: tasks + generator (gen/) + datasheet
  harness/run.py          # the engine: build repo -> agent -> corrections loop -> pytest grade
                          #   (the OMB `coding` mode shells out to this)
  Dockerfile              # grading env (python + pytest + git)   -> sdebench-base
  Dockerfile.agent*       # agent envs (opencode / claude / codex)
```

## Design rule

The decision must be **non-guessable** — a conventional value or rule the agent can guess without
memory won't discriminate the arms. Every task ships proof: the generator validates that HEAD fails
repro+hidden, the correct fix passes everything, and each plausible naive fix passes the repro but
fails hidden. Memory systems are expected to surface the *decision*, not hidden-test values (no
answer leakage); the agent still writes and tests its own fix.
