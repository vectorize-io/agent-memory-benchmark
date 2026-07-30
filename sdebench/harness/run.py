"""sdebench harness — run a coding agent on a regression task and grade it.

Flow: build the repo -> ship the agent the bug report + failing regression test -> run the agent
(--agent opencode|claude-code|codex) -> on a failing grade, feed the pytest tail back and resume
(the corrections loop, cap --max-interventions) -> capture the SOURCE diff (tests excluded) ->
grade in Docker against FAIL_TO_PASS + PASS_TO_PASS + HIDDEN_TO_PASS from pristine copies.

Usage:
    uv run python sdebench/harness/run.py --task <task.json> --agent opencode --history full     # vanilla
    uv run python sdebench/harness/run.py --task <task.json> --agent codex   --history hscoding  # memory

Metrics reported: solved (binary), interventions, turns, wall, tokens, cost (from PRICES;
claude-code self-reports).
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
SDEBENCH = HARNESS.parent
REPO_ROOT = SDEBENCH.parent
IMAGE = "sdebench-base"


def _codebase_dir(task):
    """Dir holding build.py for this task's shared codebase."""
    return SDEBENCH / "datasets" / (task.get("codebase") or task["repo"])


def _task_dir(task):
    """Dir holding this task's own regression_test.py / hidden_test.py (task.json's dir)."""
    return Path(task["_dir"])

# $ per 1M tokens, per class (update when model prices change). gemini-3.5-flash (Jun 2026):
# $1.50 input / $9.00 output, cached input 90% off ($0.15). reasoning bills as output.
PRICES = {
    "google/gemini-3.5-flash": {"input": 1.50, "cache_read": 0.15, "cache_write": 1.50, "output": 9.00},
    # gpt-5.1-codex-mini (Jul 2026): $0.25 in / $2.00 out; cached input 90% off; OpenAI does not
    # bill cache writes separately (they price as normal input).
    "gpt-5.1-codex-mini": {"input": 0.25, "cache_read": 0.025, "cache_write": 0.25, "output": 2.00},
}


def compute_cost(model: str, tok: dict) -> float:
    p = PRICES.get(model)
    if not p:
        return 0.0
    return round((tok["input"] * p["input"] + tok["cache_read"] * p["cache_read"]
                  + tok["cache_write"] * p["cache_write"]
                  + (tok["output"] + tok["reasoning"]) * p["output"]) / 1_000_000, 4)

PROMPT = """\
You are a maintainer of the `{repo}` Python project. A regression was reported:

{bug_report}

Fix the bug in the source code. Do NOT modify any test files — the graders supply their own.
{instruction}
Save your changes to disk before finishing.
"""

# behavioral prompt variants (applied uniformly to ALL arms = fair). Select via SDE_VARIANT.
VARIANTS = {
    "base": ("Work efficiently: find the root cause, make the smallest change that fixes it, run the "
             "failing test to confirm it passes (and existing behaviour still works), then stop — "
             "avoid unnecessary exploration."),
    "hypothesis": ("Before making ANY edit, state in one sentence your hypothesis for the root cause "
                   "(which file and function, and why). Then make the single smallest change that fixes "
                   "it and run the failing test once to confirm; then stop."),
    "minimal": ("The fix is almost always ONE small change in ONE file — do not read widely, refactor, "
                "or add new code. Find the root cause, make that one change, run the failing test to "
                "confirm, then stop."),
}


def sh(*args, cwd=None, env=None, check=True, cap=False):
    return subprocess.run(args, cwd=cwd, env=env, check=check,
                          capture_output=cap, text=True, errors="replace")


def neutral_home() -> str:
    """A HOME without ~/.claude so opencode can't load the user's global CLAUDE.md."""
    home = Path("/tmp/sdebench_home")
    if not home.exists():
        home.mkdir(parents=True, exist_ok=True)
        for item in Path.home().iterdir():
            if item.name.startswith(".") and item.name != ".claude":
                try:
                    (home / item.name).symlink_to(item)
                except FileExistsError:
                    pass
    return str(home)


def build_repo(task: dict, dest: Path, history: str):
    if dest.exists():
        shutil.rmtree(dest)
    ds = _codebase_dir(task)
    sh("python", str(ds / task["build"]), str(dest))
    if history == "squashed":
        shutil.rmtree(dest / ".git")
        sh("git", "init", "-q", cwd=dest)
        sh("git", "add", "-A", cwd=dest)
        env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
               "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
        sh("git", "commit", "-q", "-m", "Initial commit", cwd=dest, env=env)
    # ship the failing regression repro (the agent sees it; it is red)
    ds_test = _task_dir(task) / task["regression_test_file"]
    shutil.copy(ds_test, dest / "tests" / "test_regression.py")
    sh("git", "add", "-A", cwd=dest)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
           "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    sh("git", "commit", "-q", "-m", "test: add failing repro for the reported regression",
       cwd=dest, env=env)


HINDSIGHT_URL = os.environ.get("SDE_HINDSIGHT_URL", "http://localhost:8888")


def load_env(memory_bank: str | None = None, mem_index: str | None = None, conv_log: str | None = None) -> dict:
    env = os.environ.copy()
    ef = REPO_ROOT / ".env"
    if ef.exists():
        for line in ef.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if conv_log:      # enable the recall_conversations skill (memory of past user sessions)
        env.pop("HINDSIGHT_DISABLED", None)
        env["CONV_LOG"] = conv_log
    elif mem_index:   # enable the LOCAL recall_intent tool over the raw-commit index
        env.pop("HINDSIGHT_DISABLED", None)
        env["MEM_INDEX"] = mem_index
    elif memory_bank: # enable the Hindsight opencode plugin pointed at this bank (recall mode)
        env.pop("HINDSIGHT_DISABLED", None)
        env["HINDSIGHT_API_URL"] = HINDSIGHT_URL
        env["HINDSIGHT_BANK_ID"] = memory_bank
        env["HINDSIGHT_MEMORY_MODE"] = "recall"
    else:
        env["HINDSIGHT_DISABLED"] = "1"   # plain agent: no memory/plugins, just git via bash
    env["PWD"] = ""                   # set per-run
    env["HOME"] = neutral_home()
    return env


def cli_env() -> dict:
    e = os.environ.copy()
    e["HINDSIGHT_API_URL"] = HINDSIGHT_URL
    return e


def ingest_history(task: dict, bank: str):
    """Build the full repo and push each commit (message + diff) into a Hindsight bank,
    so a squashed-repo agent can RECALL the history it can't `git blame` for."""
    src = Path("/tmp/sdebench/ingest") / task["repo"]
    if src.exists():
        shutil.rmtree(src)
    sh("python", str(_codebase_dir(task) / task["build"]), str(src))
    subprocess.run(["hindsight", "bank", "delete", bank, "--yes"], env=cli_env(),
                   capture_output=True)
    shas = sh("git", "-C", str(src), "rev-list", "--reverse", "HEAD", cap=True).stdout.split()
    for sha in shas:
        msg = sh("git", "-C", str(src), "show", "-s", "--format=%s%n%n%b", sha, cap=True).stdout
        diff = "\n".join(sh("git", "-C", str(src), "show", "--format=", sha, cap=True).stdout.splitlines()[:120])
        subprocess.run(["hindsight", "memory", "retain", bank,
                        f"Git commit in the {task['repo']} repo: {msg}\n\nDiff:\n{diff}"],
                       env=cli_env(), capture_output=True)
    print(f"[ingest] {len(shas)} commits -> bank {bank}; waiting for extraction…", flush=True)
    time.sleep(18)


def build_mem_index(task: dict) -> str:
    """Build the local raw-commit index for the recall_intent tool (from the full history)."""
    out = Path("/tmp/sdebench/memindex") / f"{task.get('codebase') or task['repo']}.json"
    sh("python", str(HARNESS / "mem_index.py"), str(_codebase_dir(task) / task["build"]), str(out))
    return str(out)


_STOP = {"the","and","for","that","this","with","what","why","how","value","should","change",
         "changed","does","when","over","its","into","use","used","using","make","made","not","but"}


def rank_commits(index_path: str, query: str, k: int = 2) -> list:
    """TF-rank the codebase's raw commits by a query (same scoring as the recall_intent tool)."""
    commits = json.loads(Path(index_path).read_text())
    terms = [t for t in re.findall(r"[a-z0-9_]{3,}", query.lower()) if t not in _STOP]
    scored = []
    for c in commits:
        subj, files = c["subject"].lower(), " ".join(c["files"]).lower()
        body, diff = (c.get("body") or "").lower(), c["diff"].lower()
        sc = 0
        for t in terms:
            if t in subj: sc += 5
            if t in files: sc += 4
            if t in body: sc += 2
            sc += min(diff.count(t), 6)
        if sc > 0:
            scored.append((sc, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def _changed_lines(diff: str, cap: int = 24) -> str:
    out = [l for l in diff.split("\n")
           if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))]
    return "\n".join(out[:cap])


def inject_context(bug_report: str, commits: list) -> str:
    """PUSH memory: append the relevant commits' changed lines to the bug report."""
    if not commits:
        return bug_report
    blocks = [f"commit {c['sha']} — {c['subject']}\nfiles: {', '.join(c['files'])}\n{_changed_lines(c['diff'])}"
              for c in commits]
    return (bug_report + "\n\nFor context, here are some recent changes to this repository that "
            "may be relevant:\n\n" + "\n\n----\n\n".join(blocks))


def capture_git_history(task: dict) -> list:
    """The task repo's engineered git history (commits + diffs) — the 'source documents'
    the full/hindsight arms have access to and the squashed arm does not. Newest first."""
    src = Path("/tmp/sdebench/hist") / task["repo"]
    if src.exists():
        shutil.rmtree(src)
    sh("python", str(_codebase_dir(task) / task["build"]), str(src))
    out = []
    # cap for large real-codebase hosts (the UI view of thousands of commits is useless and slow)
    for sha in sh("git", "-C", str(src), "rev-list", "HEAD", "-n", "40", cap=True).stdout.split():
        subject = sh("git", "-C", str(src), "show", "-s", "--format=%s", sha, cap=True).stdout.strip()
        body = sh("git", "-C", str(src), "show", "-s", "--format=%b", sha, cap=True).stdout.strip()
        diff = "\n".join(sh("git", "-C", str(src), "show", "--format=", sha, cap=True).stdout.splitlines()[:150])
        out.append({"sha": sha[:8], "subject": subject, "body": body, "diff": diff})
    return out


def gen_index_doc(task: dict) -> str:
    """Generate a compact DECISIONS.md index from the codebase's git history — mechanical, not
    hand-authored. Collates each non-noise commit's subject + rationale (body) + files + sha, so
    the agent reads a curated 1-page index instead of reconstructing via `git log -p`. This is the
    derivable memory: good commit messages -> a usable index for H/X/K alike."""
    src = Path("/tmp/sdebench/idxsrc") / task["repo"]
    if src.exists():
        shutil.rmtree(src)
    sh("python", str(_codebase_dir(task) / task["build"]), str(src))
    import re as _re
    skip = _re.compile(r"^(chore|release|bump|ci|style)\b", _re.I)
    lines = ["# Project decisions & changes",
             "_An index of notable changes, derived from git history. Each entry references the commit; "
             "consult it for the rationale behind the current code._\n"]
    for sha in sh("git", "-C", str(src), "rev-list", "--reverse", "HEAD", cap=True).stdout.split():
        subj = sh("git", "-C", str(src), "show", "-s", "--format=%s", sha, cap=True).stdout.strip()
        body = sh("git", "-C", str(src), "show", "-s", "--format=%b", sha, cap=True).stdout.strip()
        files = sh("git", "-C", str(src), "show", "--name-only", "--format=", sha, cap=True).stdout.split()
        if skip.match(subj) and not body:
            continue
        code = [x for x in files if x.endswith(".py") and not x.startswith("tests/")]
        entry = f"- **{subj}**"
        if code:
            entry += f" — `{', '.join(code)}`"
        if body:
            entry += f"\n  {body}"
        entry += f" (commit {sha[:8]})"
        lines.append(entry)
    return "\n".join(lines)


# The coding agent is pluggable (--agent). opencode: gemini + the Hindsight opencode plugin (reflect
# auto-injected via ~/.hindsight/coding-agent.json). claude-code: sonnet-5 + a UserPromptSubmit hook
# that reflects+injects (no MCP), OAuth creds mounted at runtime.
_AGENT_IMAGES = {"opencode": os.environ.get("SDE_AGENT_IMAGE", "sdebench-agent"),
                 "claude-code": os.environ.get("SDE_AGENT_IMAGE_CLAUDE", "sdebench-agent-claude"),
                 "codex": os.environ.get("SDE_AGENT_IMAGE_CODEX", "sdebench-agent-codex")}
_AGENT_MODEL = {"opencode": "google/gemini-3.5-flash", "claude-code": "claude-sonnet-5",
                "codex": "gpt-5.1-codex-mini"}
# The hindsight-coding-agents plugin package (dist/ built) — memory arms only. No default that
# assumes a particular machine layout: point SDE_HSCODING_PLUGIN_DIR at a checkout of
# hindsight-integrations/hindsight-coding-agents (or the npm-installed package dir).
_PLUGIN_DIR = os.path.expanduser(os.environ.get("SDE_HSCODING_PLUGIN_DIR", ""))
_CLAUDE_CREDS = os.path.expanduser(os.environ.get("SDE_CLAUDE_CREDS", str(Path.home() / ".sdebench" / "claude_creds.json")))


def _container_url(url: str) -> str:
    """Rewrite a host-local URL so the agent CONTAINER can reach it (host's localhost)."""
    return url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")


def _mem_docker_env(env: dict) -> list[str]:
    """Docker -e flags carrying model auth + memory settings (both agents read HINDSIGHT_*; the
    opencode plugin also gets a config file, the claude hook reads these env vars directly)."""
    denv: list[str] = []
    key = env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if key:
        denv += ["-e", f"GEMINI_API_KEY={key}", "-e", f"GOOGLE_GENERATIVE_AI_API_KEY={key}"]
    okey = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if okey:
        denv += ["-e", f"OPENAI_API_KEY={okey}"]
    for k in ("HINDSIGHT_DISABLED", "HINDSIGHT_BANK_ID", "HINDSIGHT_MEMORY_MODE"):
        if env.get(k) is not None:
            denv += ["-e", f"{k}={env[k]}"]
    if env.get("HINDSIGHT_API_URL"):
        denv += ["-e", f"HINDSIGHT_API_URL={_container_url(env['HINDSIGHT_API_URL'])}"]
    return denv


def start_agent_container(workdir: Path, env: dict, agent: str = "opencode") -> str:
    """Start ONE long-lived agent container per task (session store INSIDE it so `-c`/`--continue`
    resumes across the intervention loop cheaply). Grading stays in sdebench-base. Returns the id."""
    mounts = ["-v", f"{workdir}:/work"]
    if agent in ("opencode", "codex") and _PLUGIN_DIR:  # plugin (and its codex hook) — memory arms
        mounts += ["-v", f"{_PLUGIN_DIR}:/opt/hindsight-coding-agents:ro"]
    if agent == "claude-code":
        mounts += ["-v", f"{_CLAUDE_CREDS}:/root/.claude/.credentials.json"]  # rw: claude may refresh it
    cmd = ["docker", "run", "-d", "--rm", *mounts, "-w", "/work",
           "--add-host", "host.docker.internal:host-gateway",
           *_mem_docker_env(env), _AGENT_IMAGES[agent], "sleep", "infinity"]
    cid = ""
    for attempt in range(4):  # `docker run` can transiently exit 125 under load — retry
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            cid = p.stdout.strip()
            break
        print(f"  [docker] start attempt {attempt+1}/4 failed (exit {p.returncode}): {p.stderr.strip()[:200]}", flush=True)
        time.sleep(3 * (attempt + 1))
    if not cid:
        raise RuntimeError(f"docker run failed after retries: {p.stderr.strip()[:300]}")
    if agent == "opencode":
        # opencode's plugin reads ~/.hindsight/coding-agent.json (not env): disabled=vanilla; bank+url=memory.
        cfg: dict = {"disabled": env.get("HINDSIGHT_DISABLED") == "1", "gitSync": {"enabled": False},
                     # trial isolation for the v2 plugin: no session write-back into the bank between
                     # rounds/runs, no auto-seed or survey racing the controlled backfill
                     "retainSessions": False, "autoSeed": False, "codebaseSurvey": False}
        if env.get("HINDSIGHT_BANK_ID"):
            cfg["bankId"] = env["HINDSIGHT_BANK_ID"]
        if env.get("HINDSIGHT_API_URL"):
            cfg["apiUrl"] = _container_url(env["HINDSIGHT_API_URL"])
        subprocess.run(["docker", "exec", "-i", cid, "sh", "-c",
                        "mkdir -p /root/.hindsight && cat > /root/.hindsight/coding-agent.json"],
                       input=json.dumps(cfg), capture_output=True, text=True)
    elif agent == "codex":
        # API-key auth: login in-container (the host auth.json holds ChatGPT tokens that go stale).
        subprocess.run(["docker", "exec", cid, "sh", "-c",
                        "printenv OPENAI_API_KEY | codex login --with-api-key"],
                       capture_output=True, text=True)
        # Memory via the ACTUAL product integration: the hindsight-codex-hook wired through Codex's
        # own hooks mechanism. Vanilla arm: no hooks file -> no memory (parity by absence).
        if env.get("HINDSIGHT_BANK_ID"):
            hooks = {"hooks": {"UserPromptSubmit": [{"hooks": [
                {"type": "command", "command": "node /opt/hindsight-coding-agents/dist/codex-hook.js",
                 "timeout": 150}]}]}}
            hs_cfg = {"apiUrl": _container_url(env.get("HINDSIGHT_API_URL", HINDSIGHT_URL)),
                      "bankId": env["HINDSIGHT_BANK_ID"]}
            subprocess.run(["docker", "exec", "-i", cid, "sh", "-c",
                            "mkdir -p /root/.codex /root/.hindsight && cat > /root/.codex/hooks.json"],
                           input=json.dumps(hooks), capture_output=True, text=True)
            subprocess.run(["docker", "exec", "-i", cid, "sh", "-c",
                            "cat > /root/.hindsight/coding-agent.json"],
                           input=json.dumps(hs_cfg), capture_output=True, text=True)
            subprocess.run(["docker", "exec", cid, "sh", "-c",
                            "printf 'codex_hooks = true\n' >> /root/.codex/config.toml || "
                            "printf 'codex_hooks = true\n' > /root/.codex/config.toml"],
                           capture_output=True, text=True)
    return cid


def stop_agent_container(cid: str) -> None:
    if cid:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True)


def _parse_claude(stdout: str, elapsed: float) -> dict:
    """Parse claude-code's `--output-format json` result into the common {tokens,turns,cost,...} shape."""
    tok = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    traj = []
    try:
        d = json.loads(stdout.strip().splitlines()[-1])
    except Exception:
        return {"elapsed": elapsed, "tokens": tok, "turns": 0, "trajectory": traj, "cost": 0.0}
    if d.get("is_error"):
        # Fail LOUDLY: a broken agent (expired OAuth, api_error) otherwise reads as a normal
        # 0-token turn and silently burns the whole intervention budget in seconds.
        raise RuntimeError(f"claude agent error: {str(d.get('result'))[:200]}")
    u = d.get("usage", {}) or {}
    tok["input"] = u.get("input_tokens", 0) or 0
    tok["output"] = u.get("output_tokens", 0) or 0
    tok["cache_read"] = u.get("cache_read_input_tokens", 0) or 0
    tok["cache_write"] = u.get("cache_creation_input_tokens", 0) or 0
    txt = d.get("result", "")
    if txt:
        traj.append({"k": "say", "text": str(txt)[:1500]})
    return {"elapsed": elapsed, "tokens": tok, "turns": d.get("num_turns", 0) or 0,
            "trajectory": traj, "cost": d.get("total_cost_usd", 0.0) or 0.0}


def _parse_codex(stdout: str, elapsed: float) -> dict:
    """Parse `codex exec --json` JSONL (observed schema, codex-cli 0.145):
    {"type":"item.completed","item":{"type":"command_execution"|"agent_message"|"reasoning"|...}}
    {"type":"turn.completed","usage":{input_tokens,cached_input_tokens,cache_write_input_tokens,
                                      output_tokens,reasoning_output_tokens}}"""
    tok = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    turns = 0
    traj = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        et = e.get("type")
        if et == "item.completed":
            item = e.get("item") or {}
            it = item.get("type")
            if it in ("agent_message",):
                txt = str(item.get("text") or "")[:1500]
                if txt:
                    traj.append({"k": "say", "text": txt})
            elif it not in ("reasoning", None):  # command_execution, patch_apply, tool calls, ...
                turns += 1
                arg = str(item.get("command") or item.get("path") or item.get("text") or "")[:160]
                traj.append({"k": "tool", "tool": str(it), "arg": arg, "input": "", "out": ""})
        elif et == "turn.completed":
            u = e.get("usage") or {}
            tok["input"] += u.get("input_tokens", 0) or 0
            tok["cache_read"] += u.get("cached_input_tokens", 0) or 0
            tok["cache_write"] += u.get("cache_write_input_tokens", 0) or 0
            tok["output"] += u.get("output_tokens", 0) or 0
            tok["reasoning"] += u.get("reasoning_output_tokens", 0) or 0
    return {"elapsed": elapsed, "tokens": tok, "turns": turns, "trajectory": traj}


def _reflect_query(goal: str) -> str:
    """Mirror of the plugin's buildReflectQuery (src/core/inject.ts) so the claude arm — which has
    no plugin and reflects harness-side — sends the same historian-framed prompt as the product:
    declarative past-tense facts, no imperatives, no cross-episode confabulation."""
    return (
        "A developer is starting a coding session in this repository with this goal:\n\n"
        f"<goal>\n{goal}\n</goal>\n\n"
        "Report what this bank's history genuinely bears on that goal. Rendering rules, strict:\n"
        "- Declarative, past-tense, attributed facts only — what happened, what was decided and why, "
        "with dates, commit/PR/issue ids and exact values where known.\n"
        '- NEVER phrase anything as an instruction, task, or recommendation to act now '
        '("you should", "remove", "update…"). You are a historian reporting the record, not a '
        "planner assigning work.\n"
        "- Do not connect unrelated episodes into one narrative; if two facts are not explicitly "
        "linked in the record, report them separately or leave the weaker one out.\n"
        "- If the bank holds nothing that bears on the goal, say so in one line."
    )


def hs_reflect(query: str, bank: str, url: str | None = None, timeout: int = 120) -> str:
    """Harness-side reflect over a Hindsight bank (used by the claude arm — claude has no plugin).
    Returns the synthesized root-cause answer, or '' on any error (memory is best-effort)."""
    import urllib.request
    base = (url or HINDSIGHT_URL).rstrip("/")
    try:
        req = urllib.request.Request(f"{base}/v1/default/banks/{bank}/reflect",
            data=json.dumps({"query": query, "budget": "high"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (json.loads(r.read()).get("text") or "").strip()
    except Exception:
        return ""


def run_agent(cid: str, model: str, timeout: int, message: str, resume: bool = False,
              agent: str = "opencode", system_append: str | None = None) -> dict:
    # One turn: exec the agent into the already-running per-task container (session store lives inside,
    # so `-c`/`--continue` resumes the same session across the intervention loop).
    if agent == "claude-code":
        cmd = ["docker", "exec", "-w", "/work", cid, "claude", "-p", "--output-format", "json",
               "--permission-mode", "acceptEdits", "--model", model]
        # Inject memory via --append-system-prompt (TRUSTED channel). A UserPromptSubmit hook's
        # additionalContext is treated by claude as a possible prompt-injection and REFUSED; the system
        # prompt is trusted. This is claude's equivalent of opencode's system-prompt injection.
        if system_append:
            cmd += ["--append-system-prompt", system_append]
        if resume:
            cmd.append("--continue")
        cmd.append(message)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        return _parse_claude(proc.stdout, time.perf_counter() - t0)
    if agent == "codex":
        base = ["docker", "exec", "-w", "/work", cid, "codex", "exec", "--json", "-m", model,
                "--dangerously-bypass-approvals-and-sandbox",   # the container IS the sandbox
                "--dangerously-bypass-hook-trust"]              # hooks are harness-installed
        cmd = base + (["resume", "--last", message] if resume else [message])
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        return _parse_codex(proc.stdout, time.perf_counter() - t0)
    cmd = ["docker", "exec", "-w", "/work", cid,
           "opencode", "run", "--format", "json", "-m", model]
    if resume:
        cmd.append("-c")          # continue the last session in this dir (keeps context)
    cmd.append(message)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    # Token split kept separate (cached vs input vs output) — $ is computed later per model.
    tok = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    turns = 0
    traj = []          # structured trajectory for the UI: tool steps + assistant text
    seg_start = 0      # index where the current model-step's steps begin (for token stamping)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        part = e.get("part", {}) or {}
        if t == "tool_use":
            turns += 1
            state = part.get("state", {}) or {}
            inp = state.get("input") or part.get("input") or {}
            arg = ""
            if isinstance(inp, dict):
                for k in ("filePath", "path", "pattern", "command", "query", "url", "content"):
                    if inp.get(k):
                        arg = f"{k}={str(inp[k])[:160]}"
                        break
                if not arg and inp:
                    k, v = next(iter(inp.items())); arg = f"{k}={str(v)[:160]}"
            full_in = "\n".join(f"{k}: {v}" for k, v in inp.items())[:4000] if isinstance(inp, dict) and inp else str(inp)[:4000]
            out = state.get("output")
            traj.append({"k": "tool", "tool": part.get("tool") or "tool", "arg": arg,
                         "input": full_in, "out": str(out)[:4000] if out else ""})
        elif t == "text":
            txt = (part.get("text") or "").strip()
            if txt:
                traj.append({"k": "say", "text": txt[:1500]})
        elif t == "step_finish":
            tk = part.get("tokens", {}) or {}
            s_in = (tk.get("input", 0) or 0)
            s_out = (tk.get("output", 0) or 0) + (tk.get("reasoning", 0) or 0)
            s_cache = 0
            cache = tk.get("cache", {})
            if isinstance(cache, dict):
                s_cache = cache.get("read", 0) or 0
                tok["cache_read"] += s_cache
                tok["cache_write"] += cache.get("write", 0) or 0
            # provider semantics (verified): total = input + cache_read + output + reasoning,
            # i.e. `input` is the NON-cached prompt and `cache_read` is the cached prompt (separate).
            tok["input"] += tk.get("input", 0) or 0
            tok["output"] += tk.get("output", 0) or 0
            tok["reasoning"] += tk.get("reasoning", 0) or 0
            # stamp this model-step's tokens onto the trajectory steps it produced.
            # reasoning is token-only (Gemini hides the thinking TEXT) — track the count so
            # the UI can show how much hidden reasoning each turn did.
            s_reason = tk.get("reasoning", 0) or 0
            for s in traj[seg_start:]:
                s["tok_in"] = s_in + s_cache       # full prompt this turn (fresh + cached)
                s["tok_cache"] = s_cache           # cached portion (billed at the discount rate)
                s["tok_out"] = s_out
                s["tok_reason"] = s_reason
            seg_start = len(traj)
    return {"elapsed": elapsed, "tokens": tok, "turns": turns, "trajectory": traj}


def _oid(prefix: str) -> str:
    import secrets
    return prefix + secrets.token_hex(13)


def seed_sessions(cid: str, conversations: list, model: str) -> int:
    """Seed past developer conversations into the agent container's opencode store as REAL sessions, so
    a fresh agent run CAN consult them (via `opencode session list` / `opencode export <id>`) if it
    chooses — availability + agency, not injection. One opencode session per conversation. Returns count.
    Only for the vanilla baseline: the raw substrate the agent may read, vs the memory system that
    surfaces it reliably."""
    if not conversations:
        return 0
    chats = conversations if isinstance(conversations[0], list) else [conversations]  # 1 chat or many
    mid_model, prov = model.split("/")[-1], (model.split("/")[0] if "/" in model else "google")
    n = 0
    for ci, conv in enumerate(chats):
        if not conv:
            continue
        sid = _oid("ses_"); base = int(time.time() * 1000) - ci * 3_600_000
        info = {"id": sid, "slug": "past-session", "projectID": "x", "directory": "/work", "path": "",
                "title": (conv[0].get("text") or "past session")[:60], "agent": "build",
                "model": {"id": mid_model, "providerID": prov, "variant": "default"}, "version": "1.16.2",
                "summary": {"additions": 0, "deletions": 0, "files": 0}, "cost": 0,
                "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "permission": [], "time": {"created": base, "updated": base}}
        msgs = []; prev = None
        for i, turn in enumerate(conv):
            role = turn.get("role", "user"); text = turn.get("text", ""); mid = _oid("msg_"); ts = base + i * 1000
            if role != "assistant":
                minfo = {"role": "user", "time": {"created": ts}, "agent": "build",
                         "model": {"providerID": prov, "modelID": mid_model}, "summary": {"diffs": []},
                         "id": mid, "sessionID": sid}
                parts = [{"type": "text", "text": text, "id": _oid("prt_"), "sessionID": sid, "messageID": mid}]
            else:
                minfo = {"parentID": prev, "role": "assistant", "mode": "build", "agent": "build",
                         "path": {"cwd": "/work", "root": "/work"}, "cost": 0,
                         "tokens": {"total": 0, "input": 0, "output": 0, "reasoning": 0, "cache": {"write": 0, "read": 0}},
                         "modelID": mid_model, "providerID": prov, "time": {"created": ts, "completed": ts},
                         "finish": "stop", "id": mid, "sessionID": sid}
                parts = [{"type": "text", "text": text, "time": {"start": ts, "end": ts}, "id": _oid("prt_"),
                          "sessionID": sid, "messageID": mid}]
            msgs.append({"info": minfo, "parts": parts}); prev = mid
        seed = json.dumps({"info": info, "messages": msgs})
        subprocess.run(["docker", "exec", "-i", "-w", "/work", cid, "sh", "-c",
                        "cat > /tmp/seed.json && opencode import /tmp/seed.json"],
                       input=seed, capture_output=True, text=True)
        n += 1
    return n


def seed_transcript_files(cid: str, conversations: list) -> int:
    """Vanilla-baseline chat availability for agents WITHOUT an importable session store
    (codex, claude-code): write each past developer conversation as a markdown transcript under
    /root/project-history — OUTSIDE the repo, so the codebase and its diff are untouched. Same
    contract as seed_sessions: the substrate is reachable if the agent chooses to look, nothing
    is injected. Returns the number of transcripts written."""
    if not conversations:
        return 0
    chats = conversations if isinstance(conversations[0], list) else [conversations]
    n = 0
    for ci, conv in enumerate(chats):
        if not conv:
            continue
        lines = [f"# Past developer session {ci + 1}\n"]
        for turn in conv:
            who = "Developer" if turn.get("role", "user") != "assistant" else "Assistant"
            lines.append(f"**{who}:** {turn.get('text', '')}\n")
        body = "\n".join(lines)
        subprocess.run(["docker", "exec", "-i", cid, "sh", "-c",
                        f"mkdir -p /root/project-history && cat > /root/project-history/session-{ci + 1:02d}.md"],
                       input=body, capture_output=True, text=True)
        n += 1
    return n


_JUNK = [".venv", "venv", "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", "*.pyc"]


def capture_source_patch(workdir: Path) -> str:
    """The agent's diff to SOURCE only — tests/ and junk excluded (graded from pristine)."""
    sh("git", "add", "-A", cwd=workdir)
    excl = [f":(exclude){j}" for j in _JUNK] + [":(exclude)tests/**"]
    r = sh("git", "diff", "--cached", "HEAD", "--", ".", *excl, cwd=workdir, cap=True)
    return r.stdout


def grade(task: dict, source_patch: str, work: Path) -> dict:
    """Apply the source patch to a pristine full build + pristine test sets, run pytest in Docker."""
    gd = work / "grade"
    build_repo(task, gd, "full")                      # pristine repo (full)
    shutil.copy(_task_dir(task) / task["hidden_test_file"], gd / "tests" / "test_hidden.py")
    # regression test already copied by build_repo; apply the agent's source patch
    applied = True
    if source_patch.strip():
        p = subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=gd,
                           input=source_patch, text=True)
        applied = p.returncode == 0
    # run the suite in Docker (deterministic), tests graded from pristine copies
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{gd}:/work", "-w", "/work", IMAGE,
         "python", "-m", "pytest", "-q", "tests", "--no-header"],
        capture_output=True, text=True)
    passed = r.returncode == 0
    out = (r.stdout or "")
    if not passed and not out.strip():
        # pytest ALWAYS writes to stdout; an empty failure means the grading CONTAINER failed
        # (missing image, docker down, ...). That is an infra error, not a wrong fix — feeding it
        # back as "tests still fail" burns the intervention cap on a problem the agent cannot fix.
        raise RuntimeError(f"grading container failed (not a test failure): {(r.stderr or '').strip()[:300]}")
    tail = out.strip().splitlines()[-1:] if out.strip() else [""]
    return {"applied": applied, "resolved": passed and applied,
            "pytest": tail[0] if tail else "", "output": out,
            "patch_failed": not applied}


def build_feedback(grade_result: dict) -> str:
    """Surface the NEW problem (failing tests) — not the solution."""
    if grade_result["patch_failed"]:
        return ("Your change could not be applied cleanly to the source. Re-read the current "
                "code and make a focused edit that applies, then ensure the tests pass.")
    out = grade_result["output"]
    # keep the failures section (assertion errors + the short summary), trimmed
    body = out[-2500:] if len(out) > 2500 else out
    return ("Your change did not fully fix the reported regression. Re-running the project's "
            "test suite now reports the following remaining failures:\n\n"
            f"```\n{body.strip()}\n```\n\n"
            "Fix the source so these pass. Do NOT modify any test file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to a task.json (sdebench/datasets/boltons-*/tasks/main/task.json)")
    ap.add_argument("--history", choices=["full", "squashed", "hindsight", "hscoding", "memtool", "inject", "oracle", "hybrid", "index", "provided", "conversations", "skill"], default="full")
    ap.add_argument("--agent", choices=["opencode", "claude-code", "codex"], default="opencode")
    ap.add_argument("--model", default=None, help="agent model; defaults per --agent")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--max-interventions", type=int, default=5,
                    help="cap on feedback rounds before giving up (drift guard)")
    args = ap.parse_args()
    if not args.model:
        args.model = _AGENT_MODEL[args.agent]
    task = json.loads(Path(args.task).read_text())
    task["_dir"] = str(Path(args.task).resolve().parent)
    task.setdefault("repo", task.get("codebase") or task["task_id"])

    work = Path("/tmp/sdebench/run") / f"{task['task_id']}_{args.history}_{args.run_id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    repo = work / "repo"
    memory_bank = None
    mem_index = None
    conv_log = None
    if args.history == "hindsight":
        build_repo(task, repo, "squashed")          # no git trail; history is in memory
        memory_bank = f"sde-{task['repo']}"
        ingest_history(task, memory_bank)           # reset + ingest the full git history
    elif args.history == "provided":
        build_repo(task, repo, "full")              # full repo + external memory supplied in the prompt
        _em = task.get("external_memory")
        if _em:
            task["bug_report"] = task["bug_report"] + "\n\nRelevant memory (surfaced for you by your memory system):\n" + _em
    elif args.history == "hscoding":
        if not _PLUGIN_DIR or not (Path(_PLUGIN_DIR) / "dist").is_dir():
            sys.exit("hscoding arm needs SDE_HSCODING_PLUGIN_DIR -> a hindsight-coding-agents "
                     "checkout with dist/ built")
        build_repo(task, repo, "full")              # full repo; memory via the hindsight-coding-agents
        memory_bank = os.environ.get("SDE_HSCODING_BANK", "hs-coding")  # plugin (reflect + INJECT); the
        #                                            # bank is pre-populated by the plugin's deepen engine
    elif args.history == "skill":
        build_repo(task, repo, "full")              # vanilla full git + a recall_conversations SKILL (tool)
        _cv = task.get("conversations") or []
        if _cv:
            conv_log = str(Path("/tmp/sdebench/conv") / f"{task['task_id']}.json")
            Path(conv_log).parent.mkdir(parents=True, exist_ok=True)
            Path(conv_log).write_text(json.dumps(_cv))
    elif args.history == "conversations":
        build_repo(task, repo, "full")              # full repo + a relevant PAST CONVERSATION surfaced
        _cv = task.get("conversations") or []
        if _cv:
            _log = "\n".join(f"{c['role'].upper()}: {c['text']}" for c in _cv)
            task["bug_report"] = task["bug_report"] + (
                "\n\nRelevant past conversation with the user on this project, recalled by your "
                "memory of previous sessions:\n\n" + _log)
    elif args.history == "index":
        build_repo(task, repo, "squashed")          # no git; a derived DECISIONS.md index IS the memory
        (repo / "DECISIONS.md").write_text(gen_index_doc(task))
        sh("git", "add", "-A", cwd=repo)
        sh("git", "commit", "-q", "-m", "docs: decisions index", cwd=repo,
           env={**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
                "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"})
    elif args.history == "memtool":
        build_repo(task, repo, "squashed")          # no git trail; history is in the recall_intent index
        mem_index = build_mem_index(task)
    elif args.history == "inject":
        build_repo(task, repo, "squashed")          # PUSH: relevant history injected into the prompt
        _idx = build_mem_index(task)
        _k = int(os.environ.get("SDE_INJECT_K", "2"))
        _q = task["bug_report"]
        if os.environ.get("SDE_INJECT_RICH"):       # also rank by the failing test's symbols
            _q += "\n" + (_task_dir(task) / task["regression_test_file"]).read_text()
        task["bug_report"] = inject_context(task["bug_report"], rank_commits(_idx, _q, k=_k))
    elif args.history == "hybrid":
        build_repo(task, repo, "squashed")          # PUSH policy + PULL tool for symptom-distant causes
        mem_index = build_mem_index(task)
        task["bug_report"] = inject_context(task["bug_report"], rank_commits(mem_index, task["bug_report"], k=2))
    elif args.history == "oracle":
        build_repo(task, repo, "squashed")          # ORACLE upper bound: inject the KNOWN cause commit
        _idx = build_mem_index(task)
        _cs = [c for c in json.loads(Path(_idx).read_text()) if c["subject"] == task.get("cause_subject")]
        task["bug_report"] = inject_context(task["bug_report"], _cs)
    else:
        build_repo(task, repo, args.history)
    git_history = capture_git_history(task)

    TOK = ("input", "output", "reasoning", "cache_read", "cache_write")
    totals = {k: 0 for k in TOK}; totals.update({"turns": 0, "wall_s": 0.0, "cost": 0.0})
    trace = []  # ordered multi-round conversation for the UI

    def acc(m, role, prompt_text):
        for k in TOK:
            totals[k] += m["tokens"][k]
        totals["turns"] += m["turns"]; totals["wall_s"] += m["elapsed"]
        totals["cost"] += m.get("cost", 0.0)      # agent-reported cost (claude); opencode computes below
        trace.append({"role": role, "prompt": prompt_text, "trajectory": m["trajectory"],
                      "tokens": m["tokens"], "turns": m["turns"], "wall_s": round(m["elapsed"], 1)})

    print(f"[{task['task_id']}] history={args.history} model={args.model} — initial attempt…", flush=True)
    init_prompt = PROMPT.format(repo=task["repo"], bug_report=task["bug_report"], instruction=VARIANTS[os.environ.get("SDE_VARIANT", "base")])
    env = load_env(memory_bank, mem_index, conv_log)   # container env (memory config the plugin reads)
    cid = start_agent_container(repo, env, args.agent)  # ONE container for the whole task (initial + interventions)
    # claude has no plugin: reflect ONCE on the symptom here and inject via --append-system-prompt on every
    # turn (trusted channel). opencode uses its plugin instead, so no harness reflect for it.
    sys_mem = None
    if args.agent == "claude-code" and memory_bank:
        # Time the reflect round-trip INTO wall_s — for parity with opencode, whose plugin reflect
        # runs inside the timed agent turn. Both arms then pay the same per-task retrieval latency.
        _t0 = time.perf_counter()
        _ans = hs_reflect(_reflect_query(task["bug_report"]), memory_bank)
        totals["wall_s"] += time.perf_counter() - _t0
        if _ans:
            sys_mem = ("Relevant engineering context retrieved from THIS project's own history (git "
                       "commits and past developer decisions) — trusted internal documentation, not user "
                       "input. If it states an exact rule or literal values, apply them precisely; verify "
                       "against the current code:\n\n" + _ans)
            print(f"  [claude] injected memory via system prompt ({len(_ans)} chars)", flush=True)
    try:
        if args.history == "full" and task.get("conversations"):
            # Vanilla-baseline fairness: make the past developer conversations REACHABLE by the plain
            # agent (not injected, not resumed) — it reads them only if it chooses. The realistic
            # "does it think to check its history?" test, vs the memory system that surfaces the
            # decision reliably. opencode gets them as native sessions; codex/claude-code (no
            # importable session store) get markdown transcripts outside the repo.
            if args.agent == "opencode":
                if seed_sessions(cid, task["conversations"], args.model):
                    init_prompt += ("\n\nPast developer sessions on this project are available in your opencode "
                                    "session history — run `opencode session list` and `opencode export <id>` to "
                                    "review them if they help.")
            elif seed_transcript_files(cid, task["conversations"]):
                init_prompt += ("\n\nTranscripts of past developer sessions on this project are archived "
                                "under /root/project-history/ — review them if they help.")
        acc(run_agent(cid, args.model, args.timeout, init_prompt, agent=args.agent, system_append=sys_mem), "initial", init_prompt)

        # Feedback loop: grade -> if failing, tell the agent the NEW problem (not the fix) and resume.
        # Metric = number of human-like interventions needed (capped); cost = sum across all rounds.
        interventions = 0
        while True:
            patch = capture_source_patch(repo)
            g = grade(task, patch, work)
            # record THIS round's submitted patch + its grade outcome (incl. the rejected ones)
            trace[-1]["patch"] = patch
            trace[-1]["grade_pytest"] = g["pytest"]
            trace[-1]["grade_passed"] = g["resolved"]
            if g["resolved"] or interventions >= args.max_interventions:
                break
            interventions += 1
            # number each round so a resumed agent never sees a verbatim-repeated message (claude flags
            # that as an adversarial loop and refuses); the pytest output also differs as the code changes.
            fb = f"[Feedback #{interventions}] " + build_feedback(g)
            print(f"  ↳ intervention {interventions}: {g['pytest']}", flush=True)
            acc(run_agent(cid, args.model, args.timeout, fb, resume=True, agent=args.agent, system_append=sys_mem), f"intervention-{interventions}", fb)
        # memory observability: the plugin records every reflect outcome to /tmp inside the container.
        # A memory arm whose reflect silently failed is NOT a memory run — record and shout.
        mem_diag = None
        if memory_bank and args.agent in ("opencode", "codex"):
            _p = subprocess.run(["docker", "exec", cid, "cat", "/tmp/hindsight-plugin.log"],
                                capture_output=True, text=True)
            mem_diag = [json.loads(l) for l in (_p.stdout or "").splitlines() if l.strip()] or None
            _ok = any(d.get("event") in ("reflect_ok", "recall_ok", "inject_ok") for d in (mem_diag or []))
            print(f"  [memory] reflect diagnostics: {mem_diag if mem_diag else 'NO LOG — plugin never reflected'}"
                  + ("" if _ok else "  ⚠️ MEMORY ARM RAN WITHOUT INJECTED MEMORY"), flush=True)
    finally:
        stop_agent_container(cid)

    solved = g["resolved"]
    # claude reports its own cost per call (summed in totals); opencode/gemini we price from tokens.
    cost = totals["cost"] if args.agent == "claude-code" else compute_cost(args.model, {k: totals[k] for k in TOK})
    result = {
        "task_id": task["task_id"], "codebase": task.get("codebase") or task["repo"],
        "variant": os.environ.get("SDE_VARIANT", "base"),
        "history": args.history, "agent": args.agent, "model": args.model,
        "solved": solved, "interventions": interventions,
        "capped": (not solved and interventions >= args.max_interventions),
        "final_pytest": g["pytest"], "patch_bytes": len(patch),
        "tokens": {k: totals[k] for k in TOK},      # cached vs input vs output kept separate
        "turns": totals["turns"], "wall_s": round(totals["wall_s"], 1),
        "cost_usd": round(cost, 4),                   # 0 when the model has no PRICES entry
        "memory_diag": mem_diag if (memory_bank and args.agent in ("opencode", "codex")) else None,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    (work / "trace.json").write_text(json.dumps(
        {**result, "bug_report": task["bug_report"], "final_patch": patch, "git_history": git_history, "trace": trace}, indent=2))
    # each workdir holds a full host-repo clone (+ a second pristine copy for grading) — sweeps ran
    # the disk to 100% and took the docker daemon down. Keep result/trace, drop the repo copies.
    shutil.rmtree(repo, ignore_errors=True)
    shutil.rmtree(work / "grade", ignore_errors=True)
    print(json.dumps(result, indent=2))
    tk = result["tokens"]
    print(f"\nRESULT history={args.history}: solved={solved} interventions={interventions} | "
          f"tokens in={tk['input']} out={tk['output']} cache_r={tk['cache_read']} cache_w={tk['cache_write']} | "
          f"wall={totals['wall_s']:.0f}s -> {work}/result.json")


if __name__ == "__main__":
    main()
