"""sdebench harness — run a coding agent on a regression task and grade it.

Flow: build the repo (full or squashed history) -> ship the agent the bug report +
failing regression test -> run opencode -> capture the SOURCE diff (tests excluded) ->
grade in Docker against FAIL_TO_PASS + PASS_TO_PASS + HIDDEN_TO_PASS from pristine copies.

Usage:
    uv run python sdebench/harness/run.py --history full      [--model google/gemini-3.5-flash]
    uv run python sdebench/harness/run.py --history squashed

Metrics reported: resolution (binary), cost (tokens; $ if --price set), speed (wall, turns).
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
                          capture_output=cap, text=True)


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


HINDSIGHT_URL = "http://localhost:8888"


def load_env(memory_bank: str | None = None, mem_index: str | None = None) -> dict:
    env = os.environ.copy()
    ef = REPO_ROOT / ".env"
    if ef.exists():
        for line in ef.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if mem_index:     # enable the LOCAL recall_intent tool over the raw-commit index
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
    for sha in sh("git", "-C", str(src), "rev-list", "HEAD", cap=True).stdout.split():
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


def run_agent(workdir: Path, model: str, timeout: int, message: str, resume: bool = False,
              memory_bank: str | None = None, mem_index: str | None = None) -> dict:
    env = load_env(memory_bank, mem_index)
    env["PWD"] = str(workdir)
    cmd = ["opencode", "run", "--format", "json", "-m", model]
    if resume:
        cmd.append("-c")          # continue the last session in this dir (keeps context)
    cmd.append(message)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(workdir), env=env, timeout=timeout,
                          capture_output=True, text=True)
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
    ap.add_argument("--task", default=str(SDEBENCH / "datasets" / "ratelimiter" / "task.json"))
    ap.add_argument("--history", choices=["full", "squashed", "hindsight", "memtool", "inject", "oracle", "hybrid", "index", "provided"], default="full")
    ap.add_argument("--model", default="google/gemini-3.5-flash")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--max-interventions", type=int, default=5,
                    help="cap on feedback rounds before giving up (drift guard)")
    args = ap.parse_args()
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
    if args.history == "hindsight":
        build_repo(task, repo, "squashed")          # no git trail; history is in memory
        memory_bank = f"sde-{task['repo']}"
        ingest_history(task, memory_bank)           # reset + ingest the full git history
    elif args.history == "provided":
        build_repo(task, repo, "full")              # full repo + external memory supplied in the prompt
        _em = task.get("external_memory")
        if _em:
            task["bug_report"] = task["bug_report"] + "\n\nRelevant memory (surfaced for you by your memory system):\n" + _em
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
    totals = {k: 0 for k in TOK}; totals.update({"turns": 0, "wall_s": 0.0})
    trace = []  # ordered multi-round conversation for the UI

    def acc(m, role, prompt_text):
        for k in TOK:
            totals[k] += m["tokens"][k]
        totals["turns"] += m["turns"]; totals["wall_s"] += m["elapsed"]
        trace.append({"role": role, "prompt": prompt_text, "trajectory": m["trajectory"],
                      "tokens": m["tokens"], "turns": m["turns"], "wall_s": round(m["elapsed"], 1)})

    print(f"[{task['task_id']}] history={args.history} model={args.model} — initial attempt…", flush=True)
    init_prompt = PROMPT.format(repo=task["repo"], bug_report=task["bug_report"], instruction=VARIANTS[os.environ.get("SDE_VARIANT", "base")])
    acc(run_agent(repo, args.model, args.timeout, init_prompt, memory_bank=memory_bank, mem_index=mem_index), "initial", init_prompt)

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
        fb = build_feedback(g)
        print(f"  ↳ intervention {interventions}: {g['pytest']}", flush=True)
        acc(run_agent(repo, args.model, args.timeout, fb, resume=True, memory_bank=memory_bank, mem_index=mem_index), f"intervention-{interventions}", fb)

    solved = g["resolved"]
    cost = compute_cost(args.model, {k: totals[k] for k in TOK})
    result = {
        "task_id": task["task_id"], "codebase": task.get("codebase") or task["repo"],
        "variant": os.environ.get("SDE_VARIANT", "base"),
        "history": args.history, "model": args.model,
        "solved": solved, "interventions": interventions,
        "capped": (not solved and interventions >= args.max_interventions),
        "final_pytest": g["pytest"], "patch_bytes": len(patch),
        "tokens": {k: totals[k] for k in TOK},      # cached vs input vs output kept separate
        "turns": totals["turns"], "wall_s": round(totals["wall_s"], 1),
        "cost_usd": round(cost, 4),                   # 0 unless --price-* given
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    (work / "trace.json").write_text(json.dumps(
        {**result, "bug_report": task["bug_report"], "final_patch": patch, "git_history": git_history, "trace": trace}, indent=2))
    print(json.dumps(result, indent=2))
    tk = result["tokens"]
    print(f"\nRESULT history={args.history}: solved={solved} interventions={interventions} | "
          f"tokens in={tk['input']} out={tk['output']} cache_r={tk['cache_read']} cache_w={tk['cache_write']} | "
          f"wall={totals['wall_s']:.0f}s -> {work}/result.json")


if __name__ == "__main__":
    main()
