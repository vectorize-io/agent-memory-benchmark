"""sdebench harness — run a coding agent on a regression task and grade it.

Flow: build the repo (full or squashed history) -> ship the agent the bug report +
failing regression test -> run opencode -> capture the SOURCE diff (tests excluded) ->
grade in Docker against FAIL_TO_PASS + PASS_TO_PASS + HIDDEN_TO_PASS from pristine copies.

Usage:
    uv run python sdebench/harness/run.py --history full      [--model google/gemini-3.5-flash]
    uv run python sdebench/harness/run.py --history squashed

Metrics reported: resolution (binary), cost (tokens; $ if --price set), speed (wall, turns).
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
SDEBENCH = HARNESS.parent
REPO_ROOT = SDEBENCH.parent
IMAGE = "sdebench-base"

PROMPT = """\
You are a maintainer of the `{repo}` Python project. A regression was reported:

{bug_report}

Fix the bug in the source code. Do NOT modify any test files — the graders supply their own.
Save your changes to disk before finishing.
"""


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
    ds = SDEBENCH / "datasets" / task["repo"]
    sh("python", str(ds / task["build"]), str(dest))
    if history == "squashed":
        shutil.rmtree(dest / ".git")
        sh("git", "init", "-q", cwd=dest)
        sh("git", "add", "-A", cwd=dest)
        env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
               "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
        sh("git", "commit", "-q", "-m", "Initial commit", cwd=dest, env=env)
    # ship the failing regression repro (the agent sees it; it is red)
    ds_test = ds / task["regression_test_file"]
    shutil.copy(ds_test, dest / "tests" / "test_regression.py")
    sh("git", "add", "-A", cwd=dest)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
           "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    sh("git", "commit", "-q", "-m", "test: add failing repro for the reported regression",
       cwd=dest, env=env)


def load_env() -> dict:
    env = os.environ.copy()
    ef = REPO_ROOT / ".env"
    if ef.exists():
        for line in ef.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    env["HINDSIGHT_DISABLED"] = "1"   # plain agent: no memory/plugins, just git via bash
    env["PWD"] = ""                   # set per-run
    env["HOME"] = neutral_home()
    return env


def run_agent(workdir: Path, task: dict, model: str, timeout: int) -> dict:
    env = load_env()
    env["PWD"] = str(workdir)
    prompt = PROMPT.format(repo=task["repo"], bug_report=task["bug_report"])
    t0 = time.perf_counter()
    proc = subprocess.run(["opencode", "run", "--format", "json", "-m", model, prompt],
                          cwd=str(workdir), env=env, timeout=timeout,
                          capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    out_tok = in_tok = turns = 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "tool_use":
            turns += 1
        elif e.get("type") == "step_finish":
            tk = e.get("part", {}).get("tokens", {}) or {}
            out_tok += (tk.get("output", 0) or 0) + (tk.get("reasoning", 0) or 0)
            in_tok += tk.get("input", 0) or 0
    return {"elapsed": elapsed, "out_tokens": out_tok, "in_tokens": in_tok, "turns": turns}


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
    ds = SDEBENCH / "datasets" / task["repo"]
    shutil.copy(ds / task["hidden_test_file"], gd / "tests" / "test_hidden.py")
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
    tail = (r.stdout or "").strip().splitlines()[-1:] if r.stdout else [""]
    return {"applied": applied, "resolved": passed and applied, "pytest": tail[0] if tail else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(SDEBENCH / "datasets" / "ratelimiter" / "task.json"))
    ap.add_argument("--history", choices=["full", "squashed"], default="full")
    ap.add_argument("--model", default="google/gemini-3.5-flash")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--price-in", type=float, default=0.0, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="$ per 1M output tokens")
    ap.add_argument("--run-id", default="r1")
    args = ap.parse_args()
    task = json.loads(Path(args.task).read_text())

    work = Path("/tmp/sdebench/run") / f"{task['task_id']}_{args.history}_{args.run_id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    repo = work / "repo"
    build_repo(task, repo, args.history)
    print(f"[{task['task_id']}] history={args.history} model={args.model} — running agent…", flush=True)
    m = run_agent(repo, task, args.model, args.timeout)
    patch = capture_source_patch(repo)
    g = grade(task, patch, work)
    cost = (m["in_tokens"] * args.price_in + m["out_tokens"] * args.price_out) / 1_000_000
    result = {
        "task_id": task["task_id"], "history": args.history, "model": args.model,
        "resolved": g["resolved"], "applied": g["applied"], "pytest": g["pytest"],
        "patch_bytes": len(patch), "turns": m["turns"],
        "out_tokens": m["out_tokens"], "in_tokens": m["in_tokens"],
        "wall_s": round(m["elapsed"], 1), "cost_usd": round(cost, 4),
    }
    out = work / "result.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nRESULT history={args.history}: resolved={g['resolved']} "
          f"turns={m['turns']} out_tok={m['out_tokens']} wall={m['elapsed']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
