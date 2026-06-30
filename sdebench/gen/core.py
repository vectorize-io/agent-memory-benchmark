"""Generator core: plant a trap into a source (H / K / F) as a full-history git repo.

build(trap, source, out)  -> a buggy-at-HEAD repo, with the decision stored in the chosen source:
  H : the correct code + decision lived in git history, then a regression dropped it.
  F : the code was always buggy; the decision lives ONLY in task.conversations (not in the repo).
task_spec(trap, source) -> (task.json dict, regression_test, hidden_test) for the harness.
"""
import json, os, subprocess
from pathlib import Path

PKG_DIR = {"pay": "pay", "confmerge": "confmerge"}


def _sh(*a, cwd=None, env=None):
    subprocess.run(a, cwd=cwd, env=env, check=True, capture_output=True)


def _commit(out, msg, day, author="Sam Dev"):
    d = f"2024-0{1 + day // 28}-{1 + day % 28:02d}T10:00:00"
    env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
           "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "d@e.com",
           "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "d@e.com"}
    _sh("git", "add", "-A", cwd=out)
    _sh("git", "commit", "-q", "-m", msg, cwd=out, env=env)


def build(trap, source, out):
    out = Path(out)
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=True)
    out.mkdir(parents=True)
    _sh("git", "init", "-q", cwd=out)
    _sh("git", "branch", "-M", "main", cwd=out)
    pkg = trap["pkg"]

    def w(path, content):
        p = out / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    day = [0]

    def commit(msg, author="Sam Dev"):
        _commit(out, msg, day[0], author); day[0] += 1

    w("pyproject.toml", f'[project]\nname = "{pkg}"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    w("README.md", f"# {pkg}\n\nA small library.\n")
    w(f"{pkg}/__init__.py", f'"""{pkg} package."""\n')
    commit("scaffold project")

    if source == "H":
        # the correct policy + its rationale lived in git history...
        w(trap["module"], trap["correct"])
        w(f"{pkg}/__init__.py", trap["init"])
        commit(trap["decision_subject"] + "\n\n" + trap["decision_rationale"], author="Priya N.")
        w("tests/test_existing.py", trap["existing_test"])
        commit("tests")
        w("README.md", f"# {pkg}\n\nA small library.\n\nSee tests/ for usage.\n")
        commit("readme")
        # ...then a regression dropped it (HEAD is buggy).
        w(trap["module"], trap["bug"])
        commit(f"refactor: simplify {Path(trap['module']).stem}", author="Priya N.")
    else:  # F: the code was always buggy; the decision lives ONLY in task.conversations
        w(trap["module"], trap["bug"])
        w(f"{pkg}/__init__.py", trap["init"])
        commit(f"feat: {Path(trap['module']).stem}")
        w("tests/test_existing.py", trap["existing_test"])
        commit("tests")
        w("README.md", f"# {pkg}\n\nA small library.\n\nSee tests/ for usage.\n")
        commit("readme")

    # noise + release
    w("CHANGELOG.md", "# Changelog\n\n## 0.2.0\n- library\n")
    commit("changelog 0.2.0")
    w("pyproject.toml", f'[project]\nname = "{pkg}"\nversion = "0.2.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.2.0")
    return out


def task_spec(trap, source, codebase):
    """Return (task.json dict, regression_test_src, hidden_test_src)."""
    task = {
        "task_id": f"gen-{trap['name']}-{source}-001",
        "codebase": codebase,
        "build": "build.py",
        "source": source,
        "cause_subject": f"refactor: simplify {Path(trap['module']).stem}",
        "bug_report": trap["bug_report"],
        "fail_to_pass": ["tests/test_regression.py"],
        "pass_to_pass": ["tests/test_existing.py"],
        "hidden_to_pass": ["tests/test_hidden.py"],
        "regression_test_file": "regression_test.py",
        "hidden_test_file": "hidden_test.py",
    }
    if source == "F":
        task["conversations"] = trap["conversation"]
    return task, trap["repro_test"], trap["hidden_test"]
