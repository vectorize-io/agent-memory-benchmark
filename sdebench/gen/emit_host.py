"""Emit boltons-hosted tasks: plant a validated generator trap as a module INTO the real boltons
repo (real history = retrieval noise; the agent navigates a real codebase). F source: the decision
lives in the realistic chat (gen/sessions.json), summarized by memsys."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from traps import TRAPS

DATASETS = HERE.parents[0] / "datasets"
SESSIONS = json.loads((HERE / "sessions.json").read_text())
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"

BUILD = '''import sys, os, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen"))
from traps import TRAPS

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = {ref!r}
TRAP = TRAPS[{trap!r}]


def main():
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    pkg = TRAP["pkg"]
    (out / pkg).mkdir(exist_ok=True)
    (out / pkg / "__init__.py").write_text(TRAP["init"])
    mp = out / TRAP["module"]; mp.parent.mkdir(parents=True, exist_ok=True); mp.write_text(TRAP["bug"])
    for p in (out / "tests").glob("test_*.py"):       # focus grading on the planted module
        p.unlink()
    (out / "tests" / ("test_" + {trap!r} + ".py")).write_text(TRAP["existing_test"])
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {{**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "feat: add " + pkg + " module"], env=env, check=True)
    print("planted " + {trap!r} + " in boltons")


if __name__ == "__main__":
    main()
'''


def emit(trap_name):
    trap = TRAPS[trap_name]
    cb = f"boltons-{trap_name}"
    ds = DATASETS / cb / "tasks" / "main"
    ds.mkdir(parents=True, exist_ok=True)
    (DATASETS / cb / "build.py").write_text(BUILD.format(ref=REF, trap=trap_name))
    (ds / "regression_test.py").write_text(trap["repro_test"])
    (ds / "hidden_test.py").write_text(trap["hidden_test"])
    task = {
        "task_id": f"{cb}-001", "codebase": cb, "build": "build.py", "source": "F",
        "bug_report": trap["bug_report"],
        "fail_to_pass": ["tests/test_regression.py"],
        "pass_to_pass": [f"tests/test_{trap_name}.py"],
        "hidden_to_pass": ["tests/test_hidden.py"],
        "regression_test_file": "regression_test.py", "hidden_test_file": "hidden_test.py",
        "conversations": SESSIONS.get(trap_name, trap["conversation"]),
    }
    (ds / "task.json").write_text(json.dumps(task, indent=2))
    return cb


for n in ("rounding", "listmerge", "budget"):
    print("emitted host task:", emit(n))
