"""Emit real-function host tasks (boltons-<name>) from realfn_traps.py, and validate discrimination
against boltons' REAL test suite."""
import json, sys, subprocess, tempfile, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from realfn_traps import REALFN_TRAPS

DATASETS = HERE.parents[0] / "datasets"
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"

BUILD = '''import sys, os, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = {ref!r}
KEEP = set({keep!r}) | {{"conftest.py", "__init__.py"}}


def main():
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    for p in (out / "tests").glob("test_*.py"):       # pass_to_pass = the real tests for this module
        if p.name not in KEEP:
            p.unlink()
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {{**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "chore: focus test suite"], env=env, check=True)
    print("built boltons-{name} @ " + REF[:8])


if __name__ == "__main__":
    main()
'''


def emit(name):
    t = REALFN_TRAPS[name]
    cb = f"boltons-{name}"
    ds = DATASETS / cb / "tasks" / "main"
    ds.mkdir(parents=True, exist_ok=True)
    (DATASETS / cb / "build.py").write_text(BUILD.format(ref=REF, keep=t["keep_tests"], name=name))
    (ds / "regression_test.py").write_text(t["repro_test"])
    (ds / "hidden_test.py").write_text(t["hidden_test"])
    task = {
        "task_id": f"{cb}-001", "codebase": cb, "build": "build.py", "source": "F", "tier": "real-function",
        "module": t["module"], "bug_report": t["bug_report"],
        "fail_to_pass": ["tests/test_regression.py"], "pass_to_pass": ["tests/" + k for k in t["keep_tests"]],
        "hidden_to_pass": ["tests/test_hidden.py"],
        "regression_test_file": "regression_test.py", "hidden_test_file": "hidden_test.py",
        "conversations": t["conversation"],
    }
    (ds / "task.json").write_text(json.dumps(task, indent=2))
    return cb


def validate(name):
    t = REALFN_TRAPS[name]
    out = Path(tempfile.mkdtemp(prefix=f"rf_{name}_"))
    try:
        subprocess.run(["python", str(DATASETS / f"boltons-{name}" / "build.py"), str(out)], check=True, capture_output=True)
        shutil.copy(DATASETS / f"boltons-{name}" / "tasks/main/regression_test.py", out / "tests/test_regression.py")
        shutil.copy(DATASETS / f"boltons-{name}" / "tasks/main/hidden_test.py", out / "tests/test_hidden.py")
        mod = out / t["module"]
        real = mod.read_text()
        keep = t["keep_tests"][0]

        def run(tp):
            return subprocess.run(["uv", "run", "--no-project", "--with", "pytest", "python", "-m", "pytest", "-q", "--no-header", tp],
                                  cwd=str(out), capture_output=True, text=True).returncode == 0
        res = {}
        res["HEAD"] = (run(f"tests/{keep}"), run("tests/test_regression.py"), run("tests/test_hidden.py"))
        mod.write_text(real.replace(t["anchor"], t["correct"], 1))
        res["CORRECT"] = (run(f"tests/{keep}"), run("tests/test_regression.py"), run("tests/test_hidden.py"))
        mod.write_text(real.replace(t["anchor"], t["naive"], 1))
        res["NAIVE"] = (run(f"tests/{keep}"), run("tests/test_regression.py"), run("tests/test_hidden.py"))
        return res
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    for n in REALFN_TRAPS:
        emit(n)
    print("emitted:", list(REALFN_TRAPS))
    print("\n=== discrimination (realtests, repro, hidden) — want HEAD T/F/F, CORRECT T/T/T, NAIVE T/T/F ===")
    ok = True
    for n in REALFN_TRAPS:
        r = validate(n)
        good = (r["HEAD"] == (True, False, False) and r["CORRECT"] == (True, True, True) and r["NAIVE"] == (True, True, False))
        ok = ok and good
        print(f"  {'OK ' if good else 'BAD'} {n:14s} HEAD={r['HEAD']} CORRECT={r['CORRECT']} NAIVE={r['NAIVE']}")
    print("ALL GOOD" if ok else "SOME BAD")
