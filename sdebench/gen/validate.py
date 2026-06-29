"""Auto-validator: prove a generated (trap, source) task actually discriminates.

For each task it checks:
  1. existing tests GREEN at HEAD          (the regression slipped past them)
  2. repro RED at HEAD                      (the bug is real)
  3. hidden RED at HEAD                     (the policy is violated)
  4. the CORRECT fix -> all GREEN
  5. every NAIVE guess -> repro PASSES but hidden FAILS   (non-guessable: can't bluff past)
  6. source isolation                       (the answer-marker lives only in the chosen source)

Usage: uv run python sdebench/gen/validate.py            # validate the whole matrix
"""
import subprocess, sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traps import TRAPS
from core import build, task_spec


def _pytest_ok(out, path):
    r = subprocess.run(["uv", "run", "--with", "pytest", "python", "-m", "pytest", "-q", path],
                       cwd=str(out), capture_output=True, text=True)
    return r.returncode == 0


def _write_tests(out, trap):
    (out / "tests" / "test_regression.py").write_text(trap["repro_test"])
    (out / "tests" / "test_hidden.py").write_text(trap["hidden_test"])


def validate(trap, source):
    out = Path(tempfile.mkdtemp(prefix=f"gen_{trap['name']}_{source}_"))
    try:
        build(trap, source, out)
        _write_tests(out, trap)
        mod = out / trap["module"]
        chk = {}
        # at HEAD (bug)
        chk["existing_green@HEAD"] = _pytest_ok(out, "tests/test_existing.py")
        chk["repro_red@HEAD"] = not _pytest_ok(out, "tests/test_regression.py")
        chk["hidden_red@HEAD"] = not _pytest_ok(out, "tests/test_hidden.py")
        # correct fix
        mod.write_text(trap["correct"])
        chk["correct_all_green"] = _pytest_ok(out, "tests")
        # naive guesses must pass repro but fail hidden
        for i, nf in enumerate(trap["naive"]):
            mod.write_text(nf)
            chk[f"naive{i}_repro_pass"] = _pytest_ok(out, "tests/test_regression.py")
            chk[f"naive{i}_hidden_fail"] = not _pytest_ok(out, "tests/test_hidden.py")
        # source isolation (marker = the answer-token)
        mod.write_text(trap["bug"])  # restore HEAD
        marker = trap["marker"].lower()
        gitlog = subprocess.run(["git", "-C", str(out), "log", "-p"], capture_output=True, text=True).stdout.lower()
        head_files = " ".join(p.read_text(errors="ignore") for p in out.rglob("*")
                              if p.is_file() and p.suffix == ".py" and "__pycache__" not in p.parts
                              and ".git" not in p.parts and "tests" not in p.parts).lower()
        if source == "H":
            chk["isolation_H"] = (marker in gitlog) and (marker not in (mod.read_text().lower()))
        elif source == "K":
            chk["isolation_K"] = marker in (out / "CONVENTIONS.md").read_text().lower()
        elif source == "F":
            chk["isolation_F"] = (marker not in gitlog) and (marker not in head_files)
        return chk
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    sources = ["H", "K", "F"]
    allok = True
    for name, trap in TRAPS.items():
        for src in sources:
            chk = validate(trap, src)
            ok = all(chk.values())
            allok = allok and ok
            flag = "PASS" if ok else "FAIL"
            print(f"[{flag}] gen-{name}-{src}")
            if not ok:
                for k, v in chk.items():
                    if not v:
                        print(f"        x {k}")
    print("\nALL VALID" if allok else "\nSOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
