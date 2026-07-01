"""H-source host task: REGRESS a real boltons fix (commit 766b5547, 'bytes2human rolls over at exact
powers of 1024', which changed the unit-selection boundary from <= to <). The decision + rationale
live in REAL git history (that commit stays in the log). The agent must restore the general fix; the
guarding test is held out (removed from the shipped suite) so it can't just be read off."""
import sys, os, re, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"
KEEP = {"conftest.py", "__init__.py", "test_strutils.py"}


def main():
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    # regression: revert the exact-power fix (< back to <=)
    su = out / "boltons/strutils.py"
    su.write_text(su.read_text().replace("if abs_bytes < next_size:", "if abs_bytes <= next_size:", 1))
    # hold out the guarding test: remove def test_bytes2human from the shipped suite (else it reveals
    # the answer and is red at HEAD). Its behavior lives in the hidden test.
    for p in (out / "tests").glob("test_*.py"):
        if p.name not in KEEP:
            p.unlink()
    ts = out / "tests/test_strutils.py"
    ts.write_text(re.sub(r"\ndef test_bytes2human\(\):.*?(?=\ndef |\Z)", "\n", ts.read_text(), flags=re.S))
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "refactor: simplify bytes2human boundary check"], env=env, check=True)
    print("built boltons-bytes2human (H) @ " + REF[:8])


if __name__ == "__main__":
    main()
