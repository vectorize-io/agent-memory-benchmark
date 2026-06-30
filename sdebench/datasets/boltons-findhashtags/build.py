import sys, os, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = '979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe'
KEEP = set(['test_strutils.py']) | {"conftest.py", "__init__.py"}


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
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "chore: focus test suite"], env=env, check=True)
    print("built boltons-findhashtags @ " + REF[:8])


if __name__ == "__main__":
    main()
