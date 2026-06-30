"""Host task: real boltons at a frozen ref. boltons.strutils.slugify strips punctuation symbols
('R&D' -> 'r-d') — the 'bug' per the team's SEO symbol-expansion policy, which lives only in a
past chat (F source). pass_to_pass = boltons' real tests/test_strutils.py."""
import sys, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"
KEEP = {"conftest.py", "__init__.py", "test_strutils.py"}


def main():
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns('.venv', '__pycache__', '.pytest_cache', '*.pyc'))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    for p in (out / "tests").glob("test_*.py"):          # focus grading on the relevant real suite
        if p.name not in KEEP:
            p.unlink()
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {"GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    import os
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "chore: focus test suite"], env={**os.environ, **env}, check=True)
    print(f"built boltons-slugify @ {REF[:8]}")


if __name__ == "__main__":
    main()
