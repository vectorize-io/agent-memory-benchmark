import sys, os, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen"))
from traps import TRAPS

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = '979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe'
TRAP = TRAPS['rounding']


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
    (out / "tests" / ("test_" + 'rounding' + ".py")).write_text(TRAP["existing_test"])
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "feat: add " + pkg + " module"], env=env, check=True)
    print("planted " + 'rounding' + " in boltons")


if __name__ == "__main__":
    main()
