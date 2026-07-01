"""HARD H-source task: REGRESS a real boltons fix (commit c463d163, 'Fix OrderedMultiDict equality to
compare values against plain mappings'). The bug — OMD.__eq__ discards the value comparison, so an OMD
== a same-keyed dict regardless of values — is DUPLICATED in dictutils.OrderedMultiDict AND
urlutils.OrderedMultiDict (backing URL.query_params). The symptom is a URL query comparison; the cause
is buried in __eq__, in two places. The decision + the 'it's duplicated in urlutils too' note live in
REAL git history (commit c463d163)."""
import sys, os, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"
KEEP = {"conftest.py", "__init__.py", "test_dictutils.py", "test_urlutils.py"}
FIXED = "                    if other[selfk] != self[selfk]:\n                        return False"
BUGGY = "                    other[selfk] == self[selfk]"


def main():
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    p = out / "boltons/dictutils.py"   # the functional OMD.__eq__ (urlutils re-exports this same class)
    assert FIXED in p.read_text(), "anchor not found"
    p.write_text(p.read_text().replace(FIXED, BUGGY, 1))
    import re
    for p in (out / "tests").glob("test_*.py"):
        if p.name not in KEEP:
            p.unlink()
    # hold out the guarding tests c463d163 added (else red at HEAD + reveal the answer)
    td = out / "tests/test_dictutils.py"
    td.write_text(re.sub(r"\ndef test_eq_with_dict\(\):.*?(?=\ndef |\Z)", "\n", td.read_text(), flags=re.S))
    tu = out / "tests/test_urlutils.py"
    tu.write_text(re.sub(r"\ndef test_query_param_dict_eq_with_dict\(\):.*?(?=\ndef |\Z)", "\n", tu.read_text(), flags=re.S))
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", "refactor: streamline OrderedMultiDict.__eq__ mapping branch"], env=env, check=True)
    print("built boltons-omdeq (H, hard) @ " + REF[:8])


if __name__ == "__main__":
    main()
