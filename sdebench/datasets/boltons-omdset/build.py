"""HARD H-source task (fork + shabby commit). OrderedMultiDict keeps THREE structures in sync: the
underlying dict (key -> value list, used by getlist), self._map (key -> cells), and the self.root
linked list (iteration order). _remove_all() clears the linked list + _map but deliberately leaves
the dict's value list in place — so __setitem__ must reset it to [v]. A 'shabby' perf commit rewrites
__setitem__ to reuse add(), which APPENDS to that stale list: getlist(k) then returns the old values
+ the new one, while __getitem__/todict/iteration still look correct. The symptom shows up in URL
query params, two modules from the cause. All 153 real tests pass (untested edge). The invariant is
documented in an earlier commit (in git history); a naive reader of __setitem__ won't see why."""
import sys, os, shutil, subprocess
from pathlib import Path

HOST = Path.home() / "dev" / "_sdebench_hosts" / "boltons"
REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"

REAL = '''    def __setitem__(self, k, v):
        if super().__contains__(k):
            self._remove_all(k)
        self._insert(k, v)
        super().__setitem__(k, [v])'''
DOCUMENTED = '''    def __setitem__(self, k, v):
        if super().__contains__(k):
            self._remove_all(k)
        # _remove_all() clears the linked list and self._map for k, but deliberately leaves the
        # underlying dict's value list untouched, so reset it to exactly [v] here. Do NOT reuse
        # add() to do this: add() appends to that stale value list, leaving getlist(k) returning
        # the old values alongside the new one (while __getitem__/iteration still look correct).
        self._insert(k, v)
        super().__setitem__(k, [v])'''
SHABBY = '''    def __setitem__(self, k, v):
        if super().__contains__(k):
            self._remove_all(k)
        self.add(k, v)'''


def commit(out, msg):
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Dana Ops", "GIT_AUTHOR_EMAIL": "d@e.com",
           "GIT_COMMITTER_NAME": "Dana Ops", "GIT_COMMITTER_EMAIL": "d@e.com"}
    subprocess.run(["git", "-C", str(out), "commit", "-q", "-m", msg], env=env, check=True)


def main():
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(HOST, out, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.run(["git", "-C", str(out), "checkout", "-q", REF], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "-B", "main"], check=True)
    du = out / "boltons/dictutils.py"
    assert REAL in du.read_text()
    # 1) good commit: document the invariant (the "why")
    du.write_text(du.read_text().replace(REAL, DOCUMENTED, 1))
    commit(out, "dictutils: document why OrderedMultiDict.__setitem__ resets the value list\n\n"
                "_remove_all() clears the linked list and _map but leaves the underlying dict's value "
                "list intact by design, so __setitem__ resets it to [v] explicitly. Reusing add() here "
                "would append to the stale list and make getlist(k) return the old values too — a bug "
                "that __getitem__/iteration/todict all hide because they read the last value. Keep the "
                "explicit reset.")
    # noise (a plausible unrelated commit between them)
    (out / "CHANGELOG.md").write_text((out / "CHANGELOG.md").read_text() + "\n- internal cleanups\n") if (out / "CHANGELOG.md").exists() else None
    commit(out, "docs: changelog housekeeping")
    # 2) shabby commit: the regression, with a misleading message
    du.write_text(du.read_text().replace(DOCUMENTED, SHABBY, 1))
    commit(out, "perf: streamline OrderedMultiDict.__setitem__")
    print("built boltons-omdset (H, hard) @ " + REF[:8])


if __name__ == "__main__":
    main()
