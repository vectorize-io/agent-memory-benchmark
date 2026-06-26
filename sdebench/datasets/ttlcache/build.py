"""Build the `ttlcache` synthetic repo — a HISTORY-DEPENDENT regression.

A refactor ("centralize cache constants; add clear()") changed DEFAULT_TTL from 300 to
600 AND dropped the comment explaining *why* it was 300 (matches the upstream auth token
lifetime). The regression repro only proves entries expire too LATE — it cannot reveal the
correct value. The hidden tests pin DEFAULT_TTL to EXACTLY 300. The number 300 now lives
ONLY in git history (the original commit's message + the pre-refactor file), so:
  - with history: `git blame`/`log` the constant -> 300 -> fixed in one shot.
  - without history: the agent must binary-search the value via feedback rounds.
The refactor also bundles a legit `clear()` (with its own test), so a lazy `git revert`
fails PASS_TO_PASS -> forces a surgical one-line fix.

Usage: python build.py <output_dir>   (default: /tmp/sdebench/ttlcache)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ttlcache")

CACHE_V1 = '''\
"""A tiny in-memory cache with per-entry time-to-live."""

# Entries expire after DEFAULT_TTL seconds. 287 is the measured p99 auth-token refresh
# interval — keep these in sync, or cached values can outlive the tokens they were
# fetched with. Do NOT change without updating the auth layer.
DEFAULT_TTL = 287


class TTLCache:
    """A cache whose entries expire DEFAULT_TTL seconds after they are set.

    Time is passed in explicitly (`now`, seconds) so behaviour is deterministic.
    """

    def __init__(self, ttl=DEFAULT_TTL):
        self.ttl = ttl
        self._store = {}

    def set(self, key, value, now):
        self._store[key] = (value, now + self.ttl)

    def get(self, key, now):
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if now >= expires_at:
            del self._store[key]
            return None
        return value
'''

# C6 refactor: DEFAULT_TTL 300 -> 600 (REGRESSION), rationale comment dropped, + clear()
CACHE_V2 = '''\
"""A tiny in-memory cache with per-entry time-to-live."""

# Default entry lifetime, in seconds.
DEFAULT_TTL = 600


class TTLCache:
    """A cache whose entries expire DEFAULT_TTL seconds after they are set.

    Time is passed in explicitly (`now`, seconds) so behaviour is deterministic.
    """

    def __init__(self, ttl=DEFAULT_TTL):
        self.ttl = ttl
        self._store = {}

    def set(self, key, value, now):
        self._store[key] = (value, now + self.ttl)

    def get(self, key, now):
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if now >= expires_at:
            del self._store[key]
            return None
        return value

    def clear(self):
        """Remove all entries."""
        self._store.clear()
'''

PYPROJECT = '[project]\nname = "ttlcache"\nversion = "{ver}"\nrequires-python = ">=3.9"\n'

TEST_BASIC = '''\
from ttlcache import TTLCache


def test_set_then_get():
    c = TTLCache()
    c.set("k", "v", now=0)
    assert c.get("k", now=10) == "v"          # present shortly after


def test_missing_key():
    assert TTLCache().get("nope", now=0) is None
'''

TEST_EXPIRY = '''\
from ttlcache import TTLCache


def test_eventually_expires():
    c = TTLCache()
    c.set("k", "v", now=0)
    assert c.get("k", now=100000) is None     # long gone, for any sane TTL


def test_overwrite_resets():
    c = TTLCache()
    c.set("k", "a", now=0)
    c.set("k", "b", now=5)
    assert c.get("k", now=10) == "b"
'''

TEST_CUSTOM = '''\
from ttlcache import TTLCache


def test_explicit_ttl():
    c = TTLCache(ttl=50)
    c.set("k", "v", now=0)
    assert c.get("k", now=40) == "v"
    assert c.get("k", now=60) is None
'''

TEST_CLEAR = '''\
from ttlcache import TTLCache


def test_clear_removes_all():
    c = TTLCache()
    c.set("a", "1", now=0)
    c.set("b", "2", now=0)
    c.clear()
    assert c.get("a", now=1) is None
    assert c.get("b", now=1) is None
'''


def main():
    if OUT.exists():
        subprocess.run(["rm", "-rf", str(OUT)], check=True)
    OUT.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=OUT, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=OUT, check=True)

    def write(path, content):
        p = OUT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def commit(msg, day):
        date = f"2024-05-{day:02d}T10:00:00"
        env = {**os.environ,
               "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
               "GIT_AUTHOR_NAME": "Sam Maintainer", "GIT_AUTHOR_EMAIL": "sam@example.com",
               "GIT_COMMITTER_NAME": "Sam Maintainer", "GIT_COMMITTER_EMAIL": "sam@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    write("pyproject.toml", PYPROJECT.format(ver="0.1.0"))
    write("README.md", "# ttlcache\n\nA tiny in-memory TTL cache.\n")
    write("ttlcache/__init__.py", '"""ttlcache package."""\n')
    commit("chore: scaffold project", 2)

    write("ttlcache/limiter.py" if False else "ttlcache/cache.py", CACHE_V1)
    write("ttlcache/__init__.py", '"""ttlcache package."""\nfrom .cache import TTLCache, DEFAULT_TTL\n\n__all__ = ["TTLCache", "DEFAULT_TTL"]\n')
    commit("feat: TTLCache with default 287s lifetime (measured p99 auth-token refresh)", 4)

    write("tests/test_basic.py", TEST_BASIC)
    commit("test: basic set/get and missing keys", 5)

    write("tests/test_expiry.py", TEST_EXPIRY)
    commit("test: expiry and overwrite", 7)

    write("tests/test_custom.py", TEST_CUSTOM)
    commit("feat: support an explicit per-cache ttl", 9)

    write("README.md", "# ttlcache\n\nA tiny in-memory TTL cache.\n\n```python\nfrom ttlcache import TTLCache\nc = TTLCache()\nc.set('k', 'v', now=time.monotonic())\n```\n")
    commit("docs: usage example", 11)

    # C6: the regression, bundled with clear()
    write("ttlcache/cache.py", CACHE_V2)
    write("tests/test_clear.py", TEST_CLEAR)
    commit("refactor: centralize cache constants; add clear()", 16)

    write("pyproject.toml", PYPROJECT.format(ver="0.4.0"))
    commit("chore: release 0.4.0", 18)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
