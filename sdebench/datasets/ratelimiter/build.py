"""Build the `ratelimiter` synthetic repo as a clean, incremental git history.

The history is engineered so a *regression* is bundled inside an otherwise-legit
commit ("perf: ... add available()"): commit C8 introduces an `int()` floor in the
refill path (silently dropping partial tokens) WHILE also adding the `available()`
accessor. A lazy `git revert` of C8 would lose `available()` and fail PASS_TO_PASS,
forcing a surgical fix — and making the git history load-bearing.

Usage: python build.py <output_dir>   (default: /tmp/sdebench/ratelimiter)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ratelimiter")

# ── limiter.py evolves across commits ───────────────────────────────────────
LIMITER_V1 = '''\
"""A small, dependency-free token-bucket rate limiter."""
import time


class TokenBucket:
    """A token bucket that refills continuously at `rate` tokens/second.

    Tokens accrue continuously, so PARTIAL tokens carry over between calls:
    at rate=0.5, half a token is available one second after the bucket empties.
    """

    def __init__(self, rate, capacity):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last = None

    def _refill(self, now):
        if self._last is None:
            self._last = now
            return
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = self._tokens + elapsed * self.rate
            self._last = now

    def try_acquire(self, cost=1, now=None):
        """Consume `cost` tokens if available; return True on success."""
        now = time.monotonic() if now is None else now
        self._refill(now)
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False
'''

# C4: cap refill at capacity
LIMITER_V2 = LIMITER_V1.replace(
    "self._tokens = self._tokens + elapsed * self.rate",
    "self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)",
)

# C5: add RateLimiter (per-key buckets)
LIMITER_V3 = LIMITER_V2 + '''

class RateLimiter:
    """Manage one TokenBucket per key (e.g. per user / per IP)."""

    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self._buckets = {}

    def try_acquire(self, key, cost=1, now=None):
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = TokenBucket(self.rate, self.capacity)
        return bucket.try_acquire(cost, now)
'''

# C8 (REGRESSION): int() floor in _refill (drops partial tokens) + available()
LIMITER_V4 = LIMITER_V3.replace(
    "            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)\n"
    "            self._last = now\n",
    "            # perf: accumulate whole tokens to avoid floating-point drift\n"
    "            self._tokens = min(self.capacity, self._tokens + int(elapsed * self.rate))\n"
    "            self._last = now\n",
).replace(
    "        if self._tokens >= cost:\n            self._tokens -= cost\n            return True\n        return False\n",
    "        if self._tokens >= cost:\n            self._tokens -= cost\n            return True\n        return False\n\n"
    "    def available(self, now=None):\n"
    '        """Return the number of tokens currently available."""\n'
    "        now = time.monotonic() if now is None else now\n"
    "        self._refill(now)\n"
    "        return self._tokens\n",
)

PYPROJECT = '''\
[project]
name = "ratelimiter"
version = "{ver}"
description = "A small token-bucket rate limiter."
requires-python = ">=3.9"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
'''

README = '''\
# ratelimiter

A tiny, dependency-free token-bucket rate limiter.

```python
from ratelimiter import TokenBucket
b = TokenBucket(rate=5, capacity=10)   # 5 tokens/sec, burst of 10
b.try_acquire()                        # -> True while tokens remain
```

Tokens refill **continuously**; partial tokens carry over between calls.
'''

TEST_BASIC = '''\
from ratelimiter import TokenBucket


def test_starts_full():
    b = TokenBucket(rate=1, capacity=5)
    assert b.try_acquire(5, now=0) is True


def test_denies_when_empty():
    b = TokenBucket(rate=1, capacity=2)
    assert b.try_acquire(2, now=0) is True
    assert b.try_acquire(1, now=0) is False
'''

TEST_CAP = '''\
from ratelimiter import TokenBucket


def test_refill_caps_at_capacity():
    b = TokenBucket(rate=10, capacity=3)
    b.try_acquire(3, now=0)            # empty it
    # after a long time it should refill only up to capacity, not beyond
    assert b.try_acquire(3, now=100) is True
    assert b.try_acquire(1, now=100) is False
'''

TEST_RL = '''\
from ratelimiter import RateLimiter


def test_keys_are_independent():
    rl = RateLimiter(rate=1, capacity=1)
    assert rl.try_acquire("a", now=0) is True
    assert rl.try_acquire("a", now=0) is False
    assert rl.try_acquire("b", now=0) is True
'''

TEST_REFILL = '''\
from ratelimiter import TokenBucket


def test_integer_rate_refills_over_time():
    b = TokenBucket(rate=2, capacity=10)
    b.try_acquire(10, now=0)          # empty
    assert b.try_acquire(1, now=0) is False
    # 2 tokens/sec for 3s -> 6 tokens available
    assert b.try_acquire(6, now=3) is True
'''

TEST_AVAILABLE = '''\
from ratelimiter import TokenBucket


def test_available_reports_tokens():
    b = TokenBucket(rate=2, capacity=10)
    assert b.available(now=0) == 10
    b.try_acquire(4, now=0)
    assert b.available(now=0) == 6
'''

CHANGELOG = '''\
# Changelog

## 0.3.0
- perf: refill accumulates whole tokens to avoid float drift
- feat: `TokenBucket.available()` accessor

## 0.2.0
- feat: `RateLimiter` with a bucket per key

## 0.1.0
- initial token bucket
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
        date = f"2024-03-{day:02d}T10:00:00"
        env = {**os.environ,
               "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
               "GIT_AUTHOR_NAME": "Dana Dev", "GIT_AUTHOR_EMAIL": "dana@example.com",
               "GIT_COMMITTER_NAME": "Dana Dev", "GIT_COMMITTER_EMAIL": "dana@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    # C1
    write("pyproject.toml", PYPROJECT.format(ver="0.1.0"))
    write("README.md", "# ratelimiter\n\nA tiny token-bucket rate limiter.\n")
    write("ratelimiter/__init__.py", '"""ratelimiter package."""\n')
    commit("chore: scaffold project", 1)
    # C2
    write("ratelimiter/limiter.py", LIMITER_V1)
    write("ratelimiter/__init__.py", '"""ratelimiter package."""\nfrom .limiter import TokenBucket\n\n__all__ = ["TokenBucket"]\n')
    commit("feat: TokenBucket with continuous refill (partial tokens carry over)", 3)
    # C3
    write("tests/test_basic.py", TEST_BASIC)
    commit("test: cover capacity, consume, and deny", 4)
    # C4
    write("ratelimiter/limiter.py", LIMITER_V2)
    write("tests/test_cap.py", TEST_CAP)
    commit("feat: cap tokens at capacity on refill", 6)
    # C5
    write("ratelimiter/limiter.py", LIMITER_V3)
    write("ratelimiter/__init__.py", '"""ratelimiter package."""\nfrom .limiter import TokenBucket, RateLimiter\n\n__all__ = ["TokenBucket", "RateLimiter"]\n')
    write("tests/test_ratelimiter.py", TEST_RL)
    commit("feat: RateLimiter manages a bucket per key", 9)
    # C6
    write("tests/test_refill.py", TEST_REFILL)
    commit("test: integer-rate refill recovers tokens over time", 11)
    # C7
    write("README.md", README)
    commit("docs: usage example and refill semantics", 13)
    # C8  <-- REGRESSION bundled with a legit accessor
    write("ratelimiter/limiter.py", LIMITER_V4)
    write("tests/test_available.py", TEST_AVAILABLE)
    commit("perf: accumulate whole tokens to avoid float drift; add available()", 18)
    # C9
    write("pyproject.toml", PYPROJECT.format(ver="0.3.0"))
    write("CHANGELOG.md", CHANGELOG)
    commit("chore: release 0.3.0", 20)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head[:10]} ({subprocess.run(['git','rev-list','--count','HEAD'],cwd=OUT,capture_output=True,text=True).stdout.strip()} commits)")


if __name__ == "__main__":
    main()
