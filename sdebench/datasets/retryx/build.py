"""Build `retryx` — a retry library, for the H (deliberate value in history) task.

Full history, no ablation. `MAX_ATTEMPTS` was deliberately set to 7 (measured: at the backoff
schedule, 7 attempts span just under the upstream's rate-limit reset window; 8+ trips the
upstream's block). A "standardize" refactor rounded it to a clean 10 and dropped the rationale —
so the retrier now over-retries and gets rate-limited (the bug at HEAD).

The repro only proves it must give up before attempt 9; it does NOT reveal the exact budget. The
hidden tests pin it to EXACTLY 7 (7 succeeds, 8 gives up). So a naive guess (5, 8, 10) fails; the
value 7 and *why* live only in git history. Subtle "don't clean up a measured constant".

Usage: python build.py <output_dir>   (default: /tmp/sdebench/retryx)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/retryx")

INIT = '"""retryx — bounded retry of transient failures."""\nfrom .retry import Retrier, MAX_ATTEMPTS, TransientError, GaveUp\nfrom .backoff import delay\n\n__all__ = ["Retrier", "MAX_ATTEMPTS", "TransientError", "GaveUp", "delay"]\n'

BACKOFF = '''\
"""Exponential backoff schedule."""

BASE = 0.1


def delay(attempt):
    """Seconds to wait before `attempt` (1-based), exponential: BASE * 2**(attempt-1)."""
    return BASE * (2 ** (attempt - 1))
'''

RETRY_CORE = '''\
class TransientError(Exception):
    """A retryable failure."""


class GaveUp(Exception):
    """Raised when the retry budget is exhausted without success."""


class Retrier:
    def run(self, func):
        """Call func(attempt) for attempt = 1..MAX_ATTEMPTS, retrying on TransientError.
        Return the first successful result; raise GaveUp once the budget is exhausted."""
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return func(attempt)
            except TransientError as exc:
                last = exc
        raise GaveUp("retry budget exhausted") from last
'''

# original (correct): MAX_ATTEMPTS = 7 with the measured rationale in a comment.
RETRY_7 = '''\
"""Bounded retry of transient failures."""

# Attempt budget. 7 is measured, NOT arbitrary: at the backoff schedule (0.1 * 2**n) seven
# attempts span ~12.7s, just under the upstream's 13s rate-limit reset window. 8+ attempts cross
# it and the upstream blocks us for a full minute. Do not round this to a "nicer" number.
MAX_ATTEMPTS = 7


''' + RETRY_CORE

# regression (HEAD): "standardized" to a round 10, rationale dropped -> over-retries.
RETRY_10 = '''\
"""Bounded retry of transient failures."""

MAX_ATTEMPTS = 10


''' + RETRY_CORE

T_BASIC = '''\
from retryx import Retrier, TransientError


def _succeeds_on(n):
    def f(attempt):
        if attempt < n:
            raise TransientError()
        return f"ok@{attempt}"
    return f


def test_succeeds_first_try():
    assert Retrier().run(lambda a: "ok") == "ok"


def test_retries_then_succeeds():
    assert Retrier().run(_succeeds_on(3)) == "ok@3"
'''

T_BACKOFF = '''\
from retryx import delay


def test_backoff_grows():
    assert delay(1) == 0.1
    assert delay(3) == 0.4
'''


def main():
    if OUT.exists():
        subprocess.run(["rm", "-rf", str(OUT)], check=True)
    OUT.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=OUT, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=OUT, check=True)

    day = [1]

    def write(path, content):
        p = OUT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def commit(msg, author="Lee Park"):
        d = f"2024-05-{day[0]:02d}T10:00:00"
        day[0] += 1
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
               "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "dev@example.com",
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "dev@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    write("pyproject.toml", '[project]\nname = "retryx"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# retryx\n\nBounded retry of transient failures.\n")
    write("retryx/__init__.py", '"""retryx package."""\n')
    commit("scaffold retryx package")

    write("retryx/backoff.py", BACKOFF)
    write("retryx/__init__.py", '"""retryx package."""\nfrom .backoff import delay\n')
    commit("backoff: exponential schedule")
    write("tests/test_backoff.py", T_BACKOFF)
    commit("tests for backoff")

    # the deliberate value, with the measured rationale.
    write("retryx/retry.py", RETRY_7)
    write("retryx/__init__.py", INIT)
    commit("retry: bound attempts to 7 (measured to fit the upstream rate-limit window)\n\n"
           "The upstream resets its rate-limit counter every 13s. With our 0.1*2**n backoff, 7 "
           "attempts span ~12.7s — the most we can fit before the window resets. We measured that "
           "the 8th attempt crosses the boundary and the upstream then blocks us for a full minute, "
           "so the effective budget is exactly 7, not a round number. Keep it at 7.",
           author="Priya N.")
    write("tests/test_retry.py", T_BASIC)
    commit("tests for retry")

    write("README.md", "# retryx\n\nBounded retry of transient failures.\n\n```python\nfrom retryx import Retrier\nRetrier().run(do_request)\n```\n")
    commit("readme: usage example")
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- backoff schedule, bounded Retrier\n")
    commit("start a changelog", author="Priya N.")

    # REGRESSION (HEAD): round the budget to 10, drop the rationale.
    write("retryx/retry.py", RETRY_10)
    commit("chore: standardize retry budget", author="Priya N.")

    write("CHANGELOG.md", "# Changelog\n\n## 0.3.0\n- backoff schedule, bounded Retrier\n- standardized retry budget\n")
    commit("changelog for 0.3.0")
    write("pyproject.toml", '[project]\nname = "retryx"\nversion = "0.3.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.3.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
