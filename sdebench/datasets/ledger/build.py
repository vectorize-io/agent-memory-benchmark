"""Build the `ledger` synthetic repo — a HISTORY-DEPENDENT regression (rounding MODE).

A refactor ("simplify rounding; add format_cents()") switched round_cents from BANKER'S
rounding (ROUND_HALF_EVEN) to ROUND_HALF_UP, and dropped the comment explaining WHY it was
half-even (to match the accounting ledger / avoid upward drift). The repro only shows ONE
amount rounding the wrong way — it does NOT reveal the rule. The hidden tests pin the mode to
banker's (half-even). A naive fix (half-up / half-down / truncate) passes the repro but fails
hidden. The rule "ROUND_HALF_EVEN, do not use half-up" now lives ONLY in git history, so:
  - with history: git log/blame the function -> "banker's rounding" -> fixed in one shot.
  - without history: the agent must discover the exact mode via feedback rounds.
The refactor also bundles a legit format_cents() (with its own test), so a lazy `git revert`
fails PASS_TO_PASS -> forces a surgical one-line fix (the rounding mode).

Usage: python build.py <output_dir>   (default: /tmp/sdebench/ledger)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ledger")

ROUND_V1 = '''\
"""Money rounding helpers for the accounting ledger."""
from decimal import Decimal, ROUND_HALF_EVEN


def round_cents(amount):
    """Round `amount` to cents using BANKER'S ROUNDING (round-half-to-even).

    We deliberately use half-to-even rather than half-up so rounded totals match the
    accounting ledger and rounding error does not accumulate in one direction. Do NOT
    switch this to half-up — it drifts the books upward over many transactions.
    """
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
'''

# C5 refactor: ROUND_HALF_EVEN -> ROUND_HALF_UP (REGRESSION), rationale dropped, + format_cents()
ROUND_V2 = '''\
"""Money rounding helpers for the accounting ledger."""
from decimal import Decimal, ROUND_HALF_UP


def round_cents(amount):
    """Round `amount` to the nearest cent."""
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def format_cents(amount):
    """Format `amount` as a dollar string, e.g. '$2.12'."""
    return f"${round_cents(amount)}"
'''

PYPROJECT = '[project]\nname = "ledger"\nversion = "{ver}"\nrequires-python = ">=3.9"\n'

TEST_BASIC = '''\
from decimal import Decimal
from ledger import round_cents


def test_rounds_typical_amounts():
    # amounts that are NOT exact halves round the same under any sane mode
    assert round_cents("1.234") == Decimal("1.23")
    assert round_cents("1.239") == Decimal("1.24")
    assert round_cents("0.006") == Decimal("0.01")


def test_passes_through_whole_cents():
    assert round_cents("5.00") == Decimal("5.00")
    assert round_cents("0.01") == Decimal("0.01")
'''

TEST_MORE = '''\
from decimal import Decimal
from ledger import round_cents


def test_rounds_large_amount():
    assert round_cents("1234.561") == Decimal("1234.56")
    assert round_cents("9.999") == Decimal("10.00")
'''

TEST_FORMAT = '''\
from ledger import format_cents


def test_format_cents():
    assert format_cents("2.5") == "$2.50"
    assert format_cents("0.1") == "$0.10"
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
        date = f"2024-06-{day:02d}T10:00:00"
        env = {**os.environ,
               "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
               "GIT_AUTHOR_NAME": "Pat Ledger", "GIT_AUTHOR_EMAIL": "pat@example.com",
               "GIT_COMMITTER_NAME": "Pat Ledger", "GIT_COMMITTER_EMAIL": "pat@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    write("pyproject.toml", PYPROJECT.format(ver="0.1.0"))
    write("README.md", "# ledger\n\nMoney rounding helpers for the accounting ledger.\n")
    write("ledger/__init__.py", '"""ledger package."""\n')
    commit("chore: scaffold project", 2)

    write("ledger/rounding.py", ROUND_V1)
    write("ledger/__init__.py", '"""ledger package."""\nfrom .rounding import round_cents\n\n__all__ = ["round_cents"]\n')
    commit("feat: round_cents uses banker's rounding (half-to-even) to match the ledger", 4)

    write("tests/test_basic.py", TEST_BASIC)
    commit("test: rounding of typical amounts", 5)

    write("tests/test_more.py", TEST_MORE)
    commit("test: large amounts and carry", 8)

    write("README.md", "# ledger\n\nMoney rounding helpers.\n\n```python\nfrom ledger import round_cents\nround_cents('2.345')\n```\n")
    commit("docs: usage example", 11)

    # C6: the regression, bundled with format_cents()
    write("ledger/rounding.py", ROUND_V2)
    write("ledger/__init__.py", '"""ledger package."""\nfrom .rounding import round_cents, format_cents\n\n__all__ = ["round_cents", "format_cents"]\n')
    write("tests/test_format.py", TEST_FORMAT)
    commit("refactor: simplify rounding; add format_cents()", 17)

    write("pyproject.toml", PYPROJECT.format(ver="0.3.0"))
    commit("chore: release 0.3.0", 19)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
