"""Shared pieces for the memory-SOURCE tasks. Same bug (round_cents half-up at HEAD) + same
repro/hidden across H/K/F/X builds; only WHERE the half-down rule lives differs.
The half-down rule is non-guessable (agents default to half-up/banker's for money)."""
import os, subprocess
from pathlib import Path

PKG = '"""pay package."""\nfrom .rounding import round_cents\n\n__all__ = ["round_cents"]\n'

# HEAD code (the bug): rounds half-up, no rationale.
ROUND_UP = '''"""Money rounding."""
from decimal import Decimal, ROUND_HALF_UP


def round_cents(amount):
    """Round an amount to the nearest whole cent."""
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
'''

# the original/correct code (used only by the History build's earlier commit).
ROUND_DOWN = '''"""Money rounding."""
from decimal import Decimal, ROUND_HALF_DOWN


def round_cents(amount):
    """Round an amount to whole cents, rounding half-cents DOWN (toward zero) to match the
    legacy billing ledger. Do NOT use half-up or banker's rounding."""
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_DOWN)
'''

T_BASIC = '''from decimal import Decimal
from pay import round_cents


def test_typical_amounts():
    # non-half amounts round the same under any sane mode
    assert round_cents("1.234") == Decimal("1.23")
    assert round_cents("1.239") == Decimal("1.24")
    assert round_cents("5.00") == Decimal("5.00")
'''

_day = [1]


def init(out):
    out = str(out)
    if Path(out).exists():
        subprocess.run(["rm", "-rf", out], check=True)
    Path(out).mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=out, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=out, check=True)
    _day[0] = 1


def write(out, path, content):
    p = Path(out) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def commit(out, msg, author="Sam Dev"):
    d = f"2024-03-{_day[0]:02d}T10:00:00"
    _day[0] += 2
    env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
           "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "d@e.com",
           "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "d@e.com"}
    subprocess.run(["git", "add", "-A"], cwd=str(out), check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=str(out), env=env, check=True)


def scaffold(out):
    """Common scaffold: pyproject, README, package with the buggy (half-up) rounding + tests.
    NONE of these mention the half-down rule — that's placed per-source by each build."""
    init(out)
    write(out, "pyproject.toml", '[project]\nname = "pay"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write(out, "README.md", "# pay\n\nMoney helpers.\n")
    write(out, "pay/__init__.py", '"""pay package."""\n')
    commit(out, "scaffold project")


def finish(out):
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(out), capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(out), capture_output=True, text=True).stdout.strip()
    print(f"built {out} @ {head} ({n} commits)")
