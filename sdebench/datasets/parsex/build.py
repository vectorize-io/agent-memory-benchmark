"""Build `parsex` — a parsing library, for the X (cross-feature pattern) task.

Full history, no ablation. Two sibling parsers: `parse_duration` (already fixed) and `parse_size`
(still buggy). They share a class of bug: an unknown/missing unit. `parse_duration` was fixed
earlier to treat an unknown unit as the BASE unit (a deliberate lenient-parsing choice for
backward-compat with bare-number configs) instead of crashing. `parse_size` never got that fix —
it raises on an unknown unit (the reported bug at HEAD).

The non-obvious part: the obvious "graceful" fix (raise a clean error, or return None) FAILS the
hidden tests, which enforce the project's established convention — unknown unit => base unit.
That convention is visible in the sibling `parse_duration` (current code + the commit that
established it). A "smart" agent notices the recurring pattern and applies the sibling's fix.

Usage: python build.py <output_dir>   (default: /tmp/sdebench/parsex)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/parsex")

SPLIT = '''\
"""Shared: split a value like '10mb' into (number, unit)."""
import re

_RE = re.compile(r"^\\s*([0-9]+(?:\\.[0-9]+)?)\\s*([a-z]*)\\s*$", re.I)


def split_value(text):
    """'10mb' -> (10.0, 'mb'); '100' -> (100.0, ''); 'abc' -> (None, '')."""
    m = _RE.match(text or "")
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2).lower()
'''

DURATION = '''\
"""Parse human durations like '5m' into seconds."""
from .split import split_value

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """'5m' -> 300.0 seconds. An unknown or missing unit is treated as the base unit (seconds),
    so bare numbers and legacy/typo'd units parse leniently instead of crashing ingestion;
    a non-numeric value returns None."""
    number, unit = split_value(text)
    if number is None:
        return None
    return number * _UNITS.get(unit, 1)
'''

# parse_size: still has the un-fixed bug — raises (KeyError) on an unknown unit.
SIZE_BUG = '''\
"""Parse human byte sizes like '10mb' into bytes."""
from .split import split_value

_UNITS = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}


def parse_size(text):
    """'10mb' -> 10485760.0 bytes."""
    number, unit = split_value(text)
    return number * _UNITS[unit]
'''

INIT = '"""parsex — lenient parsing of durations and sizes."""\nfrom .duration import parse_duration\nfrom .size import parse_size\n\n__all__ = ["parse_duration", "parse_size"]\n'

T_DURATION = '''\
from parsex import parse_duration


def test_known_units():
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200


def test_unknown_unit_is_base():
    assert parse_duration("30") == 30          # bare number -> seconds
    assert parse_duration("12x") == 12         # unknown unit -> base unit (seconds)


def test_non_numeric():
    assert parse_duration("abc") is None
'''

T_SIZE = '''\
from parsex import parse_size


def test_known_sizes():
    assert parse_size("2kb") == 2048
    assert parse_size("1mb") == 1048576
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

    def commit(msg, author="Sky Tan"):
        d = f"2024-06-{day[0]:02d}T10:00:00"
        day[0] += 1
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
               "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "dev@example.com",
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "dev@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    # the early, buggy parse_duration (raises on unknown unit) — so the fix has real history.
    DURATION_BUG = DURATION.replace(
        '''    number, unit = split_value(text)
    if number is None:
        return None
    return number * _UNITS.get(unit, 1)''',
        '''    number, unit = split_value(text)
    if number is None:
        return None
    return number * _UNITS[unit]''')

    write("pyproject.toml", '[project]\nname = "parsex"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# parsex\n\nLenient parsing of durations and sizes.\n")
    write("parsex/__init__.py", '"""parsex package."""\n')
    commit("scaffold parsex package")

    write("parsex/split.py", SPLIT)
    commit("split: number/unit splitter")
    write("parsex/duration.py", DURATION_BUG)
    write("parsex/__init__.py", '"""parsex package."""\nfrom .duration import parse_duration\n')
    commit("duration: parse '5m' style durations")
    # THE FIX-PATTERN (and rationale) — applied to duration; the sibling to copy from.
    write("parsex/duration.py", DURATION)
    commit("duration: treat an unknown unit as the base unit instead of raising\n\n"
           "Ingestion was crashing on legacy duration strings — old configs wrote bare numbers "
           "(e.g. '30' meaning 30s) and a few had typo'd units. Rather than raising and dropping the "
           "whole record, fall back to the base unit (seconds) for an unknown/missing unit; only a "
           "non-numeric value returns None. Keep parsing lenient.",
           author="Priya N.")
    write("tests/test_duration.py", T_DURATION)
    commit("tests for duration parsing")

    write("README.md", "# parsex\n\nLenient parsing of durations and sizes.\n\nUnknown/missing units fall back to the base unit; non-numeric input returns None.\n")
    commit("readme: note lenient parsing")

    # the sibling, carrying the SAME un-fixed bug.
    write("parsex/size.py", SIZE_BUG)
    write("parsex/__init__.py", INIT)
    commit("size: parse '10mb' style byte sizes", author="Priya N.")
    write("tests/test_size.py", T_SIZE)
    commit("tests for size parsing")

    write("CHANGELOG.md", "# Changelog\n\n## 0.2.0\n- duration and size parsers\n")
    commit("changelog 0.2.0")
    write("pyproject.toml", '[project]\nname = "parsex"\nversion = "0.2.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.2.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
