import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ms_x"
scaffold(out)
write(out, "pay/rounding.py", ROUND_UP); write(out, "pay/__init__.py", PKG)
commit(out, "feat: round_cents helper")
write(out, "tests/test_basic.py", T_BASIC); commit(out, "tests for rounding")
write(out, ".agent/past_fixes.md", "# Past fixes (this agent's memory of prior solves)\n\n- Bug: rounded totals didn't match the ledger. Fix: switched round_cents to ROUND_HALF_DOWN — the ledger rounds half-cents down. Lesson: for money in this org, always half-down (not half-up/banker's).\n")
commit(out, "chore: agent memory of past fixes")
finish(out)   # SOURCE X: the rule is only in a past-solve record (.agent/past_fixes.md)
