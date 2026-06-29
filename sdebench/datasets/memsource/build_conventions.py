import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ms_k"
scaffold(out)
write(out, "pay/rounding.py", ROUND_UP); write(out, "pay/__init__.py", PKG)   # always half-up (no half-down in history)
commit(out, "feat: round_cents helper")
write(out, "tests/test_basic.py", T_BASIC); commit(out, "tests for rounding")
write(out, "CONVENTIONS.md", "# Engineering conventions\n\n## Money & rounding\nAll monetary amounts must round half-cents **DOWN** (ROUND_HALF_DOWN) to match the legacy billing ledger. Do NOT use half-up or banker's rounding — they drift totals away from the ledger.\n")
commit(out, "docs: engineering conventions")
finish(out)   # SOURCE K: the rule is only in CONVENTIONS.md
