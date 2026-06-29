import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ms_f"
scaffold(out)
write(out, "pay/rounding.py", ROUND_UP); write(out, "pay/__init__.py", PKG)
commit(out, "feat: round_cents helper")
write(out, "tests/test_basic.py", T_BASIC); commit(out, "tests for rounding")
write(out, ".agent/notes.md", "# Retained user/team feedback\n\n- (code review, 2024-02) Money must round half-cents **DOWN** to match the legacy ledger. The reviewer flagged half-up/banker's rounding as a recurring mistake here — always use ROUND_HALF_DOWN for money.\n")
commit(out, "chore: keep review notes")
finish(out)   # SOURCE F: the rule is only in retained feedback (.agent/notes.md)
