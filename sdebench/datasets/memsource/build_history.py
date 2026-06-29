import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ms_h"
scaffold(out)
write(out, "pay/rounding.py", ROUND_DOWN); write(out, "pay/__init__.py", PKG)   # rule lived in code+commit
commit(out, "feat: round_cents rounds half-cents DOWN to match the legacy ledger")
write(out, "tests/test_basic.py", T_BASIC); commit(out, "tests for rounding")
write(out, "pay/rounding.py", ROUND_UP); commit(out, "refactor: simplify rounding helper")  # REGRESSION drops it
write(out, "pyproject.toml", '[project]\nname = "pay"\nversion = "0.2.0"\nrequires-python = ">=3.9"\n'); commit(out, "release 0.2.0")
finish(out)   # SOURCE H: the rule is only in git history now
