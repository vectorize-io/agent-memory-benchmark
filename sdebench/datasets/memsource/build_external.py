import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/ms_ext"
scaffold(out)
write(out, "pay/rounding.py", ROUND_UP); write(out, "pay/__init__.py", PKG)
commit(out, "feat: add round_cents helper")
write(out, "tests/test_basic.py", T_BASIC); commit(out, "add rounding tests")
write(out, "README.md", "# pay\n\nMoney helpers.\n\n```python\nfrom pay import round_cents\nround_cents('2.345')\n```\n")
commit(out, "docs: readme usage")             # no rule anywhere: git/docs/code all silent
finish(out)
