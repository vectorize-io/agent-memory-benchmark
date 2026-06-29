"""Build `minicalc` with a MULTI-FILE / MULTI-HOP regression (range off-by-one).

Reuses minicalc's modules (tokens/nodes/errors/refs/parser/sheet/functions) and engineers a
history where a "performance" refactor INLINED range expansion into TWO files — evaluator.py
and engine.py — both with the SAME off-by-one (drops the last row of a range). So:
  - symptom: SUM(A1:A3) over a vertical range drops the last cell (exercises the evaluator path).
  - the cause is NOT in SUM/COUNT (functions.py) — it's the inlined loops; and fixing only the
    evaluator passes the repro but a hidden test that goes through engine.range_values() still
    fails (the OTHER inlined copy). The correct fix touches BOTH files (or restores the shared
    refs.expand_range helper in both).
  - the policy ("ranges are inclusive of both endpoints") lives in refs.expand_range's docstring
    and history; the refactor dropped it from the inlined copies.
This is multi-hop (symptom -> evaluator path -> a second duplicated site in engine.py) and the
fix spans two files; without history the agent finds one site, needs a feedback round for the other.

Usage: python build_range.py <output_dir>   (default: /tmp/sdebench/minicalc_range)
"""
import os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import (TOKENS, NODES, ERRORS, REFS, PARSER, SHEET, FUNCTIONS, INIT,
                   T_BASIC, T_CELLS, T_FUNCS)

# Existing error tests that DON'T exercise formula-cells-in-ranges (so they stay green under the
# planted raw-vs-computed range bug): arithmetic propagation + SUM over a directly-stored error.
T_ERRORS = '''\
from minicalc import Sheet, evaluate, DIV0


def test_div_zero_propagates():
    s = Sheet()
    s.set_many({"A1": 10, "A2": 0})
    assert evaluate("=A1/A2", s) == DIV0
    assert evaluate("=A1/A2 + 1", s) == DIV0


def test_sum_propagates_stored_error():
    s = Sheet()
    s.set_many({"A1": 1, "A2": DIV0, "A3": 3})   # error stored directly in the cell
    assert evaluate("=SUM(A1:A3)", s) == DIV0
'''

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/minicalc_range")

# evaluator that uses the shared refs.expand_range helper (correct).
EVAL_HELPER = '''\
"""Evaluate an AST against a Sheet. Errors propagate through operators; for function calls the
raw argument values (errors included) are passed to the function, which decides."""
from . import nodes
from .refs import expand_range
from .errors import is_error, DIV0, VALUE, REF, NAME_ERR
from .functions import FUNCTIONS


class Evaluator:
    def __init__(self, sheet, engine):
        self.sheet = sheet
        self.engine = engine

    def eval(self, node):
        if isinstance(node, nodes.Num):
            return node.value
        if isinstance(node, nodes.Str):
            return node.value
        if isinstance(node, nodes.Bool):
            return node.value
        if isinstance(node, nodes.CellRef):
            return self.engine.cell_value(node.name)
        if isinstance(node, nodes.RangeRef):
            return [self.engine.cell_value(c) for c in expand_range(node.start, node.end)]
        if isinstance(node, nodes.UnaryOp):
            v = self.eval(node.operand)
            return v if is_error(v) else -v
        if isinstance(node, nodes.BinOp):
            return self.eval_binop(node)
        if isinstance(node, nodes.FuncCall):
            fn = FUNCTIONS.get(node.name)
            if fn is None:
                return NAME_ERR
            return fn([self.eval(a) for a in node.args])
        raise TypeError(f"cannot eval {node!r}")

    def eval_binop(self, node):
        op = node.op
        left = self.eval(node.left)
        right = self.eval(node.right)
        if is_error(left):
            return left
        if is_error(right):
            return right
        if op == "&":
            return _fmt(left) + _fmt(right)
        if op in ("=", "<>", "<", ">", "<=", ">="):
            return _compare(op, left, right)
        l, r = _num(left), _num(right)
        if op == "+": return l + r
        if op == "-": return l - r
        if op == "*": return l * r
        if op == "/": return DIV0 if r == 0 else l / r
        if op == "^": return l ** r
        return VALUE


def _num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _compare(op, a, b):
    if op == "=": return a == b
    if op == "<>": return a != b
    if op == "<": return a < b
    if op == ">": return a > b
    if op == "<=": return a <= b
    if op == ">=": return a >= b
'''

# REGRESSION: evaluator inlines the range loop but reads sheet.raw (the STORED value) instead
# of engine.cell_value (the COMPUTED value), so a range that contains a FORMULA cell drops it
# (the raw "=..." string is not a number). Ranges of plain literals are unaffected.
EVAL_INLINED_BAD = EVAL_HELPER.replace(
    "from .refs import expand_range\n",
    "from .refs import parse_ref, make_ref\n",
).replace(
    '''        if isinstance(node, nodes.RangeRef):
            return [self.engine.cell_value(c) for c in expand_range(node.start, node.end)]''',
    '''        if isinstance(node, nodes.RangeRef):
            c0, r0 = parse_ref(node.start)
            c1, r1 = parse_ref(node.end)
            if c0 > c1: c0, c1 = c1, c0
            if r0 > r1: r0, r1 = r1, r0
            out = []
            for r in range(r0, r1 + 1):             # inlined for speed
                for c in range(c0, c1 + 1):
                    out.append(self.sheet.raw(make_ref(c, r)))
            return out''',
)

# engine that exposes range_values via the shared helper (correct).
ENGINE_HELPER = '''\
"""The engine: parse + evaluate formulas against a sheet, with cell dependency resolution."""
from .tokens import tokenize
from .parser import parse
from .evaluator import Evaluator
from .refs import expand_range
from .errors import is_error, REF


class Engine:
    def __init__(self, sheet):
        self.sheet = sheet
        self._evaluating = set()

    def evaluate(self, formula):
        if isinstance(formula, str) and formula.startswith("="):
            formula = formula[1:]
        ast = parse(tokenize(formula))
        return Evaluator(self.sheet, self).eval(ast)

    def range_values(self, start, end):
        """The list of computed values for every cell in the inclusive block start..end."""
        return [self.cell_value(c) for c in expand_range(start, end)]

    def cell_value(self, name):
        name = name.upper()
        raw = self.sheet.raw(name)
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        if not isinstance(raw, str):
            return raw
        if not raw.startswith("="):
            try:
                return float(raw)
            except ValueError:
                return raw
        if name in self._evaluating:
            return REF
        self._evaluating.add(name)
        try:
            ast = parse(tokenize(raw[1:]))
            val = Evaluator(self.sheet, self).eval(ast)
        finally:
            self._evaluating.discard(name)
        self.sheet.store_value(name, val)
        return val


def evaluate(formula, sheet):
    return Engine(sheet).evaluate(formula)
'''

# REGRESSION: engine.range_values inlines the SAME loop with the SAME raw-vs-computed bug
# (the second duplicated site). Reads sheet.raw instead of self.cell_value.
ENGINE_INLINED_BAD = ENGINE_HELPER.replace(
    "from .refs import expand_range\n",
    "from .refs import parse_ref, make_ref\n",
).replace(
    '''    def range_values(self, start, end):
        """The list of computed values for every cell in the inclusive block start..end."""
        return [self.cell_value(c) for c in expand_range(start, end)]''',
    '''    def range_values(self, start, end):
        """The list of computed values for every cell in the block start..end."""
        c0, r0 = parse_ref(start)
        c1, r1 = parse_ref(end)
        if c0 > c1: c0, c1 = c1, c0
        if r0 > r1: r0, r1 = r1, r0
        out = []
        for r in range(r0, r1 + 1):                 # inlined for speed
            for c in range(c0, c1 + 1):
                out.append(self.sheet.raw(make_ref(c, r)))
        return out''',
)

T_RANGE = '''\
from minicalc import Sheet, evaluate, Engine


def test_range_values_helper():
    s = Sheet()
    s.set_many({"A1": 1, "A2": 2, "A3": 3})
    assert Engine(s).range_values("A1", "A3") == [1, 2, 3]


def test_two_d_range():
    s = Sheet()
    s.set_many({"A1": 1, "B1": 2, "A2": 3, "B2": 4})
    assert evaluate("=SUM(A1:B2)", s) == 10
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

    day = [1]

    def commit(msg, author="Robin Vale"):
        d = f"2023-{(day[0] // 28) % 12 + 1:02d}-{(day[0] % 28) + 1:02d}T09:00:00"
        day[0] += 2
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
               "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "dev@example.com",
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "dev@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    write("pyproject.toml", '[project]\nname = "minicalc"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# minicalc\n\nA tiny spreadsheet formula engine.\n")
    write("minicalc/__init__.py", '"""minicalc package."""\n')
    commit("scaffold minicalc package")
    write("minicalc/tokens.py", TOKENS); commit("tokenizer: numbers, strings, cells, operators, ranges")
    write("minicalc/nodes.py", NODES); commit("AST node types")
    write("minicalc/errors.py", ERRORS); commit("error values (#DIV/0!, #REF!, #VALUE!, #N/A)")
    write("minicalc/refs.py", REFS); commit("cell references and inclusive ranges")
    write("tests/test_tokens.py", "from minicalc.tokens import tokenize\n\n\ndef test_tok():\n    assert [t.kind for t in tokenize('A1+2')][:3] == ['CELL','OP','NUMBER']\n")
    commit("tokenizer tests")
    write("minicalc/parser.py", PARSER); commit("recursive-descent parser with operator precedence")
    write("minicalc/sheet.py", SHEET); commit("sheet: a grid of raw cells")
    write("minicalc/functions.py", FUNCTIONS); commit("built-in functions", author="Mara K.")
    write("minicalc/evaluator.py", EVAL_HELPER)
    write("minicalc/engine.py", ENGINE_HELPER)
    write("minicalc/__init__.py", INIT)
    commit("evaluator + engine: wire formulas to the sheet (ranges via refs.expand_range)")
    write("tests/test_basic.py", T_BASIC); commit("tests: arithmetic precedence and concat")
    write("tests/test_cells.py", T_CELLS); commit("tests: cell refs, ranges, formula cells")
    write("tests/test_funcs.py", T_FUNCS); commit("tests: MIN/MAX/ROUND/IF")
    write("tests/test_errors.py", T_ERRORS); commit("tests: error propagation")
    write("tests/test_range.py", T_RANGE); commit("tests: range_values helper and 2D ranges")
    # noise
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- engine, evaluator, range helpers\n")
    commit("start a changelog")
    write("README.md", "# minicalc\n\nA tiny spreadsheet formula engine.\n\nSupports cell refs, inclusive ranges, operators, and built-in functions.\n")
    commit("readme: list capabilities", author="Mara K.")
    # THE MULTI-FILE REGRESSION: inline range expansion into BOTH files, both off-by-one
    write("minicalc/evaluator.py", EVAL_INLINED_BAD)
    write("minicalc/engine.py", ENGINE_INLINED_BAD)
    commit("perf: inline range expansion in the hot paths", author="Mara K.")
    # noise after
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- engine, evaluator, range helpers\n- perf: inlined range expansion\n")
    commit("changelog: note the perf work")
    write("pyproject.toml", '[project]\nname = "minicalc"\nversion = "0.4.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.4.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
