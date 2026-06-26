"""Build `minicalc` — a LARGER, HARDER codebase: a spreadsheet formula engine.

~9 modules (tokenizer, parser, AST, refs, sheet, functions, evaluator, errors, engine),
longer files, and a long noisy git history. The planted regression is FAR FROM ITS SYMPTOM
and history-dependent + non-guessable:

  Policy (documented in history): the evaluator passes raw cell values — INCLUDING error
  cells — to each function, and EACH FUNCTION decides how to handle errors. SUM and the
  arithmetic operators PROPAGATE errors; COUNT/AVG/MIN/MAX SKIP error cells (count/aggregate
  only the numbers). A refactor ("centralize argument evaluation") made the evaluator
  SHORT-CIRCUIT: if any argument is an error it returns that error before the function runs.
  Symptom: COUNT(A1:A5) with one #DIV/0! cell returns #DIV/0! instead of 4 — but the bug is
  in evaluator.py (Call arg handling), not in COUNT. The fix is non-guessable (real engines
  vary) and underdetermined (special-casing COUNT passes the repro but fails AVG/MIN/MAX in
  the hidden tests); only history states "the evaluator must not short-circuit Call args".

Usage: python build.py <output_dir>   (default: /tmp/sdebench/minicalc)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/minicalc")

# ── module sources ──────────────────────────────────────────────────────────
ERRORS = '''\
"""Spreadsheet error values (#DIV/0!, #REF!, #VALUE!, #N/A) and helpers."""


class CalcError:
    """An error value that flows through a formula like Excel's #DIV/0! etc."""

    __slots__ = ("code",)

    def __init__(self, code):
        self.code = code

    def __eq__(self, other):
        return isinstance(other, CalcError) and other.code == self.code

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        return self.code


DIV0 = CalcError("#DIV/0!")
REF = CalcError("#REF!")
VALUE = CalcError("#VALUE!")
NA = CalcError("#N/A")
NAME_ERR = CalcError("#NAME?")


def is_error(v):
    return isinstance(v, CalcError)
'''

TOKENS = '''\
"""Tokenizer: turn a formula string into a flat list of tokens."""
import re

NUMBER, STRING, NAME, CELL, OP, LPAREN, RPAREN, COMMA, COLON, EOF = (
    "NUMBER", "STRING", "NAME", "CELL", "OP", "LPAREN", "RPAREN", "COMMA", "COLON", "EOF")

# Two-char operators must be tried before one-char ones.
_OPS2 = ("<=", ">=", "<>")
_OPS1 = "+-*/^&=<>"
_CELL = re.compile(r"[A-Za-z]+[0-9]+")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_]*")
_NUM = re.compile(r"[0-9]+(?:\\.[0-9]+)?")


class Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


def tokenize(text):
    toks, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            toks.append(Token(STRING, text[i + 1:j]))
            i = j + 1
            continue
        if text[i:i + 2] in _OPS2:
            toks.append(Token(OP, text[i:i + 2]))
            i += 2
            continue
        if ch in _OPS1:
            toks.append(Token(OP, ch))
            i += 1
            continue
        if ch == "(":
            toks.append(Token(LPAREN, ch)); i += 1; continue
        if ch == ")":
            toks.append(Token(RPAREN, ch)); i += 1; continue
        if ch == ",":
            toks.append(Token(COMMA, ch)); i += 1; continue
        if ch == ":":
            toks.append(Token(COLON, ch)); i += 1; continue
        m = _CELL.match(text, i)
        if m and not (m.end() < n and text[m.end()].isalpha()):
            toks.append(Token(CELL, m.group().upper())); i = m.end(); continue
        m = _NUM.match(text, i)
        if m:
            toks.append(Token(NUMBER, float(m.group()))); i = m.end(); continue
        m = _NAME.match(text, i)
        if m:
            toks.append(Token(NAME, m.group().upper())); i = m.end(); continue
        raise ValueError(f"unexpected character {ch!r} at {i}")
    toks.append(Token(EOF, None))
    return toks
'''

NODES = '''\
"""AST node types produced by the parser and consumed by the evaluator."""


class Num:
    __slots__ = ("value",)
    def __init__(self, value): self.value = value

class Str:
    __slots__ = ("value",)
    def __init__(self, value): self.value = value

class Bool:
    __slots__ = ("value",)
    def __init__(self, value): self.value = value

class CellRef:
    __slots__ = ("name",)
    def __init__(self, name): self.name = name

class RangeRef:
    __slots__ = ("start", "end")
    def __init__(self, start, end): self.start, self.end = start, end

class BinOp:
    __slots__ = ("op", "left", "right")
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right

class UnaryOp:
    __slots__ = ("op", "operand")
    def __init__(self, op, operand): self.op, self.operand = op, operand

class FuncCall:
    __slots__ = ("name", "args")
    def __init__(self, name, args): self.name, self.args = name, args
'''

REFS = '''\
"""Cell references and ranges. A1 -> (col, row); A1:B3 -> the inclusive block of cells."""
from .tokens import tokenize, CELL


def col_to_idx(col):
    """'A' -> 0, 'B' -> 1, ... 'AA' -> 26."""
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def idx_to_col(idx):
    col = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        col = chr(ord("A") + rem) + col
    return col


def parse_ref(name):
    """'B3' -> (col_idx=1, row=3)."""
    i = 0
    while i < len(name) and name[i].isalpha():
        i += 1
    return col_to_idx(name[:i]), int(name[i:])


def make_ref(col_idx, row):
    return f"{idx_to_col(col_idx)}{row}"


def expand_range(start, end):
    """All cell names in the block start..end, INCLUSIVE of both endpoints,
    row-major. e.g. expand_range('A1','B2') -> ['A1','B1','A2','B2']."""
    c0, r0 = parse_ref(start)
    c1, r1 = parse_ref(end)
    if c0 > c1:
        c0, c1 = c1, c0
    if r0 > r1:
        r0, r1 = r1, r0
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.append(make_ref(c, r))
    return cells
'''

SHEET = '''\
"""The sheet: a grid of cells, each holding a raw literal or a formula string."""


class Sheet:
    def __init__(self):
        self._raw = {}        # cell name -> raw string/number
        self._values = {}     # cell name -> last computed value

    def set(self, name, raw):
        self._raw[name.upper()] = raw

    def set_many(self, mapping):
        for k, v in mapping.items():
            self.set(k, v)

    def raw(self, name):
        return self._raw.get(name.upper())

    def has(self, name):
        return name.upper() in self._raw

    def cells(self):
        return list(self._raw.keys())

    def store_value(self, name, value):
        self._values[name.upper()] = value

    def value(self, name):
        return self._values.get(name.upper())
'''

PARSER = '''\
"""Recursive-descent parser. Precedence (low -> high):
comparison < concat(&) < add/sub < mul/div < unary(-) < power(^, right-assoc) < primary."""
from . import nodes
from .tokens import (tokenize, NUMBER, STRING, NAME, CELL, OP,
                     LPAREN, RPAREN, COMMA, COLON, EOF)

_CMP = {"=", "<", ">", "<=", ">=", "<>"}


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind):
        t = self.next()
        if t.kind != kind:
            raise ValueError(f"expected {kind}, got {t.kind}")
        return t

    def parse(self):
        node = self.comparison()
        if self.peek().kind != EOF:
            raise ValueError(f"unexpected trailing {self.peek().kind}")
        return node

    def comparison(self):
        node = self.concat()
        while self.peek().kind == OP and self.peek().value in _CMP:
            op = self.next().value
            node = nodes.BinOp(op, node, self.concat())
        return node

    def concat(self):
        node = self.addsub()
        while self.peek().kind == OP and self.peek().value == "&":
            self.next()
            node = nodes.BinOp("&", node, self.addsub())
        return node

    def addsub(self):
        node = self.muldiv()
        while self.peek().kind == OP and self.peek().value in ("+", "-"):
            op = self.next().value
            node = nodes.BinOp(op, node, self.muldiv())
        return node

    def muldiv(self):
        node = self.unary()
        while self.peek().kind == OP and self.peek().value in ("*", "/"):
            op = self.next().value
            node = nodes.BinOp(op, node, self.unary())
        return node

    def unary(self):
        if self.peek().kind == OP and self.peek().value == "-":
            self.next()
            return nodes.UnaryOp("-", self.unary())
        return self.power()

    def power(self):
        node = self.primary()
        if self.peek().kind == OP and self.peek().value == "^":
            self.next()
            return nodes.BinOp("^", node, self.unary())  # right-assoc
        return node

    def primary(self):
        t = self.peek()
        if t.kind == NUMBER:
            self.next(); return nodes.Num(t.value)
        if t.kind == STRING:
            self.next(); return nodes.Str(t.value)
        if t.kind == LPAREN:
            self.next(); node = self.comparison(); self.expect(RPAREN); return node
        if t.kind == NAME:
            name = self.next().value
            if name in ("TRUE", "FALSE"):
                return nodes.Bool(name == "TRUE")
            self.expect(LPAREN)
            args = []
            if self.peek().kind != RPAREN:
                args.append(self.comparison())
                while self.peek().kind == COMMA:
                    self.next(); args.append(self.comparison())
            self.expect(RPAREN)
            return nodes.FuncCall(name, args)
        if t.kind == CELL:
            self.next()
            if self.peek().kind == COLON:
                self.next()
                end = self.expect(CELL).value
                return nodes.RangeRef(t.value, end)
            return nodes.CellRef(t.value)
        raise ValueError(f"unexpected {t.kind}")


def parse(tokens):
    return Parser(tokens).parse()
'''

FUNCTIONS = '''\
"""Built-in functions. ERROR-HANDLING POLICY (load-bearing — see git history):
the evaluator hands each function the RAW argument values, INCLUDING error cells, and each
function decides what to do with them:
  - SUM and the arithmetic operators PROPAGATE errors (any error in -> error out).
  - COUNT / AVERAGE / MIN / MAX SKIP error cells and non-numbers; they aggregate only the
    numbers present. (AVERAGE of no numbers is #DIV/0!.)
This lets COUNT(range) stay meaningful even when some cells in the range are errors."""
from .errors import is_error, DIV0, VALUE, NA


def _numbers(values):
    """Flatten args, keep only real numbers, silently dropping errors/blanks/strings."""
    out = []
    for v in values:
        if isinstance(v, list):
            out.extend(_numbers(v))
        elif isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _first_error(values):
    for v in values:
        if isinstance(v, list):
            e = _first_error(v)
            if e is not None:
                return e
        elif is_error(v):
            return v
    return None


def fn_sum(args):
    err = _first_error(args)          # SUM PROPAGATES errors
    if err is not None:
        return err
    return sum(_numbers(args))


def fn_count(args):
    return float(len(_numbers(args)))  # COUNT SKIPS errors


def fn_average(args):
    nums = _numbers(args)             # AVERAGE SKIPS errors
    if not nums:
        return DIV0
    return sum(nums) / len(nums)


def fn_min(args):
    nums = _numbers(args)
    return min(nums) if nums else DIV0

def fn_max(args):
    nums = _numbers(args)
    return max(nums) if nums else DIV0


def fn_if(args):
    cond = args[0]
    if is_error(cond):
        return cond
    return args[1] if cond else (args[2] if len(args) > 2 else False)


def fn_round(args):
    x, ndigits = args[0], int(args[1]) if len(args) > 1 else 0
    if is_error(x):
        return x
    return float(round(x, ndigits))


def fn_abs(args):
    x = args[0]
    return x if is_error(x) else abs(x)


def fn_concat(args):
    out = []
    for v in args:
        if is_error(v):
            return v
        out.append(_fmt(v))
    return "".join(out)


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


FUNCTIONS = {
    "SUM": fn_sum, "COUNT": fn_count, "AVERAGE": fn_average, "AVG": fn_average,
    "MIN": fn_min, "MAX": fn_max, "IF": fn_if, "ROUND": fn_round, "ABS": fn_abs,
    "CONCAT": fn_concat,
}
'''

# evaluator — CORRECT: does NOT short-circuit Call args; each function handles errors.
EVAL_OK = '''\
"""Evaluate an AST against a Sheet. The evaluator PROPAGATES errors through operators, but
for function calls it passes the raw argument values (errors included) to the function and
lets the function decide — it does NOT short-circuit a call just because an argument errored
(that would break COUNT/AVERAGE/MIN/MAX over ranges that contain an error cell)."""
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
            if is_error(v):
                return v
            return -v
        if isinstance(node, nodes.BinOp):
            return self.eval_binop(node)
        if isinstance(node, nodes.FuncCall):
            fn = FUNCTIONS.get(node.name)
            if fn is None:
                return NAME_ERR
            args = [self.eval(a) for a in node.args]   # NO short-circuit: pass errors through
            return fn(args)
        raise TypeError(f"cannot eval {node!r}")

    def eval_binop(self, node):
        op = node.op
        left = self.eval(node.left)
        right = self.eval(node.right)
        if is_error(left):                  # arithmetic/compare PROPAGATE errors
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

# evaluator — REGRESSION: "centralize argument evaluation" adds a short-circuit that returns
# the first error among a call's arguments BEFORE running the function. Drops the rationale.
EVAL_BAD = EVAL_OK.replace(
    '''"""Evaluate an AST against a Sheet. The evaluator PROPAGATES errors through operators, but
for function calls it passes the raw argument values (errors included) to the function and
lets the function decide — it does NOT short-circuit a call just because an argument errored
(that would break COUNT/AVERAGE/MIN/MAX over ranges that contain an error cell)."""''',
    '''"""Evaluate an AST against a Sheet. Errors propagate through operators and calls."""''',
).replace(
    '''            args = [self.eval(a) for a in node.args]   # NO short-circuit: pass errors through
            return fn(args)''',
    '''            args = [self.eval(a) for a in node.args]
            for a in args:                       # centralized: bail out early on any error arg
                if is_error(a):
                    return a
                if isinstance(a, list):
                    for x in a:
                        if is_error(x):
                            return x
            return fn(args)''',
)

ENGINE = '''\
"""The engine: parse + evaluate formulas against a sheet, with cell dependency resolution."""
from .tokens import tokenize
from .parser import parse
from .evaluator import Evaluator
from .errors import is_error, REF


class Engine:
    def __init__(self, sheet):
        self.sheet = sheet
        self._evaluating = set()   # cycle guard

    def evaluate(self, formula):
        """Evaluate a standalone formula string (may start with '=')."""
        if isinstance(formula, str) and formula.startswith("="):
            formula = formula[1:]
        ast = parse(tokenize(formula))
        return Evaluator(self.sheet, self).eval(ast)

    def cell_value(self, name):
        """The computed value of a cell: literal numbers/strings as-is, '=' formulas evaluated."""
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

INIT = '''\
"""minicalc — a tiny spreadsheet formula engine."""
from .sheet import Sheet
from .engine import Engine, evaluate
from .errors import CalcError, DIV0, REF, VALUE, NA, is_error

__all__ = ["Sheet", "Engine", "evaluate", "CalcError", "DIV0", "REF", "VALUE", "NA", "is_error"]
'''

# ── existing tests (green at HEAD: they avoid error-cells-in-aggregates) ─────
T_BASIC = '''\
from minicalc import Sheet, evaluate


def test_arithmetic_precedence():
    s = Sheet()
    assert evaluate("=2+3*4", s) == 14
    assert evaluate("=(2+3)*4", s) == 20
    assert evaluate("=2^3^2", s) == 512        # power is right-associative


def test_string_concat():
    s = Sheet()
    assert evaluate('="a"&"b"&"c"', s) == "abc"
    assert evaluate('="x"&1', s) == "x1"
'''

T_CELLS = '''\
from minicalc import Sheet, evaluate


def test_cell_refs_and_ranges():
    s = Sheet()
    s.set_many({"A1": 1, "A2": 2, "A3": 3})
    assert evaluate("=A1+A2+A3", s) == 6
    assert evaluate("=SUM(A1:A3)", s) == 6
    assert evaluate("=COUNT(A1:A3)", s) == 3
    assert evaluate("=AVERAGE(A1:A3)", s) == 2


def test_formula_cells():
    s = Sheet()
    s.set_many({"A1": 10, "A2": "=A1*2", "A3": "=A2+5"})
    assert evaluate("=A3", s) == 25
'''

T_FUNCS = '''\
from minicalc import Sheet, evaluate


def test_functions():
    s = Sheet()
    s.set_many({"A1": 5, "A2": 2, "A3": 9})
    assert evaluate("=MIN(A1:A3)", s) == 2
    assert evaluate("=MAX(A1:A3)", s) == 9
    assert evaluate("=ROUND(3.14159, 2)", s) == 3.14
    assert evaluate("=IF(A1>A2, 100, 200)", s) == 100
'''

T_ERRORS = '''\
from minicalc import Sheet, evaluate, DIV0


def test_div_zero_propagates():
    s = Sheet()
    s.set_many({"A1": 10, "A2": 0})
    assert evaluate("=A1/A2", s) == DIV0
    assert evaluate("=A1/A2 + 1", s) == DIV0      # propagates through arithmetic


def test_sum_propagates_error():
    s = Sheet()
    s.set_many({"A1": 1, "A2": "=1/0", "A3": 3})
    assert evaluate("=SUM(A1:A3)", s) == DIV0     # SUM propagates
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

    # progressive build with noise interleaved
    write("pyproject.toml", '[project]\nname = "minicalc"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# minicalc\n\nA tiny spreadsheet formula engine.\n")
    write("minicalc/__init__.py", '"""minicalc package."""\n')
    commit("scaffold minicalc package")
    write("minicalc/tokens.py", TOKENS)
    commit("tokenizer: numbers, strings, cells, operators, ranges")
    write("minicalc/nodes.py", NODES)
    commit("AST node types")
    write("minicalc/errors.py", ERRORS)
    commit("error values (#DIV/0!, #REF!, #VALUE!, #N/A)")
    write("minicalc/refs.py", REFS)
    commit("cell references and inclusive ranges")
    write("tests/test_tokens.py", "from minicalc.tokens import tokenize, NUMBER, CELL\n\n\ndef test_tokenize_basic():\n    ks = [t.kind for t in tokenize('A1+2')]\n    assert ks[:3] == ['CELL', 'OP', 'NUMBER']\n")
    commit("tokenizer tests")
    write("minicalc/parser.py", PARSER)
    commit("recursive-descent parser with operator precedence")
    write("minicalc/sheet.py", SHEET)
    commit("sheet: a grid of raw cells")
    write("minicalc/functions.py", FUNCTIONS)
    commit("built-in functions; document the error-handling policy", author="Mara K.")
    write("minicalc/evaluator.py", EVAL_OK)
    write("minicalc/engine.py", ENGINE)
    write("minicalc/__init__.py", INIT)
    commit("evaluator + engine: wire formulas to the sheet")
    write("tests/test_basic.py", T_BASIC)
    commit("tests: arithmetic precedence and concat")
    write("tests/test_cells.py", T_CELLS)
    commit("tests: cell refs, ranges, formula cells")
    write("README.md", "# minicalc\n\nA tiny spreadsheet formula engine.\n\n```python\nfrom minicalc import Sheet, evaluate\ns = Sheet(); s.set_many({'A1': 1, 'A2': 2})\nevaluate('=SUM(A1:A2)', s)\n```\n")
    commit("readme: usage example")
    write("tests/test_funcs.py", T_FUNCS)
    commit("tests: MIN/MAX/ROUND/IF")
    write("tests/test_errors.py", T_ERRORS)
    commit("tests: error propagation through arithmetic and SUM")
    # noise
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- tokenizer, parser, evaluator, engine\n")
    commit("start a changelog")
    write("minicalc/refs.py", REFS + "\n\ndef ref_in_range(name, start, end):\n    \"\"\"True if cell `name` falls within the block start..end.\"\"\"\n    return name.upper() in set(expand_range(start, end))\n")
    commit("refs: add ref_in_range helper", author="Mara K.")
    write("README.md", "# minicalc\n\nA tiny spreadsheet formula engine.\n\nSupports cell refs, ranges, the usual operators, and a set of built-in functions.\nSee tests/ for usage.\n")
    commit("readme: list capabilities")
    # THE REGRESSION — bundled in an otherwise-plausible refactor
    write("minicalc/evaluator.py", EVAL_BAD)
    commit("refactor: centralize argument evaluation in the evaluator", author="Mara K.")
    # more noise after
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- tokenizer, parser, evaluator, engine\n- evaluator refactor; misc cleanups\n")
    commit("changelog: note the evaluator refactor")
    write("minicalc/engine.py", ENGINE.replace('"""The engine: parse + evaluate formulas against a sheet, with cell dependency resolution."""',
        '"""The engine: parse + evaluate formulas against a sheet, with cell dependency resolution.\n\nUse Engine(sheet).evaluate(formula) or the module-level evaluate(formula, sheet)."""'))
    commit("engine: expand the module docstring")
    write("pyproject.toml", '[project]\nname = "minicalc"\nversion = "0.4.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.4.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
