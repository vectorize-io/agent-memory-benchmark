"""Held-out (HIDDEN_TO_PASS): the policy is that COUNT/AVERAGE/MIN/MAX SKIP error cells while
SUM and the arithmetic operators still PROPAGATE them. A COUNT-only fix fails the aggregates;
a 'skip errors everywhere' fix fails SUM/arithmetic. Only the correct general fix passes."""
from minicalc import Sheet, evaluate, DIV0


def _s():
    s = Sheet()
    s.set_many({"A1": 1, "A2": "=1/0", "A3": 3, "A4": 4, "A5": 5})
    return s


def test_average_skips_errors():
    assert evaluate("=AVERAGE(A1:A5)", _s()) == 3.25   # mean of 1, 3, 4, 5


def test_min_max_skip_errors():
    assert evaluate("=MIN(A1:A5)", _s()) == 1
    assert evaluate("=MAX(A1:A5)", _s()) == 5


def test_sum_still_propagates():
    assert evaluate("=SUM(A1:A5)", _s()) == DIV0       # SUM must STILL propagate the error


def test_arithmetic_still_propagates():
    s = Sheet(); s.set_many({"A1": 10, "A2": 0})
    assert evaluate("=A1/A2 + 5", s) == DIV0
