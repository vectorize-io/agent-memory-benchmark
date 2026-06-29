"""Regression repro (FAIL_TO_PASS): a SUM over a range that contains a FORMULA cell drops it."""
from minicalc import Sheet, evaluate


def test_sum_range_with_formula_cell():
    s = Sheet()
    s.set_many({"A1": 1, "A2": "=1+1", "A3": 3})   # A2 is a formula = 2
    assert evaluate("=SUM(A1:A3)", s) == 6
