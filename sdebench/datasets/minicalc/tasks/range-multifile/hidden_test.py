"""Held-out (HIDDEN_TO_PASS): ranges must use COMPUTED cell values everywhere. The bug was
duplicated in two code paths — the evaluator (used by SUM/COUNT/...) and Engine.range_values.
A fix to only one path leaves the other returning raw, uncomputed cell contents."""
from minicalc import Sheet, evaluate, Engine


def test_engine_range_values_computes_formulas():     # the engine.py path
    s = Sheet()
    s.set_many({"A1": 1, "A2": "=1+1", "A3": 3})
    assert Engine(s).range_values("A1", "A3") == [1, 2, 3]


def test_aggregates_over_formula_cells():             # the evaluator path, other functions
    s = Sheet()
    s.set_many({"A1": 10, "A2": "=A1/2", "A3": "=A1*2"})  # 10, 5, 20
    assert evaluate("=COUNT(A1:A3)", s) == 3
    assert evaluate("=MAX(A1:A3)", s) == 20
    assert evaluate("=AVERAGE(A1:A3)", s) == 35 / 3


def test_two_d_range_with_formula():
    s = Sheet()
    s.set_many({"A1": 1, "B1": "=1+1", "A2": 3, "B2": "=2+2"})  # 1,2,3,4
    assert evaluate("=SUM(A1:B2)", s) == 10
