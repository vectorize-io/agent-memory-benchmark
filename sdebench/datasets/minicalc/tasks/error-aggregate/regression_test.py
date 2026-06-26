"""Regression repro (FAIL_TO_PASS): COUNT over a range with an error cell returns the error."""
from minicalc import Sheet, evaluate


def test_count_skips_error_cells():
    s = Sheet()
    s.set_many({"A1": 1, "A2": "=1/0", "A3": 3, "A4": 4, "A5": 5})
    # COUNT should count the 4 numeric cells and ignore the #DIV/0! cell
    assert evaluate("=COUNT(A1:A5)", s) == 4
