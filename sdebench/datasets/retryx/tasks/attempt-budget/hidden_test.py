"""Held-out (HIDDEN_TO_PASS): the budget is EXACTLY the measured value — 7 attempts. 7 must
succeed, 8 must give up. (A round guess like 5/8/10 fails one side.)"""
import pytest
from retryx import Retrier, GaveUp, TransientError


def _succeeds_on(n):
    def f(attempt):
        if attempt < n:
            raise TransientError()
        return "ok"
    return f


def test_seventh_attempt_still_runs():
    assert Retrier().run(_succeeds_on(7)) == "ok"     # budget >= 7


def test_eighth_attempt_gives_up():
    with pytest.raises(GaveUp):
        Retrier().run(_succeeds_on(8))                # budget <= 7
