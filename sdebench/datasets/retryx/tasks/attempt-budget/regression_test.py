"""Regression repro (FAIL_TO_PASS): the retrier over-retries past the budget it must respect."""
import pytest
from retryx import Retrier, GaveUp, TransientError


def _succeeds_on(n):
    def f(attempt):
        if attempt < n:
            raise TransientError()
        return "ok"
    return f


def test_gives_up_within_budget():
    # an operation that would only succeed on attempt 9 must be abandoned — retrying that far
    # trips the upstream's rate limit. The retrier must give up before reaching it.
    with pytest.raises(GaveUp):
        Retrier().run(_succeeds_on(9))
