import pytest
from retryx import Retrier, GaveUp, TransientError


def _s(n):
    def f(a):
        if a < n:
            raise TransientError()
        return "ok"
    return f


def test_gives_up_before_nine():
    with pytest.raises(GaveUp):
        Retrier().run(_s(9))
