import pytest
from retryx import Retrier, GaveUp, TransientError


def _s(n):
    def f(a):
        if a < n:
            raise TransientError()
        return "ok"
    return f


def test_seven_ok():
    assert Retrier().run(_s(7)) == "ok"


def test_eight_gives_up():
    with pytest.raises(GaveUp):
        Retrier().run(_s(8))
