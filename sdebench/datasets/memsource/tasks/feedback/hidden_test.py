from decimal import Decimal
from pay import round_cents


def test_half_down():
    assert round_cents("2.125") == Decimal("2.12")   # tie -> down
    assert round_cents("2.135") == Decimal("2.13")   # tie -> down (banker's/half-up give 2.14)
    assert round_cents("0.015") == Decimal("0.01")


def test_non_half():
    assert round_cents("2.137") == Decimal("2.14")   # 7>5 -> up (truncation gives 2.13)
