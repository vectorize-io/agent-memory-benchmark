"""Held-out (HIDDEN_TO_PASS): the mode must be ROUND_HALF_DOWN (ties toward zero)."""
from decimal import Decimal
from billing import round_cents


def test_half_cents_round_toward_zero():
    assert round_cents("2.125") == Decimal("2.12")   # tie -> down
    assert round_cents("2.135") == Decimal("2.13")   # tie -> down (banker's/half-up give 2.14)
    assert round_cents("0.015") == Decimal("0.01")   # tie -> down


def test_non_half_rounds_normally():
    assert round_cents("2.137") == Decimal("2.14")   # 7 > 5 -> up (truncation gives 2.13)
