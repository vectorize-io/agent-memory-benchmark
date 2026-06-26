"""Held-out tests (HIDDEN_TO_PASS): the mode must be BANKER'S rounding (half-to-even).

half-up fails the *.x25/*.005 cases; half-down/truncate fail the *.x35/*.015 cases.
Only round-half-to-even passes all four.
"""
from decimal import Decimal
from ledger import round_cents


def test_half_rounds_down_to_even():
    assert round_cents("2.125") == Decimal("2.12")   # 2 is even -> down  (kills half-up)
    assert round_cents("0.005") == Decimal("0.00")   # 0 is even -> down  (kills half-up)


def test_half_rounds_up_to_even():
    assert round_cents("2.135") == Decimal("2.14")   # 3 is odd  -> up    (kills half-down/truncate)
    assert round_cents("0.015") == Decimal("0.02")   # 1 is odd  -> up    (kills half-down/truncate)
