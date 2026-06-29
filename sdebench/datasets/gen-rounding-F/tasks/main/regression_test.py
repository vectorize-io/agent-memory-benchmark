from decimal import Decimal
from pay import round_cents


def test_matches_ledger():
    assert round_cents("2.125") == Decimal("2.12")
