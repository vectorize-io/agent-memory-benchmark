from decimal import Decimal
from pay import round_cents


def test_matches_ledger():
    # the legacy ledger has 2.125 as 2.12; we are producing 2.13
    assert round_cents("2.125") == Decimal("2.12")
