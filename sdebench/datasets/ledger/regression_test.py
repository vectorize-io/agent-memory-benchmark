"""Regression repro (FAIL_TO_PASS): half-cent amounts round the wrong way vs the ledger.

Note: this shows ONE amount rounding wrong — it does NOT reveal the rounding RULE.
"""
from decimal import Decimal
from ledger import round_cents


def test_half_cent_matches_ledger():
    # the ledger has 2.125 as 2.12; we are producing 2.13
    assert round_cents("2.125") == Decimal("2.12")
