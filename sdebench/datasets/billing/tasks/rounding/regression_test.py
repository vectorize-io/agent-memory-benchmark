"""Regression repro (FAIL_TO_PASS): half-cent amounts round the wrong way vs the legacy exports."""
from decimal import Decimal
from billing import round_cents


def test_half_cent_matches_legacy():
    # legacy exports have 2.125 as 2.12; we're producing 2.13
    assert round_cents("2.125") == Decimal("2.12")
