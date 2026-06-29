"""Regression repro (FAIL_TO_PASS): parse_size crashes on an unknown unit instead of parsing it."""
from parsex import parse_size


def test_unknown_unit_does_not_crash():
    # '10pb' uses a unit parse_size doesn't know; it must parse leniently, not raise
    assert parse_size("10pb") == 10
