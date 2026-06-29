"""Held-out (HIDDEN_TO_PASS): parse_size must follow the SAME convention as its sibling
parse_duration — an unknown/missing unit falls back to the base unit (bytes); non-numeric is None.
(Raising, or returning None/0 on an unknown unit, fails.)"""
from parsex import parse_size


def test_unknown_unit_is_base():
    assert parse_size("10pb") == 10        # unknown unit -> bytes
    assert parse_size("512") == 512        # bare number -> bytes


def test_known_units_still_work():
    assert parse_size("2kb") == 2048


def test_non_numeric_is_none():
    assert parse_size("abc") is None
