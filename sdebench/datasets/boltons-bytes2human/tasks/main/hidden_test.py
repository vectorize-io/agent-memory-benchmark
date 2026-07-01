"""Held-out (HIDDEN_TO_PASS): the rollover must be GENERAL — every exact power of 1024 advances a unit
(the documented fix used '<'). A naive special-case for 1024 passes the repro but fails here."""
from boltons.strutils import bytes2human


def test_all_exact_powers_roll_over():
    assert bytes2human(1024) == "1K"
    assert bytes2human(1024 ** 2) == "1M"
    assert bytes2human(1024 ** 3) == "1G"
    assert bytes2human(1024 ** 4) == "1T"


def test_sub_boundary_and_midrange_unchanged():
    assert bytes2human(1023) == "1023B"
    assert bytes2human(2048) == "2K"
    assert bytes2human(128991) == "126K"
