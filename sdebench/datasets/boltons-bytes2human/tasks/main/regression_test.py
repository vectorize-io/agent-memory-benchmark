"""Repro (FAIL_TO_PASS): bytes2human no longer rolls over at an exact power of 1024."""
from boltons.strutils import bytes2human


def test_exact_kilobyte_rolls_over():
    assert bytes2human(1024) == "1K"
