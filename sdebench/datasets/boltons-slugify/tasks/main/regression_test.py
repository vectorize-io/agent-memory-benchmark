"""Repro (FAIL_TO_PASS): slugify drops the ampersand instead of preserving the searchable word."""
from boltons.strutils import slugify


def test_ampersand_preserved():
    assert slugify("R&D", delim="-") == "r-and-d"
