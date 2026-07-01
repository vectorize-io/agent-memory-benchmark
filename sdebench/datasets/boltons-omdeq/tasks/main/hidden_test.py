"""Held-out (HIDDEN_TO_PASS): OrderedMultiDict.__eq__ must compare VALUES against a plain mapping, so an
OMD (on the LEFT) only equals a same-keyed dict when the values match too — reached via URL query params
and directly. (With a dict on the left, dict.__eq__ runs and masks the bug.)"""
from boltons.urlutils import URL
from boltons.dictutils import OrderedMultiDict as OMD


def test_url_query_equality():
    assert (URL("http://x/?a=1&b=2").query_params == {"a": "1", "b": "999"}) is False
    assert (URL("http://x/?a=1&b=2").query_params == {"a": "1", "b": "2"}) is True


def test_omd_compares_values():
    assert (OMD([("a", 1), ("b", 2)]) == {"a": 1, "b": 999}) is False
    assert (OMD([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}) is True
    assert (OMD([("a", 1)]) == {"a": 1, "b": 2}) is False
