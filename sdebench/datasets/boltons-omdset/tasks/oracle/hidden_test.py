"""Held-out (HIDDEN_TO_PASS): __setitem__ on a multi-value key must leave EXACTLY the new value under
that key (getlist == [v]) while leaving other keys and iteration order intact — reached via URL query
params and directly on OrderedMultiDict."""
from boltons.urlutils import URL
from boltons.dictutils import OrderedMultiDict as OMD


def test_query_setitem_replaces():
    qp = URL("http://x/?a=1&a=2&a=3").query_params
    qp["a"] = "9"
    assert qp.getlist("a") == ["9"]
    assert qp["a"] == "9"


def test_omd_setitem_replaces_multivalue():
    o = OMD()
    o.add("a", 1); o.add("a", 2); o.add("b", 3)
    o["a"] = 99
    assert o.getlist("a") == [99]
    assert o.getlist("b") == [3]
    assert list(o.iteritems(multi=True)) == [("b", 3), ("a", 99)]


def test_single_value_setitem_unaffected():
    o = OMD()
    o.add("a", 1)
    o["a"] = 2
    assert o.getlist("a") == [2]
