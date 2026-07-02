"""Repro (FAIL_TO_PASS): assigning a query param replaces its values, but getlist keeps the old ones."""
from boltons.urlutils import URL


def test_query_setitem_replaces_values():
    qp = URL("http://x/?a=1&a=2").query_params
    qp["a"] = "9"
    assert qp.getlist("a") == ["9"]
