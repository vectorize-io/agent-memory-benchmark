"""Repro (FAIL_TO_PASS): two URLs whose query params differ only in a VALUE compare equal."""
from boltons.urlutils import URL


def test_url_query_values_matter():
    assert (URL("http://x/?a=1&b=2").query_params == {"a": "1", "b": "999"}) is False
