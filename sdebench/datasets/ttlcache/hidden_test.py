"""Held-out tests (HIDDEN_TO_PASS): the default lifetime must be EXACTLY 300s.

A fix that merely makes the repro pass (any TTL <= ~480) fails here unless it is 300.
"""
from ttlcache import TTLCache


def test_default_ttl_exact_boundary():
    c = TTLCache()
    c.set("k", "v", now=0)
    assert c.get("k", now=286) == "v"     # valid just before expiry
    assert c.get("k", now=287) is None    # expired exactly at 287s


def test_default_ttl_relative_to_set_time():
    c = TTLCache()
    c.set("k", "v", now=1000)
    assert c.get("k", now=1286) == "v"
    assert c.get("k", now=1287) is None
