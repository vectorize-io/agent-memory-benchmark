"""Regression repro (FAIL_TO_PASS): default-TTL entries linger far past their lifetime.

Note: this proves the entry expires too LATE — it does NOT reveal the correct lifetime.
"""
from ttlcache import TTLCache


def test_default_entry_expires_promptly():
    c = TTLCache()
    c.set("session", "data", now=0)
    # An entry created over eight minutes ago must already be evicted.
    assert c.get("session", now=500) is None
