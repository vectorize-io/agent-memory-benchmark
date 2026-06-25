"""Regression repro filed with the bug report (FAIL_TO_PASS).

Bug: with a FRACTIONAL rate, the bucket never refills — partial tokens are dropped.
This test is RED on the broken HEAD and GREEN once the refill floor is removed.
"""
from ratelimiter import TokenBucket


def test_partial_tokens_carry_over():
    b = TokenBucket(rate=0.5, capacity=1)
    assert b.try_acquire(1, now=0) is True       # consume the one token
    assert b.try_acquire(1, now=0) is False       # empty
    # at 0.5 tokens/sec, half a token has accrued after 1s ...
    assert b.available(now=1) == 0.5
    # ... and a whole token after 2s, so a request now succeeds
    assert b.try_acquire(1, now=2) is True
