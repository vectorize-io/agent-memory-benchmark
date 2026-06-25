"""Held-out tests (HIDDEN_TO_PASS) — never shown to the agent.

Same behaviour (partial-token refill) exercised with different rates/intervals and
through RateLimiter, so a fix that special-cases the visible repro's inputs still fails.
"""
from ratelimiter import TokenBucket, RateLimiter


def test_quarter_rate_accrues():
    b = TokenBucket(rate=0.25, capacity=2)
    b.try_acquire(2, now=0)
    assert b.available(now=2) == 0.5          # 0.25 * 2s
    assert b.try_acquire(1, now=4) is True     # 0.25 * 4s = 1.0


def test_fractional_partial_cost():
    b = TokenBucket(rate=0.1, capacity=1)
    b.try_acquire(1, now=0)
    assert b.try_acquire(0.5, now=5) is True   # 0.1 * 5s = 0.5
    assert b.try_acquire(0.1, now=5) is False


def test_ratelimiter_fractional_rate():
    rl = RateLimiter(rate=0.5, capacity=1)
    assert rl.try_acquire("u", now=0) is True
    rl.try_acquire("u", now=1)                  # a partial refill step (0.5 should accrue)
    # second 0.5 accrues over the next second -> 1.0 total; int() floors each step to 0
    assert rl.try_acquire("u", now=2) is True
