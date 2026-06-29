"""Regression repro (FAIL_TO_PASS): a nested overlay clobbers sibling keys."""
from confmerge import apply_updates


def test_nested_overlay_preserves_siblings():
    base = {"db": {"host": "h", "port": 5432}}
    out = apply_updates(base, {"db": {"port": 5433}})
    assert out == {"db": {"host": "h", "port": 5433}}   # host must be preserved
