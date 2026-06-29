"""Held-out (HIDDEN_TO_PASS): list values must be UNION-ed across layers (de-duplicated), not
replaced or naively appended. This policy lives only in the project's history."""
from confmerge import apply_updates


def test_list_values_union_across_layers():
    base = {"middleware": ["auth", "logging"]}
    out = apply_updates(base, {"middleware": ["cors"]})
    assert out == {"middleware": ["auth", "logging", "cors"]}   # base kept (replace would drop them)


def test_list_union_dedupes():
    base = {"middleware": ["auth", "logging"]}
    out = apply_updates(base, {"middleware": ["logging", "cors"]})
    assert out == {"middleware": ["auth", "logging", "cors"]}   # naive append would duplicate 'logging'


def test_deeply_nested_merge():
    base = {"a": {"b": {"x": 1, "y": 2}}}
    out = apply_updates(base, {"a": {"b": {"y": 3}}})
    assert out == {"a": {"b": {"x": 1, "y": 3}}}
