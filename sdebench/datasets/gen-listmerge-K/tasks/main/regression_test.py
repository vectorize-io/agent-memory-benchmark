from confmerge import apply_updates


def test_nested_merge():
    out = apply_updates({"db": {"host": "h", "port": 5432}}, {"db": {"port": 5433}})
    assert out == {"db": {"host": "h", "port": 5433}}
