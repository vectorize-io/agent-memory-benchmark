from confmerge import apply_updates


def test_list_union():
    assert apply_updates({"mw": ["auth", "log"]}, {"mw": ["cors"]}) == {"mw": ["auth", "log", "cors"]}
def test_list_dedup():
    assert apply_updates({"mw": ["auth", "log"]}, {"mw": ["log", "cors"]}) == {"mw": ["auth", "log", "cors"]}
def test_deep():
    assert apply_updates({"a": {"b": {"x": 1, "y": 2}}}, {"a": {"b": {"y": 3}}}) == {"a": {"b": {"x": 1, "y": 3}}}
