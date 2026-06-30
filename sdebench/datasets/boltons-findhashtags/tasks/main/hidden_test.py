from boltons.strutils import find_hashtags


def test_only_all_numeric_dropped():
    assert find_hashtags("#42 #data") == ["data"]
    assert find_hashtags("#2nd #ai2 #2024") == ["2nd", "ai2"]
