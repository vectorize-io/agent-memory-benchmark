from boltons.strutils import find_hashtags


def test_year_tags_kept_others_dropped():
    assert find_hashtags("#42 #data") == ["data"]
    assert find_hashtags("#2024 #launch") == ["2024", "launch"]
    assert find_hashtags("#42 #99") == []
    assert find_hashtags("#2nd") == ["2nd"]
