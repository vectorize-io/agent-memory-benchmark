from boltons.strutils import find_hashtags


def test_drops_pure_number_tag():
    assert find_hashtags("#1 #python") == ["python"]
