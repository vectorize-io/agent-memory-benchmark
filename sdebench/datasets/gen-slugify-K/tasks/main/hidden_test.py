from slugkit import slugify


def test_ampersand_to_and():
    assert slugify("Tom & Jerry") == "tom-and-jerry"
    assert slugify("R&D") == "r-and-d"


def test_collapses_separators():
    assert slugify("  Hello   World  ") == "hello-world"
