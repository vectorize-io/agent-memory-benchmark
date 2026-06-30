from boltons.strutils import slugify


def test_ampersand_preserved():
    assert slugify("R&D", delim="-") == "r-and-d"
