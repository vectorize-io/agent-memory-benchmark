from boltons.strutils import slugify


def test_symbol_abbreviations():
    assert slugify("R&D", delim="-") == "r-and-d"
    assert slugify("$5 sale", delim="-") == "usd-5-sale"
    assert slugify("50% off", delim="-") == "50-pct-off"


def test_plain_titles_unchanged():
    assert slugify("Hello World", delim="-") == "hello-world"
    assert slugify("First post! Hi!!!!~1") == "first_post_hi_1"
