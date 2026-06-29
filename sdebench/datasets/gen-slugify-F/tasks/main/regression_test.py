from slugkit import slugify


def test_basic_slug():
    assert slugify("Hello World") == "hello-world"
