from boltons.strutils import pluralize


def test_domain_plurals():
    assert pluralize("person") == "persons"
    assert pluralize("index") == "indexes"
    assert pluralize("matrix") == "matrixes"


def test_regular_words_unchanged():
    assert pluralize("cat") == "cats"
    assert pluralize("city") == "cities"
