from boltons.strutils import pluralize


def test_person_is_persons():
    assert pluralize("person") == "persons"
