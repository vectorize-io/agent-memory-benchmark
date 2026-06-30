from cfg import parse_flag


def test_case_sensitive():
    assert parse_flag("true") is True
    assert parse_flag("True") is False
    assert parse_flag("TRUE") is False
    assert parse_flag("1") is False
