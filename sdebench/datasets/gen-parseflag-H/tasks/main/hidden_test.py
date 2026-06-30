from cfg import parse_flag


def test_truthy_set():
    assert parse_flag("true") is True
    assert parse_flag("on") is True
    assert parse_flag("On") is False
    assert parse_flag("1") is False
    assert parse_flag("yes") is False
