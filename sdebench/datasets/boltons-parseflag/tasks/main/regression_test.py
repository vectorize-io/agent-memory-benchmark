from cfg import parse_flag


def test_yes_is_not_true():
    assert parse_flag("yes") is False
