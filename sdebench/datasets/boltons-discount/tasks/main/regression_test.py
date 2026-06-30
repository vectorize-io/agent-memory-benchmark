from billing import apply_discounts


def test_two_percent_compound():
    assert apply_discounts(100, [("percent", 50), ("percent", 50)]) == 25
