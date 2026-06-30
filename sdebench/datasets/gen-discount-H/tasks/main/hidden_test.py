from billing import apply_discounts


def test_percent_before_fixed():
    assert apply_discounts(100, [("fixed", 10), ("percent", 50)]) == 40
    assert apply_discounts(100, [("percent", 50), ("fixed", 10)]) == 40


def test_compounds():
    assert apply_discounts(100, [("percent", 50), ("percent", 50)]) == 25
