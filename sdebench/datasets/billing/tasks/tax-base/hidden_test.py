"""Held-out (HIDDEN_TO_PASS): tax is charged on the DISCOUNTED subtotal, for any discount."""
from billing import Money, LineItem, Invoice, Discount


def test_percent_discount_tax_base():
    inv = Invoice([LineItem("w", "100.00", 1)], [Discount("percent", 20)])  # -> 80
    assert inv.tax() == Money("5.80")
    assert inv.total() == Money("85.80")


def test_fixed_discount_tax_base():
    inv = Invoice([LineItem("w", "100.00", 2)], [Discount("fixed", "40.00")])  # 200 -> 160
    assert inv.tax() == Money("11.60")
    assert inv.total() == Money("171.60")
