"""Regression repro (FAIL_TO_PASS): tax on a discounted invoice is wrong."""
from billing import Money, LineItem, Invoice, Discount


def test_tax_on_discounted_invoice():
    inv = Invoice([LineItem("widget", "100.00", 1)], [Discount("percent", 20)])
    # 20% off 100 = 80; tax should be charged on 80 (7.25% = 5.80)
    assert inv.tax() == Money("5.80")
