"""Build the `billing` codebase — a LARGER repo with a LONGER, NOISIER git history.

One codebase, many commits across several features (money, discounts, tax, invoices),
with vague/realistic messages. TWO regressions are planted and bundled inside otherwise-
legit refactors, and the guarantees they break live in earlier commits — all buried in
noise so "find the relevant history" is actually exercised. Multiple sdebench TASKS are
defined on this single shared history (see tasks/).

Planted regressions (each a task):
  A) money.round_cents: should round half-cents DOWN (legacy billing) — a "tidy" commit
     switched it to half-up.
  B) invoice tax base: tax should apply to the DISCOUNTED subtotal (2019 policy) — a
     "pipeline" refactor switched it to the pre-discount subtotal.

Usage: python build.py <output_dir>   (default: /tmp/sdebench/billing)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/billing")

# ── module sources (vN = the state after the Nth change to that file) ───────────

MONEY_OK = '''\
"""Money values and cent rounding for the billing engine."""
from decimal import Decimal, ROUND_HALF_DOWN

CENT = Decimal("0.01")


def round_cents(amount):
    """Round a money amount to whole cents.

    Half-cents round DOWN (toward zero) to match the legacy billing system whose exports
    customer invoices are reconciled against. Do NOT use half-up or banker's rounding —
    either makes our totals disagree with the legacy ledger.
    """
    return Decimal(str(amount)).quantize(CENT, rounding=ROUND_HALF_DOWN)


def _amt(x):
    return x.amount if isinstance(x, Money) else Decimal(str(x))


class Money:
    """An immutable money amount, always held at cent precision."""

    __slots__ = ("amount",)

    def __init__(self, amount):
        self.amount = round_cents(amount)

    def __add__(self, other):
        return Money(self.amount + _amt(other))

    def __sub__(self, other):
        return Money(self.amount - _amt(other))

    def __mul__(self, factor):
        return Money(self.amount * Decimal(str(factor)))

    def __eq__(self, other):
        return self.amount == _amt(other)

    def __hash__(self):
        return hash(self.amount)

    def __repr__(self):
        return f"Money({self.amount})"

    def __str__(self):
        return f"${self.amount}"
'''

# Regression A: a "tidy" commit switches the rounding mode to half-up (drops the rationale).
MONEY_BAD = MONEY_OK.replace(
    "from decimal import Decimal, ROUND_HALF_DOWN", "from decimal import Decimal, ROUND_HALF_UP"
).replace(
    '''    """Round a money amount to whole cents.

    Half-cents round DOWN (toward zero) to match the legacy billing system whose exports
    customer invoices are reconciled against. Do NOT use half-up or banker's rounding —
    either makes our totals disagree with the legacy ledger.
    """
    return Decimal(str(amount)).quantize(CENT, rounding=ROUND_HALF_DOWN)''',
    '''    """Round a money amount to the nearest whole cent."""
    return Decimal(str(amount)).quantize(CENT, rounding=ROUND_HALF_UP)''',
)

DISCOUNT = '''\
"""Discounts applied to an order subtotal."""
from decimal import Decimal

from .money import Money, _amt


class Discount:
    """A percentage or fixed-amount discount."""

    def __init__(self, kind, value, label=""):
        if kind not in ("percent", "fixed"):
            raise ValueError("kind must be 'percent' or 'fixed'")
        self.kind = kind
        self.value = Decimal(str(value))
        self.label = label

    def apply(self, subtotal):
        base = _amt(subtotal)
        if self.kind == "percent":
            return Money(base * (Decimal(1) - self.value / Decimal(100)))
        return Money(base - self.value)


def stack(subtotal, discounts):
    """Apply discounts in order; each one operates on the running amount."""
    amount = Money(_amt(subtotal))
    for d in discounts:
        amount = d.apply(amount)
    return amount
'''

TAX = '''\
"""Sales tax."""
from decimal import Decimal

from .money import Money, _amt

# Default combined sales-tax rate (state + local). Overridable per invoice.
DEFAULT_RATE = Decimal("0.0725")


def tax_for(amount, rate=DEFAULT_RATE):
    """Tax owed on `amount` at `rate`."""
    return Money(_amt(amount) * Decimal(str(rate)))
'''

INVOICE_OK = '''\
"""Invoice assembly: line items -> subtotal -> discounts -> tax -> total."""
from .money import Money
from .discount import stack
from .tax import tax_for


class LineItem:
    def __init__(self, name, unit_price, qty=1):
        self.name = name
        self.unit_price = Money(unit_price)
        self.qty = int(qty)

    def line_total(self):
        return self.unit_price * self.qty


class Invoice:
    """An invoice. Tax is charged on the DISCOUNTED subtotal (policy since 2019:
    customers are taxed on what they actually pay, not the pre-discount list price)."""

    def __init__(self, items, discounts=None, tax_rate=None):
        self.items = list(items)
        self.discounts = list(discounts or [])
        self.tax_rate = tax_rate

    def subtotal(self):
        total = Money(0)
        for it in self.items:
            total = total + it.line_total()
        return total

    def discounted_subtotal(self):
        return stack(self.subtotal(), self.discounts)

    def tax(self):
        base = self.discounted_subtotal()   # taxed on what the customer actually pays
        return tax_for(base, self.tax_rate) if self.tax_rate is not None else tax_for(base)

    def total(self):
        return self.discounted_subtotal() + self.tax()
'''

# Regression B: a "pipeline" refactor taxes the pre-discount subtotal instead.
INVOICE_BAD = INVOICE_OK.replace(
    '''    def tax(self):
        base = self.discounted_subtotal()   # taxed on what the customer actually pays
        return tax_for(base, self.tax_rate) if self.tax_rate is not None else tax_for(base)''',
    '''    def tax(self):
        base = self.subtotal()
        return tax_for(base, self.tax_rate) if self.tax_rate is not None else tax_for(base)''',
).replace(
    '''    """An invoice. Tax is charged on the DISCOUNTED subtotal (policy since 2019:
    customers are taxed on what they actually pay, not the pre-discount list price)."""''',
    '''    """An invoice: items, optional discounts, optional tax rate."""''',
)

INIT = '''\
"""billing — a tiny billing engine (money, discounts, tax, invoices)."""
from .money import Money, round_cents
from .discount import Discount, stack
from .tax import tax_for, DEFAULT_RATE
from .invoice import LineItem, Invoice

__all__ = ["Money", "round_cents", "Discount", "stack", "tax_for", "DEFAULT_RATE",
           "LineItem", "Invoice"]
'''

# ── existing tests (all GREEN at HEAD — they avoid the half-cent / discounted-tax cases
#    that the planted regressions break, which is how the regressions slipped through) ──

T_MONEY = '''\
from decimal import Decimal
from billing import Money, round_cents


def test_round_non_half():
    assert round_cents("1.234") == Decimal("1.23")
    assert round_cents("1.239") == Decimal("1.24")


def test_money_arithmetic():
    assert Money("2.00") + Money("3.50") == Money("5.50")
    assert (Money("1.00") * 3) == Money("3.00")
    assert str(Money("4.2")) == "$4.20"
'''

T_DISCOUNT = '''\
from billing import Money, Discount, stack


def test_percent_discount():
    assert Discount("percent", 10).apply(Money("100.00")) == Money("90.00")


def test_fixed_and_stack():
    assert Discount("fixed", 5).apply(Money("20.00")) == Money("15.00")
    assert stack(Money("100.00"), [Discount("percent", 10), Discount("fixed", 5)]) == Money("85.00")
'''

T_TAX = '''\
from billing import Money, tax_for


def test_tax_default_rate():
    assert tax_for(Money("100.00")) == Money("7.25")


def test_tax_custom_rate():
    assert tax_for(Money("200.00"), rate="0.05") == Money("10.00")
'''

T_INVOICE = '''\
from billing import Money, LineItem, Invoice


def test_subtotal_and_total_no_discount():
    inv = Invoice([LineItem("widget", "10.00", 2), LineItem("gadget", "5.00", 1)])
    assert inv.subtotal() == Money("25.00")
    # 25.00 + 7.25% tax = 25.00 + 1.8125 -> 1.81
    assert inv.total() == Money("26.81")
'''


def main():
    if OUT.exists():
        subprocess.run(["rm", "-rf", str(OUT)], check=True)
    OUT.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=OUT, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=OUT, check=True)

    def write(path, content):
        p = OUT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    day = [1]

    def commit(msg, author="Jordan Reyes"):
        d = f"2023-{(day[0] // 28) + 1:02d}-{(day[0] % 28) + 1:02d}T09:00:00"
        day[0] += 3
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
               "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "dev@example.com",
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "dev@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    # 1 scaffold
    write("pyproject.toml", '[project]\nname = "billing"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# billing\n\nA tiny billing engine.\n")
    write("billing/__init__.py", '"""billing package."""\n')
    commit("initial project layout")
    # 2 money (correct rounding)  <-- GUARANTEE for task A
    write("billing/money.py", MONEY_OK)
    write("billing/__init__.py", '"""billing package."""\nfrom .money import Money, round_cents\n')
    commit("add Money type with cent rounding")
    # 3 noise
    write("billing/money.py", MONEY_OK)  # (no-op content; add a test)
    write("tests/test_money.py", T_MONEY)
    commit("tests for money arithmetic")
    # 4 discounts
    write("billing/discount.py", DISCOUNT)
    write("billing/__init__.py", '"""billing package."""\nfrom .money import Money, round_cents\nfrom .discount import Discount, stack\n')
    commit("discounts: percent and fixed, with stacking")
    # 5 noise
    write("tests/test_discount.py", T_DISCOUNT)
    commit("cover discount stacking")
    # 6 noise: readme
    write("README.md", "# billing\n\nA tiny billing engine: money, discounts, tax, invoices.\n\n```python\nfrom billing import Invoice, LineItem\n```\n")
    commit("flesh out the readme")
    # 7 tax
    write("billing/tax.py", TAX)
    write("billing/__init__.py", '"""billing package."""\nfrom .money import Money, round_cents\nfrom .discount import Discount, stack\nfrom .tax import tax_for, DEFAULT_RATE\n')
    commit("sales tax helper with a configurable rate", author="Priya N.")
    # 8 noise
    write("tests/test_tax.py", T_TAX)
    commit("tax tests")
    # 9 invoice (correct: tax on discounted)  <-- GUARANTEE for task B
    write("billing/invoice.py", INVOICE_OK)
    write("billing/__init__.py", INIT)
    commit("invoice assembly: subtotal, discounts, tax, total")
    # 10 noise
    write("tests/test_invoice.py", T_INVOICE)
    commit("invoice integration test")
    # 11 noise: unrelated feature
    write("README.md", "# billing\n\nA tiny billing engine: money, discounts, tax, invoices.\n\nSupports percentage and fixed discounts, configurable tax, and multi-item invoices.\n")
    commit("note supported features in readme", author="Priya N.")
    # 12 noise: start a changelog
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- Money, discounts, tax, invoices\n")
    commit("start a changelog")
    # 13 REGRESSION A (rounding) bundled in a "tidy" commit (drops the rationale)
    write("billing/money.py", MONEY_BAD)
    commit("tidy money module; normalize rounding helper", author="Priya N.")
    # 14 noise
    write("README.md", "# billing\n\nA tiny billing engine.\n\nSupports percentage and fixed discounts, configurable tax, and multi-item invoices.\nSee tests/ for usage.\n")
    commit("readme: point at tests for usage")
    # 15 noise: changelog entry (vague, unrelated-sounding)
    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- Money, discounts, tax, invoices\n- Internal cleanups to money and discount modules\n")
    commit("changelog: note recent cleanups")
    # 16 REGRESSION B (tax base) bundled in a "pipeline" refactor
    write("billing/invoice.py", INVOICE_BAD)
    commit("refactor invoice pipeline; simplify tax step", author="Priya N.")
    # 17 noise
    write("CHANGELOG.md", "# Changelog\n\n## 0.4.0\n- Money, discounts, tax, invoices\n- Internal cleanups to money and discount modules\n- Invoice pipeline refactor\n")
    commit("changelog for 0.4.0")
    # 18 release
    write("pyproject.toml", '[project]\nname = "billing"\nversion = "0.4.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.4.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
