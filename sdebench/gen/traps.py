"""Trap library for the sdebench generator.

A *trap* is a non-obvious policy (against the agent's default) plus everything needed to plant it:
the buggy HEAD code, the correct code, the NAIVE guesses (which pass the repro but fail hidden),
the decision text/rationale (what gets stored in a source), and the tests. The generator
(core.py) plants any trap into any source (H git history / K doc / F conversation); the validator
(validate.py) proves the trap discriminates. Adding a trap here = N new tasks.

Each trap dict:
  name, pkg, module (path), init (package __init__), import_line
  bug      -> module source at HEAD (the regression dropped/violated the policy)
  correct  -> module source with the right policy
  naive    -> [module sources] each = a plausible guess that PASSES the repro but FAILS hidden
  decision_subject, decision_rationale -> the one-liner + why (stored in H commit / K doc / F chat)
  conversation -> turns for the F source (the user states the policy)
  existing_test, repro_test, hidden_test, bug_report
"""


def _round_mod(mode, doc):
    return (f'"""Money rounding."""\nfrom decimal import Decimal, {mode}\n\n\n'
            f'def round_cents(amount):\n    """Round an amount to whole cents.{doc}"""\n'
            f'    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding={mode})\n')


ROUNDING = {
    "name": "rounding",
    "marker": "HALF_DOWN",   # the answer-token used by the source-isolation check
    "pkg": "pay",
    "module": "pay/rounding.py",
    "init": '"""pay package."""\nfrom .rounding import round_cents\n\n__all__ = ["round_cents"]\n',
    "import_line": "from pay import round_cents",
    "bug": _round_mod("ROUND_HALF_UP", ""),
    "correct": _round_mod("ROUND_HALF_DOWN",
                          "\n\n    Half-cents round DOWN (toward zero) to match the legacy ledger; "
                          "do not use half-up or banker's rounding."),
    "naive": [_round_mod("ROUND_HALF_EVEN", "")],   # banker's: passes repro (2.125->2.12), fails 2.135
    "decision_subject": "fix: align cent rounding with the legacy billing ledger",
    "decision_rationale": ("Finance flagged a month-end reconciliation mismatch on the half-cent amounts — "
                           "the rounding mode round_cents used disagreed with the legacy ledger we reconcile "
                           "against. Adjusted the mode so the totals line up again."),
    "conversation": [
        {"role": "user", "text": "Our monthly invoice totals keep landing a cent off from the legacy billing ledger we reconcile against, and finance keeps flagging it."},
        {"role": "assistant", "text": "I'll switch round_cents to banker's rounding (ROUND_HALF_EVEN) — the usual standard for currency."},
        {"role": "user", "text": "We already tried banker's last quarter and it still drifted; the disagreement is specifically on the exact half-cent cases."},
        {"role": "assistant", "text": "Got it. For a half-cent, should I round it up to the next cent then?"},
        {"role": "user", "text": "No, the other way."},
        {"role": "assistant", "text": "Understood — round half-cents down (ROUND_HALF_DOWN), toward zero, to match the ledger."},
    ],
    "existing_test": ('from decimal import Decimal\nfrom pay import round_cents\n\n\n'
                      'def test_non_half():\n    assert round_cents("1.234") == Decimal("1.23")\n'
                      '    assert round_cents("1.239") == Decimal("1.24")\n'),
    "repro_test": ('from decimal import Decimal\nfrom pay import round_cents\n\n\n'
                   'def test_matches_ledger():\n    assert round_cents("2.125") == Decimal("2.12")\n'),
    "hidden_test": ('from decimal import Decimal\nfrom pay import round_cents\n\n\n'
                    'def test_half_down():\n    assert round_cents("2.125") == Decimal("2.12")\n'
                    '    assert round_cents("2.135") == Decimal("2.13")\n'
                    '    assert round_cents("0.015") == Decimal("0.01")\n\n\n'
                    'def test_non_half_up():\n    assert round_cents("2.137") == Decimal("2.14")\n'),
    "bug_report": ("Rounded amounts no longer match the legacy ledger. Half-cent amounts round the "
                   "wrong way — e.g. 2.125 should be 2.12 but we produce 2.13. A failing repro is at "
                   "tests/test_regression.py. Fix the rounding to match the ledger, without breaking "
                   "existing behaviour."),
}


def _merge_mod(body):
    return '"""Apply config overlays onto a base config."""\n\n\ndef apply_updates(base, updates):\n' + body


LISTMERGE = {
    "name": "listmerge",
    "marker": "union",
    "pkg": "confmerge",
    "module": "confmerge/merge.py",
    "init": '"""confmerge package."""\nfrom .merge import apply_updates\n\n__all__ = ["apply_updates"]\n',
    "import_line": "from confmerge import apply_updates",
    "bug": _merge_mod('    result = dict(base)\n    result.update(updates)\n    return result\n'),
    "correct": _merge_mod(
        '    result = dict(base)\n'
        '    for key, value in updates.items():\n'
        '        if isinstance(value, dict) and isinstance(result.get(key), dict):\n'
        '            result[key] = apply_updates(result[key], value)\n'
        '        elif isinstance(value, list) and isinstance(result.get(key), list):\n'
        '            # union with the base list (de-duplicated, base order first)\n'
        '            merged = list(result[key])\n'
        '            for item in value:\n'
        '                if item not in merged:\n'
        '                    merged.append(item)\n'
        '            result[key] = merged\n'
        '        else:\n'
        '            result[key] = value\n'
        '    return result\n'),
    "naive": [
        # deep-merge but REPLACE lists
        _merge_mod('    result = dict(base)\n'
                   '    for key, value in updates.items():\n'
                   '        if isinstance(value, dict) and isinstance(result.get(key), dict):\n'
                   '            result[key] = apply_updates(result[key], value)\n'
                   '        else:\n'
                   '            result[key] = value\n'
                   '    return result\n'),
        # deep-merge but APPEND lists (no dedup)
        _merge_mod('    result = dict(base)\n'
                   '    for key, value in updates.items():\n'
                   '        if isinstance(value, dict) and isinstance(result.get(key), dict):\n'
                   '            result[key] = apply_updates(result[key], value)\n'
                   '        elif isinstance(value, list) and isinstance(result.get(key), list):\n'
                   '            result[key] = result[key] + value\n'
                   '        else:\n'
                   '            result[key] = value\n'
                   '    return result\n'),
    ],
    "decision_subject": "fix: stop config overlays from dropping inherited list entries",
    "decision_rationale": ("A service overlay was wiping list values a base layer had set — we lost the auth "
                           "middleware in prod once. Changed how apply_updates combines list values on "
                           "conflict so entries from the base layer survive."),
    "conversation": [
        {"role": "user", "text": "When a service overlay sets a list like middleware, it's blowing away what the base config already put there — we lost auth in prod because of this."},
        {"role": "assistant", "text": "I'll make apply_updates append the overlay's list items onto the base list."},
        {"role": "user", "text": "Appending caused its own mess — on every config reload the same entries piled up as duplicates."},
        {"role": "assistant", "text": "So should I keep the base list and ignore the overlay's additions?"},
        {"role": "user", "text": "No, keep both, no repeats."},
        {"role": "assistant", "text": "Understood — union the two lists, de-duplicated, base order first."},
    ],
    "existing_test": ('from confmerge import apply_updates\n\n\n'
                      'def test_flat():\n    assert apply_updates({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}\n'
                      '    assert apply_updates({"a": 1}, {"c": 4}) == {"a": 1, "c": 4}\n'),
    "repro_test": ('from confmerge import apply_updates\n\n\n'
                   'def test_nested_merge():\n'
                   '    out = apply_updates({"db": {"host": "h", "port": 5432}}, {"db": {"port": 5433}})\n'
                   '    assert out == {"db": {"host": "h", "port": 5433}}\n'),
    "hidden_test": ('from confmerge import apply_updates\n\n\n'
                    'def test_list_union():\n'
                    '    assert apply_updates({"mw": ["auth", "log"]}, {"mw": ["cors"]}) == {"mw": ["auth", "log", "cors"]}\n'
                    'def test_list_dedup():\n'
                    '    assert apply_updates({"mw": ["auth", "log"]}, {"mw": ["log", "cors"]}) == {"mw": ["auth", "log", "cors"]}\n'
                    'def test_deep():\n'
                    '    assert apply_updates({"a": {"b": {"x": 1, "y": 2}}}, {"a": {"b": {"y": 3}}}) == {"a": {"b": {"x": 1, "y": 3}}}\n'),
    "bug_report": ("Partial config overlays clobber nested settings — apply_updates({'db': {'host': 'h', "
                   "'port': 5432}}, {'db': {'port': 5433}}) drops 'host'. Make nested overlays merge instead "
                   "of replacing. A failing repro is at tests/test_regression.py."),
}

def _slug_mod(body):
    return '"""URL slug helper."""\nimport re\n\n\ndef slugify(text):\n' + body


SLUGIFY = {
    "name": "slugify",
    "marker": "ampersand",
    "pkg": "slugkit",
    "module": "slugkit/slug.py",
    "init": '"""slugkit package."""\nfrom .slug import slugify\n\n__all__ = ["slugify"]\n',
    "import_line": "from slugkit import slugify",
    # HEAD bug: not lowercased, underscores
    "bug": _slug_mod('    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")\n'),
    # correct: lowercase + dashes + expand ampersand to "and"
    "correct": _slug_mod('    # Expand an ampersand to the word "and" (SEO: "R&D" -> "r-and-d").\n'
                         '    text = text.lower().replace("&", " and ")\n'
                         '    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")\n'),
    # naive: standard slug, drops "&"
    "naive": [_slug_mod('    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n')],
    "decision_subject": "fix: improve slug generation for titles containing symbols",
    "decision_rationale": ("Some product titles with special characters were slugging in a way that hurt "
                           "their search ranking. Adjusted how slugify handles those characters so the slugs "
                           "keep the searchable words."),
    "conversation": [
        {"role": "user", "text": "Some of our product titles are slugging in a way that tanks their search ranking — specifically the ones with special characters in them."},
        {"role": "assistant", "text": "I'll strip all non-alphanumeric characters and replace runs of them with a single dash."},
        {"role": "user", "text": "That stripping is exactly what's losing us traffic — dropping the character changes what people actually search for, like our 'R&D' page."},
        {"role": "assistant", "text": "Should I just leave the ampersand character in the slug then?"},
        {"role": "user", "text": "No, spell it out."},
        {"role": "assistant", "text": "Got it — expand '&' to the word 'and', so 'R&D' becomes 'r-and-d'."},
    ],
    "existing_test": 'from slugkit import slugify\n\n\ndef test_alnum():\n    assert slugify("abc123") == "abc123"\n',
    "repro_test": 'from slugkit import slugify\n\n\ndef test_basic_slug():\n    assert slugify("Hello World") == "hello-world"\n',
    "hidden_test": ('from slugkit import slugify\n\n\n'
                    'def test_ampersand_to_and():\n    assert slugify("Tom & Jerry") == "tom-and-jerry"\n'
                    '    assert slugify("R&D") == "r-and-d"\n\n\n'
                    'def test_collapses_separators():\n    assert slugify("  Hello   World  ") == "hello-world"\n'),
    "bug_report": ("slugify is producing bad slugs — 'Hello World' comes out as 'Hello_World' (not "
                   "lowercased, underscores instead of dashes). Make it produce clean URL slugs. A "
                   "failing repro is at tests/test_regression.py."),
}

_RETRY_CORE = ('class TransientError(Exception):\n    pass\n\n\nclass GaveUp(Exception):\n    pass\n\n\n'
               'class Retrier:\n    def run(self, func):\n        last = None\n'
               '        for attempt in range(1, MAX_ATTEMPTS + 1):\n'
               '            try:\n                return func(attempt)\n'
               '            except TransientError as exc:\n                last = exc\n'
               '        raise GaveUp("exhausted") from last\n')


def _retry_mod(maxv, comment=""):
    return f'"""Bounded retry."""\n{comment}MAX_ATTEMPTS = {maxv}\n\n\n' + _RETRY_CORE


_RETRY_HELPER = ('def _s(n):\n    def f(a):\n        if a < n:\n            raise TransientError()\n'
                 '        return "ok"\n    return f\n')

BUDGET = {
    "name": "budget",
    "marker": "rate-limit",
    "pkg": "retryx",
    "module": "retryx/retry.py",
    "init": ('"""retryx package."""\nfrom .retry import Retrier, MAX_ATTEMPTS, TransientError, GaveUp\n\n'
             '__all__ = ["Retrier", "MAX_ATTEMPTS", "TransientError", "GaveUp"]\n'),
    "import_line": "from retryx import Retrier",
    "bug": _retry_mod(10),
    "correct": _retry_mod(7, "# 7 is measured: at our backoff it spans just under the upstream's rate-limit\n"
                             "# reset window; 8+ trips it and the upstream blocks us. Do not round.\n"),
    "naive": [_retry_mod(8), _retry_mod(5)],
    "decision_subject": "fix: stop the retrier from tripping the upstream rate limit",
    "decision_rationale": ("Our retry loop was hammering the upstream past its rate-limit reset window and "
                           "getting us blocked. Tuned the attempt budget so a full set of retries fits "
                           "inside the window."),
    "conversation": [
        {"role": "user", "text": "The upstream keeps rate-limiting us — our retrier is clearly making too many attempts and crossing their reset window."},
        {"role": "assistant", "text": "I'll lower MAX_ATTEMPTS to a round 5 to be safe."},
        {"role": "user", "text": "5 is too conservative — we actually measured the window and we can fit more attempts than that before it resets."},
        {"role": "assistant", "text": "So should I set it to 10?"},
        {"role": "user", "text": "No, fewer than that."},
        {"role": "assistant", "text": "Understood — exactly 7 attempts; the 8th trips the window."},
    ],
    "existing_test": ('from retryx import Retrier, TransientError\n\n\n' + _RETRY_HELPER +
                      '\n\ndef test_succeeds():\n    assert Retrier().run(lambda a: "ok") == "ok"\n'
                      '    assert Retrier().run(_s(3)) == "ok"\n'),
    "repro_test": ('import pytest\nfrom retryx import Retrier, GaveUp, TransientError\n\n\n' + _RETRY_HELPER +
                   '\n\ndef test_gives_up_before_nine():\n    with pytest.raises(GaveUp):\n        Retrier().run(_s(9))\n'),
    "hidden_test": ('import pytest\nfrom retryx import Retrier, GaveUp, TransientError\n\n\n' + _RETRY_HELPER +
                    '\n\ndef test_seven_ok():\n    assert Retrier().run(_s(7)) == "ok"\n\n\n'
                    'def test_eight_gives_up():\n    with pytest.raises(GaveUp):\n        Retrier().run(_s(8))\n'),
    "bug_report": ("The upstream is rate-limiting us — our Retrier retries well past where it should stop "
                   "and crosses the rate-limit window. Restore the intended attempt budget. A failing "
                   "repro is at tests/test_regression.py."),
}

TRAPS = {t["name"]: t for t in [ROUNDING, LISTMERGE, SLUGIFY, BUDGET]}

from conversations import CONVERSATIONS
import json as _json
from pathlib import Path as _Path
_sf = _Path(__file__).resolve().parent / "sessions.json"   # long LLM-generated realistic sessions
_SESS = _json.loads(_sf.read_text()) if _sf.exists() else {}
for _n in TRAPS:
    TRAPS[_n]["conversation"] = _SESS.get(_n) or CONVERSATIONS.get(_n)
