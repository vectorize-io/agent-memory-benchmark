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
    "decision_subject": "round_cents rounds half-cents DOWN to match the legacy ledger",
    "decision_rationale": ("Money rounds half-cents DOWN (ROUND_HALF_DOWN), always — half-up and "
                           "banker's rounding both drift our totals away from the legacy ledger we "
                           "reconcile against; finance flagged a discrepancy from exactly this."),
    "conversation": [
        {"role": "user", "text": "Add a discount to the invoice total and get the cents right."},
        {"role": "assistant", "text": "Done — I used banker's rounding (ROUND_HALF_EVEN), the money standard."},
        {"role": "user", "text": "No, not banker's. We round half-cents DOWN (ROUND_HALF_DOWN) here, always — "
                                 "banker's and half-up drift our totals off the legacy ledger; finance flagged "
                                 "a 3-cent discrepancy last quarter from exactly this. Use half-down for money."},
        {"role": "assistant", "text": "Understood — ROUND_HALF_DOWN for all money rounding in this project."},
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
    "decision_subject": "apply_updates: union list values instead of replacing them",
    "decision_rationale": ("When an overlay sets a list, UNION it with the base list (de-duplicated, "
                           "base order first) — never replace or naively append. Replacing dropped "
                           "middleware that a base layer had installed (disabling auth in prod); "
                           "appending duplicated entries on every reload."),
    "conversation": [
        {"role": "user", "text": "Make config overlays merge nested settings properly."},
        {"role": "assistant", "text": "Done — deep-merge for dicts, and I replace list values on conflict."},
        {"role": "user", "text": "Don't replace lists — union them (deduped). Replacing dropped the "
                                 "middleware the base layer set and disabled auth in prod once. Always union lists."},
        {"role": "assistant", "text": "Understood — union list values (de-duplicated), never replace or append."},
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

TRAPS = {t["name"]: t for t in [ROUNDING, LISTMERGE]}
