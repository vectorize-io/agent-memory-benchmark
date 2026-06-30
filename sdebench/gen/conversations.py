"""Realistic verbose agent sessions (modeled on real Claude Code transcripts).

Real sessions are NOT tidy turn-pairs: the human turns are short and sparse, and between them the
assistant does a lot of inline work — reads files, runs tests, narrates output, reasons, edits —
often proposing a WRONG approach first. The load-bearing decision is buried: stated tersely by the
user ("Down."), and/or only confirmed in a long assistant turn. This is what a memory system must
summarize a feedback story out of — you cannot keep "scattered user turns" (you'd lose the decision)
nor dump the whole transcript (at real scale it's ~100k chars of tool noise).
"""

CONVERSATIONS = {
    "rounding": [
        {"role": "user", "text": "Finance says our monthly invoices don't reconcile with the legacy billing ledger — off by a cent here and there. Can you dig in?"},
        {"role": "assistant", "text": "Let me find where we round money. Reading pay/rounding.py… there's `round_cents` which quantizes to 0.01 with ROUND_HALF_UP. Let me reproduce a discrepancy: `python -c \"from pay import round_cents; print(round_cents('2.125'))\"` → prints 2.13, but the ledger shows 2.12 for that amount, so half-cents are going the wrong way. The usual fix for currency is banker's rounding (round-half-to-even) since it avoids upward bias — I'll switch ROUND_HALF_UP → ROUND_HALF_EVEN and re-run the failing test. Done: 2.125 → 2.12 now, repro passes. I'll go with banker's rounding."},
        {"role": "user", "text": "We already tried banker's last quarter — it still drifted. The mismatch is specifically on the exact half-cent values."},
        {"role": "assistant", "text": "Fair, let me not assume and actually check against the ledger rather than the textbook. Pulling a spread of half-cent values and what banker's produces: 2.125→2.12, 2.135→2.14, 0.015→0.02. If banker's still drifted at reconciliation, the ledger isn't rounding to even — it's using a single consistent direction for every half-cent. That's either always up to the next cent, or always down toward zero. I don't want to guess wrong again: for a half-cent, which way does the legacy ledger go — up or down?"},
        {"role": "user", "text": "Down."},
        {"role": "assistant", "text": "Got it — half-cents round DOWN. Setting the mode to ROUND_HALF_DOWN and verifying the full set: 2.125→2.12, 2.135→2.13, 0.015→0.01, and the existing suite still passes (14 passed). That lines up with the ledger now. So the rule for this project is: money rounds half-cents DOWN (ROUND_HALF_DOWN) to match the legacy ledger — not half-up, and not banker's."},
        {"role": "user", "text": "Perfect, that reconciles. Thanks."},
    ],
    "listmerge": [
        {"role": "user", "text": "Heads up — a config overlay change took down auth in staging. When a service sets its own middleware list, the base layer's middleware seems to vanish. Can you look at how we apply overlays?"},
        {"role": "assistant", "text": "Looking at confmerge/merge.py — `apply_updates` deep-merges dicts but for list values it just assigns the overlay's list, so on a key collision the base list is replaced wholesale. That's why base middleware disappears: the service overlay `{middleware: [cors]}` overwrites `[auth, logging]`. Quick repro confirms it. The natural fix is to combine them — I'll have it append the overlay's items onto the base list so nothing is lost. Editing… now `[auth, logging] + [cors]` → `[auth, logging, cors]`, base survives. Running the tests, green."},
        {"role": "user", "text": "Careful — we shipped append once before and it caused a different incident. On every config reload the same entries piled up, so lists grew without bound."},
        {"role": "assistant", "text": "Right, plain append isn't idempotent — reloading re-applies the overlay and duplicates accumulate. So we need to combine the lists but without repeats, and we should keep the base layer's ordering stable so reloads are deterministic. Let me reconsider the contract: keep everything the base layer set, add anything new the overlay introduces, and drop duplicates. Before I finalize — when both layers list the same item, do you want me to preserve the base position or the overlay position?"},
        {"role": "user", "text": "Base. No repeats."},
        {"role": "assistant", "text": "Implemented: for list values, union the base list with the overlay's, de-duplicated, base order first. So `[auth, logging]` overlaid with `[logging, cors]` → `[auth, logging, cors]` — no duplicate logging, auth still first, and it's stable across reloads. Full suite passes (14). The rule: config overlays UNION list values (de-duplicated, base order first), never replace and never naive-append."},
        {"role": "user", "text": "That's the one. Ship it."},
    ],
    "slugify": [
        {"role": "user", "text": "Marketing is complaining that some product pages dropped out of search after the last slug change. Mostly titles with symbols in them. Can you check slugify?"},
        {"role": "assistant", "text": "Reading slugkit/slug.py — `slugify` lowercases, then replaces any run of non-alphanumerics with a dash and trims. For 'Hello World' → 'hello-world', fine. But for a title like 'R&D Roadmap' the '&' is a non-alphanumeric, so it becomes 'r-d-roadmap'. So symbols are just being stripped. The cleanest normalization is to drop punctuation entirely and collapse separators — I'll keep that behavior but make sure we collapse cleanly. Running a few examples to confirm the trimming works."},
        {"role": "user", "text": "No — stripping the symbol is exactly the problem. 'R&D' becoming 'r-d' changes the term people actually search for. We lost the R&D page ranking because of it."},
        {"role": "assistant", "text": "I see — it's not a formatting nicety, it changes the searchable token, so dropping the character loses the keyword. So I shouldn't remove those characters; I need to preserve their meaning in the slug. For the ampersand specifically — should I leave the literal '&' in the slug (URLs with a raw ampersand are awkward), or convert it to a word?"},
        {"role": "user", "text": "Spell it out."},
        {"role": "assistant", "text": "Done — before slugifying I replace '&' with the word 'and', so 'R&D' → 'r-and-d' and 'Tom & Jerry' → 'tom-and-jerry'. Existing slugs without ampersands are unchanged, and the hidden cases pass. The rule: slugify expands '&' to 'and' (so the searchable word survives) rather than dropping the ampersand."},
        {"role": "user", "text": "Great, that should bring the pages back."},
    ],
    "budget": [
        {"role": "user", "text": "The upstream API started blocking us intermittently with 429s. I think our retry logic is too aggressive. Can you look at the retrier?"},
        {"role": "assistant", "text": "Reading retryx/retry.py — `Retrier.run` loops up to MAX_ATTEMPTS, which is currently 10, with the exponential backoff in backoff.py. So a fully-failing call makes 10 attempts; summed against our 0.1*2^n backoff that spans well past the upstream's rate-limit reset window, which is what trips the block. The safe move is to cut the budget down — I'll lower MAX_ATTEMPTS to a round 5, which comfortably fits inside the window. Editing and running the suite."},
        {"role": "user", "text": "5 is leaving retries on the table — we actually measured the window and we can fit more attempts than that before it resets."},
        {"role": "assistant", "text": "Okay, so it's not 'as low as possible' — it's the exact number of attempts whose cumulative backoff stays just under the reset window. From the backoff schedule, the cumulative wait crosses the window somewhere between the 7th and 8th attempt. So the budget should be the largest count that still fits — but I want the precise figure you measured rather than my estimate. What did the measurement land on?"},
        {"role": "user", "text": "Seven. Eight trips it."},
        {"role": "assistant", "text": "Set MAX_ATTEMPTS to 7 then — 7 attempts fit just under the rate-limit reset window, and the 8th would cross it and get us blocked. Verified the boundary in a quick simulation and the suite passes. The rule: the retry budget is exactly 7 attempts (measured to the upstream's rate-limit window); it's deliberately not a round number, so don't 'standardize' it."},
        {"role": "user", "text": "Right, leave it at 7."},
    ],
}
