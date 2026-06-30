"""Real-function host traps: a non-guessable policy planted on an EDGE of a REAL boltons function,
graded against boltons' REAL test suite (pass_to_pass). F source — the policy lives only in a chat.

The real boltons code IS the "bug" (it doesn't implement the project's policy); the agent must add
the policy. `correct`/`naive` are (anchor, replacement) transforms used ONLY to VALIDATE discrimination
(real tests green at HEAD + after; HEAD fails repro+hidden; naive passes repro but fails hidden).

Each trap is on `boltons.strutils`, so pass_to_pass = the real tests/test_strutils.py.
"""

REALFN_TRAPS = {
    # ── pluralize: project uses formal/DB plural forms, not boltons' defaults ──────────────────
    "pluralize": {
        "module": "boltons/strutils.py",
        "keep_tests": ["test_strutils.py"],
        "import": "from boltons.strutils import pluralize",
        "anchor": "    orig_word, word = word, word.strip().lower()\n    if not word or word in _IRR_P2S:",
        "correct": ('    _OVR = {"person": "persons", "index": "indexes", "matrix": "matrixes"}\n'
                    '    if word.strip().lower() in _OVR:\n        return _OVR[word.strip().lower()]\n'
                    '    orig_word, word = word, word.strip().lower()\n    if not word or word in _IRR_P2S:'),
        "naive": ('    _OVR = {"person": "persons"}\n'
                  '    if word.strip().lower() in _OVR:\n        return _OVR[word.strip().lower()]\n'
                  '    orig_word, word = word, word.strip().lower()\n    if not word or word in _IRR_P2S:'),
        "repro_test": ('from boltons.strutils import pluralize\n\n\n'
                       'def test_person_is_persons():\n    assert pluralize("person") == "persons"\n'),
        "hidden_test": ('from boltons.strutils import pluralize\n\n\n'
                        'def test_domain_plurals():\n    assert pluralize("person") == "persons"\n'
                        '    assert pluralize("index") == "indexes"\n    assert pluralize("matrix") == "matrixes"\n\n\n'
                        'def test_regular_words_unchanged():\n    assert pluralize("cat") == "cats"\n'
                        '    assert pluralize("city") == "cities"\n'),
        "bug_report": ("Our records use the wrong plural forms for several domain terms. pluralize('person') "
                       "returns 'people' but our schema and legal docs say 'persons', and the same mismatch hits "
                       "a few other terms. Make pluralize produce the project's standard plural forms. A failing "
                       "repro is at tests/test_regression.py."),
        "conversation": [
            {"role": "user", "text": "Our exports have inconsistent plurals — pluralize('person') gives 'people' but the data schema and the legal team both require 'persons'. A few other terms are off too. Can you align the helper with our standard?"},
            {"role": "assistant", "text": "boltons.pluralize uses an irregulars table — 'person' -> 'people'. I can add a project override for 'person' -> 'persons'. Are there specific other terms?"},
            {"role": "user", "text": "Yes — 'index' must be 'indexes' (we mean DB indexes, not 'indices'), and 'matrix' must be 'matrixes', not 'matrices'. Those three are the ones our schema uses."},
            {"role": "assistant", "text": "Understood — a small override map: person->persons, index->indexes, matrix->matrixes, applied before boltons' normal logic. Everything else (cat->cats, city->cities) stays as-is."},
            {"role": "user", "text": "Exactly those three. Don't touch the rest."},
        ],
    },
    # ── under2camel: restore known acronyms to uppercase instead of Title-casing ────────────────
    "under2camel": {
        "module": "boltons/strutils.py",
        "keep_tests": ["test_strutils.py"],
        "import": "from boltons.strutils import under2camel",
        "anchor": "    return ''.join(w.capitalize() or '_' for w in under_string.split('_'))",
        "correct": ('    _ACR = {"http": "HTTP", "https": "HTTPS", "url": "URL", "api": "API", "io": "IO", "id": "ID"}\n'
                    "    return ''.join(_ACR.get(w, w.capitalize()) or '_' for w in under_string.split('_'))"),
        "naive": ('    _ACR = {"http": "HTTP"}\n'
                  "    return ''.join(_ACR.get(w, w.capitalize()) or '_' for w in under_string.split('_'))"),
        "repro_test": ('from boltons.strutils import under2camel\n\n\n'
                       'def test_http_acronym():\n    assert under2camel("http_response") == "HTTPResponse"\n'),
        "hidden_test": ('from boltons.strutils import under2camel\n\n\n'
                        'def test_acronyms():\n    assert under2camel("http_response") == "HTTPResponse"\n'
                        '    assert under2camel("api_key") == "APIKey"\n    assert under2camel("parse_url") == "ParseURL"\n'
                        '    assert under2camel("user_id") == "UserID"\n\n\n'
                        'def test_non_acronyms_unchanged():\n    assert under2camel("user_name") == "UserName"\n'
                        '    assert under2camel("complex_tokenizer") == "ComplexTokenizer"\n'),
        "bug_report": ("under2camel mangles our acronyms. 'http_response' becomes 'HttpResponse' instead of "
                       "'HTTPResponse', so generated class names don't match our code style. Make under2camel keep "
                       "the project's known acronyms uppercase. A failing repro is at tests/test_regression.py."),
        "conversation": [
            {"role": "user", "text": "The code generator produces class names like 'HttpResponse' and 'ApiKey' from snake_case, but our codebase style is 'HTTPResponse', 'APIKey'. under2camel is title-casing the acronyms. Can you fix it to keep our acronyms uppercase?"},
            {"role": "assistant", "text": "under2camel just capitalizes each underscore-separated part, so 'http' -> 'Http'. I can special-case acronyms. Which ones does your style guide list?"},
            {"role": "user", "text": "HTTP, HTTPS, URL, API, IO, and ID. Those stay all-caps. Everything else title-cases normally."},
            {"role": "assistant", "text": "Got it — a known-acronym set {HTTP, HTTPS, URL, API, IO, ID}; if a part matches, emit it uppercase, otherwise capitalize as before. So 'user_id' -> 'UserID', 'parse_url' -> 'ParseURL', 'user_name' -> 'UserName'."},
            {"role": "user", "text": "Perfect, exactly those six."},
        ],
    },
    # ── find_hashtags: drop all-numeric hashtags (not just leading-digit) ───────────────────────
    "findhashtags": {
        "module": "boltons/strutils.py",
        "keep_tests": ["test_strutils.py"],
        "import": "from boltons.strutils import find_hashtags",
        "anchor": "    return HASHTAG_RE.findall(string)",
        "correct": "    return [h for h in HASHTAG_RE.findall(string) if not h.isdigit()]",
        "naive": "    return [h for h in HASHTAG_RE.findall(string) if not h[:1].isdigit()]",
        "repro_test": ('from boltons.strutils import find_hashtags\n\n\n'
                       'def test_drops_pure_number_tag():\n    assert find_hashtags("#1 #python") == ["python"]\n'),
        "hidden_test": ('from boltons.strutils import find_hashtags\n\n\n'
                        'def test_only_all_numeric_dropped():\n'
                        '    assert find_hashtags("#42 #data") == ["data"]\n'
                        '    assert find_hashtags("#2nd #ai2 #2024") == ["2nd", "ai2"]\n'),
        "bug_report": ("Our trending feed is full of junk tags like #1 and #2024 — find_hashtags is returning "
                       "pure-number hashtags, which aren't real tags on our platform. Make find_hashtags drop the "
                       "invalid ones. A failing repro is at tests/test_regression.py."),
        "conversation": [
            {"role": "user", "text": "The trending hashtags list is polluted with numbers — '#1', '#2024', '#42' keep showing up. Those aren't valid tags for us. Can you filter them out in find_hashtags?"},
            {"role": "assistant", "text": "I can filter the results. The simplest rule is to drop any hashtag that starts with a digit — that removes '#1', '#2024', '#42'."},
            {"role": "user", "text": "No, careful — tags like '#2nd' and '#ai2' are legit and some start with a digit. The rule is: drop a tag only if it's ALL digits. '#42' goes, '#2nd' stays."},
            {"role": "assistant", "text": "Understood — drop a hashtag only when it is entirely numeric (all digits); anything with a letter stays. So '#42' and '#2024' are dropped, '#2nd' and '#ai2' are kept."},
            {"role": "user", "text": "Right — all-digits only."},
        ],
    },
    "slugify": {'module': 'boltons/strutils.py', 'keep_tests': ['test_strutils.py'], 'import': 'from boltons.strutils import slugify', 'anchor': "    ret = delim.join(split_punct_ws(text)) or delim if text else ''", 'correct': "    for _s, _w in (('&', ' and '), ('$', ' usd '), ('%', ' pct ')):\n        text = text.replace(_s, _w)\n    ret = delim.join(split_punct_ws(text)) or delim if text else ''", 'naive': "    for _s, _w in (('&', ' and '), ('$', ' dollar '), ('%', ' percent ')):\n        text = text.replace(_s, _w)\n    ret = delim.join(split_punct_ws(text)) or delim if text else ''", 'repro_test': 'from boltons.strutils import slugify\n\n\ndef test_ampersand_preserved():\n    assert slugify("R&D", delim="-") == "r-and-d"\n', 'hidden_test': 'from boltons.strutils import slugify\n\n\ndef test_symbol_abbreviations():\n    assert slugify("R&D", delim="-") == "r-and-d"\n    assert slugify("$5 sale", delim="-") == "usd-5-sale"\n    assert slugify("50% off", delim="-") == "50-pct-off"\n\n\ndef test_plain_titles_unchanged():\n    assert slugify("Hello World", delim="-") == "hello-world"\n    assert slugify("First post! Hi!!!!~1") == "first_post_hi_1"\n', 'bug_report': "slugify is mangling product titles that contain symbols. 'R&D' slugifies to 'r-d' — the '&' is dropped entirely, which changed the search term and tanked the page's ranking. Symbols in titles (&, $, %) need to be preserved as the team's standard words in the slug, not stripped. A failing repro is at tests/test_regression.py. Fix boltons.strutils.slugify without breaking its existing behaviour.", 'conversation': [{'role': 'user', 'text': "Marketing flagged that several product pages with symbols in the title dropped out of search. slugify is stripping the symbols — 'R&D' becomes 'r-d'. We need the symbols to come through as words."}, {'role': 'assistant', 'text': "Looking at boltons/strutils.py — slugify calls split_punct_ws which discards punctuation, so '&', '$', '%' all vanish. Reproduced 'R&D' -> 'r-d'. I'll expand them to words before the split: '&' -> 'and', '$' -> 'dollar', '%' -> 'percent'. That gives 'r-and-d', 'usd'... wait, what wording do you want for currency and percent?"}, {'role': 'user', 'text': "Not 'dollar' or 'percent' — we standardized on our analytics tags. '$' becomes 'usd' and '%' becomes 'pct'. So '$5 sale' -> 'usd-5-sale' and '50% off' -> '50-pct-off'. The ampersand is 'and' as you said."}, {'role': 'assistant', 'text': "Got it — the map is '&' -> 'and', '$' -> 'usd', '%' -> 'pct', matching the analytics tags, not the natural English words. Only those three symbols; leave the rest of the punctuation handling as-is."}, {'role': 'user', 'text': 'Right. usd and pct, exactly. Ship it.'}, {'role': 'assistant', 'text': "Done — slugify expands & -> and, $ -> usd, % -> pct before the existing split, so 'R&D' -> 'r-and-d', '$5 sale' -> 'usd-5-sale', '50% off' -> '50-pct-off'. Plain titles are unchanged and the strutils tests still pass."}]}
}
