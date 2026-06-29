"""Build `confmerge` — a config-overlay library, for the K (conventions/decisions) task.

FULL history, no ablation. The current code regressed to a SHALLOW merge (nested dicts get
clobbered — the reported bug). The obvious fix is a deep-merge, which passes the repro. But the
project made a NON-OBVIOUS, documented decision about LIST values: they must be UNION-ed (merged,
de-duplicated), NOT replaced — because replacing dropped middleware that base layers had added.
That decision lives only in a real commit's message/rationale in the history.

So a naive agent deep-merges and *replaces* lists (or appends with duplicates) → passes the repro
but FAILS the hidden list test. A "smart" agent that reads `git log` on merge.py sees that
replacing lists was tried and rejected in favour of union → gets it right. The memory is fully
present (full git history); the task tests whether the agent consults it.

Usage: python build.py <output_dir>   (default: /tmp/sdebench/confmerge)
"""
import os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdebench/confmerge")

INIT = '"""confmerge — layer config overlays onto a base config."""\nfrom .merge import apply_updates\nfrom .loader import load_overlay\n\n__all__ = ["apply_updates", "load_overlay"]\n'

LOADER = '''\
"""Tiny helper to turn a flat `key=value` overlay string into a dict (numbers coerced)."""


def load_overlay(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        try:
            value = int(value)
        except ValueError:
            pass
        out[key.strip()] = value
    return out
'''

# v1: deep-merge nested dicts, but REPLACE list values on conflict.
MERGE_REPLACE = '''\
"""Apply config overlays onto a base config."""


def apply_updates(base, updates):
    """Return a new config with `updates` overlaid onto `base`. Nested dicts are deep-merged;
    list values are replaced."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = apply_updates(result[key], value)
        else:
            result[key] = value
    return result
'''

# v2 (the DECISION): union list values instead of replacing them. Rationale in the commit msg.
MERGE_UNION = '''\
"""Apply config overlays onto a base config."""


def apply_updates(base, updates):
    """Return a new config with `updates` overlaid onto `base`.

    Nested dicts are deep-merged. List values are UNION-ed with the base list (de-duplicated,
    base order preserved) rather than replaced, so an overlay never silently drops list entries
    that an earlier layer added (e.g. middleware). Pass a brand-new key to set a list outright.
    """
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = apply_updates(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list):
            merged = list(result[key])
            for item in value:
                if item not in merged:
                    merged.append(item)
            result[key] = merged
        else:
            result[key] = value
    return result
'''

# v3 (the REGRESSION at HEAD): "streamlined" to a shallow update — nested dicts get clobbered.
MERGE_SHALLOW = '''\
"""Apply config overlays onto a base config."""


def apply_updates(base, updates):
    """Return a new config with `updates` overlaid onto `base`."""
    result = dict(base)
    result.update(updates)
    return result
'''

# existing tests: only FLAT overlays (green under every version — which is how the regression
# slipped past them; nested-merge and list-union were never covered by a committed test).
T_BASIC = '''\
from confmerge import apply_updates


def test_flat_overlay():
    assert apply_updates({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_adds_new_key():
    assert apply_updates({"a": 1}, {"c": 4}) == {"a": 1, "c": 4}
'''

T_LOADER = '''\
from confmerge import load_overlay


def test_load_overlay():
    assert load_overlay("a=1\\nb = hello\\n# comment\\n") == {"a": 1, "b": "hello"}
'''


def main():
    if OUT.exists():
        subprocess.run(["rm", "-rf", str(OUT)], check=True)
    OUT.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=OUT, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=OUT, check=True)

    day = [1]

    def write(path, content):
        p = OUT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def commit(msg, author="Dana Ops"):
        d = f"2024-04-{day[0]:02d}T10:00:00"
        day[0] += 1
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d,
               "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "dev@example.com",
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "dev@example.com"}
        subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=OUT, env=env, check=True)

    write("pyproject.toml", '[project]\nname = "confmerge"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n')
    write("README.md", "# confmerge\n\nLayer config overlays onto a base config.\n")
    write("confmerge/__init__.py", '"""confmerge package."""\n')
    commit("scaffold confmerge package")

    write("confmerge/loader.py", LOADER)
    write("confmerge/__init__.py", '"""confmerge package."""\nfrom .loader import load_overlay\n')
    commit("loader: parse flat key=value overlays")
    write("tests/test_loader.py", T_LOADER)
    commit("tests for the overlay loader")

    write("confmerge/merge.py", MERGE_REPLACE)
    write("confmerge/__init__.py", INIT)
    commit("apply_updates: deep-merge nested config (lists are replaced)")
    write("tests/test_merge.py", T_BASIC)
    commit("tests for flat overlays")

    write("README.md", "# confmerge\n\nLayer config overlays onto a base config.\n\n```python\nfrom confmerge import apply_updates\napply_updates(base, overlay)\n```\n")
    commit("readme: usage example")

    # THE DECISION: union list values (rejecting the replace approach), with rationale.
    write("confmerge/merge.py", MERGE_UNION)
    commit("merge: union list values instead of replacing them\n\n"
           "Replacing a list on overlay silently dropped entries an earlier layer had added — e.g. a "
           "service overlay that set `middleware: [cors]` wiped the `[auth, logging]` the base layer "
           "installed, disabling auth in prod. Overlays must ADD to list values, not replace them, so "
           "union the lists (de-duplicated, base order first). To set a list outright, use a fresh key.",
           author="Priya N.")

    write("CHANGELOG.md", "# Changelog\n\n## Unreleased\n- overlay loader, apply_updates (deep merge + list union)\n")
    commit("start a changelog")
    write("confmerge/loader.py", LOADER + "\n\ndef overlay_keys(text):\n    \"\"\"The keys defined by an overlay string.\"\"\"\n    return list(load_overlay(text).keys())\n")
    commit("loader: add overlay_keys helper", author="Priya N.")
    write("README.md", "# confmerge\n\nLayer config overlays onto a base config.\n\nDeep-merges nested dicts; list values are merged (unioned) across layers.\nSee tests/ for usage.\n")
    commit("readme: note merge semantics")

    # THE REGRESSION (HEAD): "streamline" reverts to a shallow update — nested dicts clobbered,
    # and the list-union decision is lost. Bundled with a plausible cleanup message.
    write("confmerge/merge.py", MERGE_SHALLOW)
    commit("refactor: streamline apply_updates", author="Priya N.")

    write("CHANGELOG.md", "# Changelog\n\n## 0.4.0\n- overlay loader, apply_updates\n- streamlined merge\n")
    commit("changelog for 0.4.0")
    write("pyproject.toml", '[project]\nname = "confmerge"\nversion = "0.4.0"\nrequires-python = ">=3.9"\n')
    commit("release 0.4.0")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    print(f"built {OUT} @ {head} ({n} commits)")


if __name__ == "__main__":
    main()
