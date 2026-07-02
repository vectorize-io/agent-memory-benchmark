#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook — the claude-code equivalent of the opencode memory plugin's
auto-inject (no MCP/tool: we only need to INJECT). Reflects the prompt over the project's Hindsight
bank and returns the synthesized root-cause answer as additionalContext. Inert (no-op) unless a bank
is configured, so the SAME image serves the vanilla baseline and the memory arm.

Config via env (the harness sets these on the container for the memory arm only):
  HINDSIGHT_API_URL, HINDSIGHT_BANK_ID, HINDSIGHT_DISABLED (off-switch), HINDSIGHT_REFLECT_TIMEOUT_MS.
"""
import json, os, sys, urllib.request


def main():
    url = os.environ.get("HINDSIGHT_API_URL")
    bank = os.environ.get("HINDSIGHT_BANK_ID")
    if os.environ.get("HINDSIGHT_DISABLED") or not url or not bank:
        return  # vanilla baseline: no memory
    try:
        ev = json.load(sys.stdin)
    except Exception:
        ev = {}
    prompt = (ev.get("prompt") or "").strip()
    if not prompt:
        return
    timeout = int(os.environ.get("HINDSIGHT_REFLECT_TIMEOUT_MS", "120000")) / 1000
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/default/banks/{bank}/reflect",
            data=json.dumps({"query": prompt, "budget": "high"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ans = (json.loads(r.read()).get("text") or "").strip()
    except Exception:
        ans = ""  # best-effort: memory never breaks the agent
    if ans:
        ctx = ("Relevant project memory, surfaced from THIS repository's git history and past developer "
               "conversations — a past decision that likely explains this issue. If it states an EXACT "
               "rule or literal values, apply them PRECISELY as given; verify against the current code:\n\n"
               + ans)
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit", "additionalContext": ctx}}))


if __name__ == "__main__":
    main()
