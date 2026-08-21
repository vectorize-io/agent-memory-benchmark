"""
AGY direct LLM — calls Antigravity CLI via subprocess, no HTTP gateway.
Eliminates the gateway as a failure point.
"""

import json
import os
import subprocess

from .base import LLM, Schema

_MAX_RETRIES = 3


class AgyDirectLLM(LLM):
    """Calls agy CLI directly. No HTTP gateway, no intermediary process."""

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("AGY_MODEL", "gemini-3.6-flash")
        self._effort = os.environ.get("AGY_EFFORT", "low")

    @property
    def model_id(self) -> str:
        return f"agy:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        fields_desc = ", ".join(schema.required)
        full_prompt = prompt + f"\n\nRespond as a JSON object with these fields: {fields_desc}. Output ONLY valid JSON."

        for attempt in range(_MAX_RETRIES):
            try:
                result = subprocess.run(
                    ["agy", "--model", self._model, "--effort", self._effort,
                     "-p", full_prompt, "--output-format", "json"],
                    capture_output=True, text=True, timeout=120,
                    env={**os.environ},
                )
                if result.returncode != 0:
                    raise RuntimeError(f"agy exit {result.returncode}: {result.stderr[:200]}")

                data = json.loads(result.stdout)
                if data.get("status") == "SUCCESS":
                    response = data.get("response", "")
                    return self._parse_json(response)

                raise RuntimeError(f"agy error: {data.get('error', 'unknown')}")

            except subprocess.TimeoutExpired:
                if attempt < _MAX_RETRIES - 1:
                    import time; time.sleep(2)
                    continue
                raise
            except Exception:
                if attempt < _MAX_RETRIES - 1:
                    import time; time.sleep(2)
                    continue
                raise

    def _parse_json(self, text: str) -> dict:
        import re
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {f: "" for f in ["reasoning", "answer", "choice", "reason", "correct"]}
