"""
GLM-5.2 LLM for AMB — via Z.ai coding plan endpoint.

GLM-5.2 is a reasoning model: outputs reasoning_content separately from content.
Uses prompt-based JSON extraction (Z.ai doesn't support response_format).

Tuned for benchmark use (2026-08-09):
- max_tokens=16384: GLM-5.2 reasoning can consume 5k-10k tokens before producing
  the actual content. 8192 was too low — judge prompts with long answers saw
  reasoning eat the entire budget, leaving content empty → score=0 fallback.
- Request is rebuilt every attempt (urllib Request objects are single-use).
- Reasoning-only responses are parsed for trailing JSON (GLM sometimes emits
  the score inside reasoning_content when content is truncated).
- Structured logging on failure for post-mortem.
"""

import json
import logging
import os
import time as _time
import urllib.request

from .gateway import GatewayLLM

logger = logging.getLogger(__name__)

# GLM-5.2 reasoning consumes many tokens before the answer lands in `content`.
# 16k gives headroom for long judge prompts (answer + context + rubric).
# Coding plan has quota; benchmark correctness > token economy here.
_MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "16384"))

# Per-request timeout — reasoning on long prompts can take 2-3 min.
_TIMEOUT = int(os.environ.get("GLM_TIMEOUT", "240"))


class GlmLLM(GatewayLLM):
    """GLM-5.2 via Z.ai coding plan. Expensive per-request, use sparingly."""

    def __init__(self, model: str | None = None):
        self._base = os.environ.get(
            "GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4"
        ).rstrip("/")
        self._key = os.environ.get("GLM_API_KEY", "")
        self._model = model or os.environ.get("GLM_MODEL", "glm-5.2")

    @property
    def model_id(self) -> str:
        return f"glm:{self._model}"

    def _call(self, prompt: str) -> str:
        url = f"{self._base}/chat/completions"
        body_dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
        }

        last_err = None
        for attempt in range(3):
            # Rebuild request every attempt — urllib Request is single-use
            # after urlopen consumes the body stream.
            body = json.dumps(body_dict).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self._key}"} if self._key else {}),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg = data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "").strip()
                    reasoning = msg.get("reasoning_content", "").strip()
                    finish = data.get("choices", [{}])[0].get("finish_reason", "")

                    if content:
                        return content

                    # Content empty but reasoning has the answer — GLM sometimes
                    # emits the JSON score at the end of reasoning_content when
                    # content gets truncated. Try to salvage it.
                    if reasoning:
                        salvaged = _extract_json_from_text(reasoning)
                        if salvaged:
                            logger.info(
                                "[glm] salvaged JSON from reasoning_content "
                                "(attempt %d, finish=%s, reasoning=%d chars)",
                                attempt + 1, finish, len(reasoning),
                            )
                            return salvaged

                    last_err = f"empty content (finish={finish}, reasoning={len(reasoning)} chars)"
                    logger.warning(
                        "[glm] empty response attempt %d/%d: %s",
                        attempt + 1, 3, last_err,
                    )
            except Exception as e:
                last_err = str(e)
                logger.warning("[glm] error attempt %d/%d: %s", attempt + 1, 3, last_err)

            if attempt < 2:
                _time.sleep(5 * (attempt + 1))  # 5s, 10s backoff

        logger.error("[glm] all 3 attempts failed: %s", last_err)
        return ""


def _extract_json_from_text(text: str) -> str:
    """Try to find a JSON object ({...}) in text — GLM reasoning fallback.

    Looks for the last JSON object in the text (reasoning often ends with
    the final answer). Returns the raw JSON string, or "" if none found.
    """
    # Look for ```json ... ``` blocks first (GLM markdown-wraps sometimes)
    import re
    md = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", text, re.IGNORECASE)
    if md:
        return md.group(1).strip()

    # Fallback: last {...} block in the text
    matches = re.findall(r"\{[^{}]*\"score\"[^{}]*\}", text)
    if matches:
        return matches[-1].strip()

    return ""
