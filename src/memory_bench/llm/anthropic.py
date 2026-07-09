import json
import os
import re
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5


class _StructuredOutputError(ValueError):
    """Raised when the model response does not match the requested schema."""


class _ToolUseUnsupportedError(RuntimeError):
    """Raised when an Anthropic-compatible endpoint does not support forced tools."""


def _parse_json_payload(text: str) -> dict:
    text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(payload, dict):
            raise _StructuredOutputError("Model response must be a JSON object")
        return payload

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = json.loads(fenced.group(1))
        if not isinstance(payload, dict):
            raise _StructuredOutputError("Model response must be a JSON object")
        return payload

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise _StructuredOutputError("Model response must be a JSON object")
        return payload

    raise json.JSONDecodeError("Could not find JSON object in model response", text, 0)


def _coerce_text_payload(text: str, schema: Schema) -> dict | None:
    text = text.strip()
    if not text:
        return None
    if len(schema.required) != 1:
        return None

    field = schema.required[0]
    spec = schema.properties.get(field, {})
    field_type = spec.get("type", "string")

    if field_type == "string":
        return {field: text}

    if field_type == "boolean":
        lowered = text.lower()
        if lowered == "true":
            return {field: True}
        if lowered == "false":
            return {field: False}

    return None


def _validate_schema_payload(payload: dict, schema: Schema) -> dict:
    extra = sorted(set(payload) - set(schema.properties))
    if extra:
        raise _StructuredOutputError(f"Model response included unsupported field(s): {', '.join(extra)}")

    missing = [field for field in schema.required if field not in payload]
    if missing:
        raise _StructuredOutputError(f"Model response omitted required field(s): {', '.join(missing)}")

    for field, value in payload.items():
        spec = schema.properties.get(field, {})
        expected_type = spec.get("type", "string")
        if expected_type == "string":
            valid = isinstance(value, str)
        elif expected_type == "boolean":
            valid = isinstance(value, bool)
        elif expected_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "array":
            valid = isinstance(value, list)
        elif expected_type == "object":
            valid = isinstance(value, dict)
        else:
            valid = True

        if not valid:
            raise _StructuredOutputError(
                f"Model response field '{field}' must be {expected_type}, got {type(value).__name__}"
            )

    return payload


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None):
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Anthropic provider requires ANTHROPIC_API_KEY")

        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        self._client = Anthropic(
            api_key=api_key,
            base_url=base_url or None,
            max_retries=0,
        )
        self._model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-sonnet-4-6"
        )

    @property
    def model_id(self) -> str:
        return f"anthropic:{self._model}"

    @staticmethod
    def _schema_json(schema: Schema) -> dict:
        return {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }

    def _generate_with_tool(self, prompt: str, schema: Schema) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "structured_response",
                    "description": "Return the structured response matching the requested schema.",
                    "input_schema": self._schema_json(schema),
                }
            ],
            tool_choice={"type": "tool", "name": "structured_response"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "structured_response":
                payload = getattr(block, "input", None)
                if not isinstance(payload, dict):
                    raise _StructuredOutputError("Tool input must be a JSON object")
                return _validate_schema_payload(payload, schema)

        raise _ToolUseUnsupportedError("Anthropic response did not include the forced structured_response tool call")

    def _generate_with_text_json(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        system_prompt = (
            "Return only a valid JSON object matching this schema. "
            "Do not wrap JSON in markdown fences.\n\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        try:
            payload = _parse_json_payload(text)
        except json.JSONDecodeError:
            coerced = _coerce_text_payload(text, schema)
            if coerced is None:
                raise _StructuredOutputError("Model response was not valid JSON") from None
            payload = coerced
        return _validate_schema_payload(payload, schema)

    @staticmethod
    def _looks_like_tool_unsupported(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "tool_choice",
                "tools",
                "tool use",
                "tool_use",
                "unknown field",
                "extra fields",
                "not supported",
                "unsupported",
                "invalid request",
            )
        )

    @staticmethod
    def _is_retryable_status_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in (429, 500, 502, 503, 504):
            return True

        if status_code != 400:
            return False

        # Some Anthropic-compatible gateways return HTTP 400 while embedding an
        # upstream/internal 500 in the JSON body.
        msg = str(exc)
        return (
            "操作失败" in msg
            or "'code': '500'" in msg
            or '"code": "500"' in msg
            or '"code":"500"' in msg
        )

    def generate(self, prompt: str, schema: Schema) -> dict:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        delay = _RETRY_BASE_DELAY
        last_exc = None
        use_tool = True

        for attempt in range(_MAX_RETRIES):
            try:
                if use_tool:
                    try:
                        return self._generate_with_tool(prompt, schema)
                    except (APIStatusError, _ToolUseUnsupportedError, Exception) as e:
                        if isinstance(e, APIStatusError) and e.status_code not in (400, 404, 422):
                            raise
                        if (
                            not isinstance(e, (APIStatusError, _ToolUseUnsupportedError))
                            and not self._looks_like_tool_unsupported(e)
                        ):
                            raise
                        use_tool = False
                return self._generate_with_text_json(prompt, schema)
            except (RateLimitError, APIConnectionError) as e:
                last_exc = e
            except APIStatusError as e:
                last_exc = e
                if not self._is_retryable_status_error(e):
                    raise
            except _StructuredOutputError as e:
                last_exc = e
            except Exception as e:
                last_exc = e
                msg = str(e)
                if "429" not in msg and "rate" not in msg.lower():
                    raise

            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Anthropic request failed after {_MAX_RETRIES} retries: {last_exc}")
