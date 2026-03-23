import json
import os
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        delay = _RETRY_BASE_DELAY
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},
                    },
                )
                text = response.choices[0].message.content
                return json.loads(text)
            except Exception as e:
                last_exc = e
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        raise RuntimeError(f"OpenAI request failed after {_MAX_RETRIES} retries: {last_exc}")
