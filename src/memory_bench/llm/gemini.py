import logging
import time

from google import genai
from google.genai import types

from .base import LLM, Schema, ToolDef

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "string":  types.Type.STRING,
    "boolean": types.Type.BOOLEAN,
    "integer": types.Type.INTEGER,
    "number":  types.Type.NUMBER,
}

# Retry config for 429 RESOURCE_EXHAUSTED
_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5   # seconds — doubles on each attempt (5, 10, 20, 40, 80, 160)


class GeminiLLM(LLM):
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self._client = genai.Client()
        self._model = model

    @property
    def model_id(self) -> str:
        return f"gemini:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        import json as _json
        genai_schema = self._build_schema(schema)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=genai_schema,
        )
        delay = _RETRY_BASE_DELAY
        last_text = ""
        for attempt in range(_MAX_RETRIES):
            response = self._generate_raw(prompt, config=config)
            if response.parsed is not None:
                return response.parsed
            # Try to extract text and parse manually
            try:
                raw_text = ""
                if hasattr(response, 'candidates') and response.candidates:
                    c = response.candidates[0]
                    if c.content and c.content.parts:
                        for part in c.content.parts:
                            if hasattr(part, 'text') and part.text:
                                raw_text += part.text
                elif hasattr(response, 'text') and response.text:
                    raw_text = response.text
                last_text = raw_text
                if raw_text:
                    # Try direct JSON parse
                    parsed = _json.loads(raw_text)
                    if all(k in parsed for k in schema.required):
                        return parsed
                    # Try extracting JSON block
                    import re as _re
                    m = _re.search(r'\{.*\}', raw_text, _re.DOTALL)
                    if m:
                        parsed = _json.loads(m.group())
                        if all(k in parsed for k in schema.required):
                            return parsed
            except Exception:
                pass
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
        # Last resort: build dict from text using first required field as answer
        if last_text and schema.required:
            return {k: last_text for k in schema.required}
        raise RuntimeError(f"Gemini returned unparseable response after {_MAX_RETRIES} retries")

    def tool_loop(self, prompt: str, tools: list[ToolDef], max_tool_calls: int = 10) -> str:
        fn_map = {t.name: t.fn for t in tools}
        gemini_tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        k: types.Schema(
                            type=_TYPE_MAP.get(v.get("type", "string"), types.Type.STRING),
                            description=v.get("description", ""),
                        )
                        for k, v in t.parameters.items()
                    },
                    required=t.required,
                ),
            )
            for t in tools
        ])]

        contents: list = [prompt]
        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            response = self._generate_raw(contents, config=types.GenerateContentConfig(tools=gemini_tools))
            parts = response.candidates[0].content.parts

            # Collect any function calls in this response
            fn_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if not fn_calls:
                # No function calls — return the text from the first text part
                for p in parts:
                    if getattr(p, "text", None):
                        return p.text
                return ""

            # Append model turn to history
            contents = list(contents) + [response.candidates[0].content]

            # Execute all function calls and append results
            result_parts = []
            for fc in fn_calls:
                result = fn_map[fc.name](**dict(fc.args))
                result_parts.append(types.Part(
                    function_response=types.FunctionResponse(name=fc.name, response={"result": result})
                ))
                tool_call_count += 1

            contents.append(types.Content(role="user", parts=result_parts))

        # Max calls reached — ask for final answer without tools
        response = self._generate_raw(contents)
        for p in response.candidates[0].content.parts:
            if getattr(p, "text", None):
                return p.text
        return ""

    def _generate_raw(self, contents, config: types.GenerateContentConfig | None = None):
        config = config or types.GenerateContentConfig()
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or
                        "503" in msg or "UNAVAILABLE" in msg):
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning("[gemini] retry %d/%d after %.0fs — %s", attempt + 1, _MAX_RETRIES, delay, msg[:120])
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        raise RuntimeError(f"Gemini request failed after {_MAX_RETRIES} retries")

    @staticmethod
    def _build_schema(schema: Schema) -> types.Schema:
        properties = {}
        for name, spec in schema.properties.items():
            prop = types.Schema(
                type=_TYPE_MAP.get(spec.get("type", "string"), types.Type.STRING),
            )
            if "description" in spec:
                prop.description = spec["description"]
            properties[name] = prop

        return types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=schema.required,
        )
