import asyncio
import re
import time

from .base import ResponseMode
from .rag import RAGMode
from ..llm.base import LLM, Schema
from ..llm.gemini import GeminiLLM
from ..memory.base import MemoryProvider
from ..models import AnswerResult

_MAX_TOOL_CALLS = 8
_MAX_HEADER_RANGE = 120
_SEARCH_MAX_TOKENS = 4000   # Token budget for search_steps

_TOOL_SCHEMA = Schema(
    properties={
        "tool": {
            "type": "string",
            "enum": ["get_step_headers", "get_steps", "search_steps", "answer"],
            "description": (
                "get_step_headers: compact summaries for a range of steps (action + outcome label). Use to scan and orient. "
                "get_steps: full verbatim content for up to 8 specific step numbers. Use for exact values/IDs/code. "
                "search_steps: semantic + keyword search over all steps, returns full verbatim matches. Use when you don't know which step number but know what to look for. "
                "answer: final answer — ONLY when you have clear evidence from fetched steps."
            ),
        },
        "step_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Step numbers for get_steps (max 8).",
        },
        "range_from": {
            "type": "integer",
            "description": "Start of range for get_step_headers (inclusive).",
        },
        "range_to": {
            "type": "integer",
            "description": "End of range for get_step_headers (inclusive).",
        },
        "query": {
            "type": "string",
            "description": "Search query for search_steps.",
        },
        "answer": {
            "type": "string",
            "description": "Your final answer.",
        },
    },
    required=["tool"],
)

# First round: answer blocked — force at least one retrieval
_TOOL_SCHEMA_NO_ANSWER = Schema(
    properties={
        "tool": {
            "type": "string",
            "enum": ["get_step_headers", "get_steps", "search_steps"],
            "description": (
                "get_step_headers: compact summaries for a range of steps (action + outcome label). Use to scan and orient. "
                "get_steps: full verbatim content for up to 8 specific step numbers. Use for exact values/IDs/code. "
                "search_steps: semantic + keyword search over all steps, returns full verbatim matches. Use when you don't know which step number but know what to look for."
            ),
        },
        "step_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Step numbers for get_steps (max 8).",
        },
        "range_from": {
            "type": "integer",
            "description": "Start of range for get_step_headers (inclusive).",
        },
        "range_to": {
            "type": "integer",
            "description": "End of range for get_step_headers (inclusive).",
        },
        "query": {
            "type": "string",
            "description": "Search query for search_steps.",
        },
    },
    required=["tool"],
)

_TOOL_PROMPT = """You are an expert analyst of agent trajectory logs stored in a memory bank.

Steps retrieved so far:
{context}

Question: {question}

Tools remaining: {tools_left}. Choose ONE action:

- get_step_headers(range_from, range_to): Compact step summaries for a range. Fast — use to scan the trajectory structure.
- get_steps(step_numbers=[N,...]): Full verbatim content for specific steps (max 8). Use when you know which step and need exact output/code/IDs.
- search_steps(query): Semantic + keyword search — returns full verbatim steps matching the query. Use when you need to find which step contains a specific string or event.
- answer: Final answer. ONLY when fetched context contains clear evidence. Never answer from memory or general knowledge.

Rules:
1. ORIENT FIRST: If you don't know exactly which steps are relevant, call get_step_headers on the relevant range BEFORE get_steps. Never guess step numbers blindly.
2. LOOP DETECTION: Loop START = first step of SECOND occurrence of repeating cycle.
3. "Visited page X" = FIRST arrival at X after being elsewhere.
4. Element IDs like [29575] must come from verbatim step content.
5. Turn N = Step N exactly.
6. For range/count/pattern questions, get_step_headers the full relevant range first, then get_steps for the specific steps you need.
7. ACTION COUNTING: When counting or listing actions, always use the `action_detail` field (e.g. 'go to drawer 1', 'open cabinet 3'), never the generic `action` field (e.g. 'go to', 'open'). Each distinct action_detail value is a separate action."""


def _parse_step_numbers(raw) -> list[int]:
    """Coerce LLM output to a list of ints — handles int, str '5', list [5], list ['5']."""
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        return [int(x) for x in re.findall(r'\d+', raw)]
    if isinstance(raw, list):
        result = []
        for s in raw:
            if isinstance(s, (int, float)):
                result.append(int(s))
            elif isinstance(s, str):
                result.extend(int(x) for x in re.findall(r'\d+', s))
        return result
    return []


def _sort_key(content: str) -> int:
    m = re.search(r'step_number=(\d+)', content)
    if m:
        return int(m.group(1))
    m2 = re.search(r'Step (\d+)', content)
    return int(m2.group(1)) if m2 else 9999


class AMAAgentMode(ResponseMode):
    """AMA-Agent: pure tool loop — get_step_headers + get_steps + search_steps + answer."""

    name = "ama-agent"
    description = "Specialised for trajectory datasets (ama-bench). Uses get_step_headers / get_steps / search_steps tools; requires at least one retrieval before answering."

    def __init__(self, llm: LLM | None = None, k: int = 40):
        self._llm = llm or GeminiLLM()
        self._rag = RAGMode(llm=self._llm, k=k)
        self.k = k

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id, meta=meta))

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "open", user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        t0 = time.perf_counter()
        retrieval_query = meta.get("retrieval_query") or query
        query_timestamp = meta.get("query_timestamp")
        has_by_steps = hasattr(memory, "async_retrieve_by_steps")

        by_id: dict[str, object] = {}
        tools_called: list[dict] = []
        called_signatures: set[str] = set()

        def _rebuild_context() -> str:
            if not by_id:
                return "(no steps fetched yet)"
            return "\n\n".join(
                doc.content for doc in sorted(by_id.values(), key=lambda d: _sort_key(d.content))
            )

        context = _rebuild_context()

        if has_by_steps and task_type == "open":
            for i in range(_MAX_TOOL_CALLS):
                schema = _TOOL_SCHEMA_NO_ANSWER if i == 0 else _TOOL_SCHEMA
                action = await asyncio.to_thread(
                    self._llm.generate,
                    _TOOL_PROMPT.format(
                        context=context[:60000],
                        question=query,
                        tools_left=_MAX_TOOL_CALLS - len(tools_called),
                    ),
                    schema,
                )
                if not isinstance(action, dict):
                    break

                tool = action.get("tool", "answer")

                if tool == "answer":
                    if i == 0:
                        break  # shouldn't happen (schema blocks it), but guard anyway
                    final_answer = action.get("answer", "")
                    retrieve_ms = (time.perf_counter() - t0) * 1000
                    return AnswerResult(
                        answer=final_answer,
                        reasoning="",
                        context=context,
                        retrieve_time_ms=round(retrieve_ms, 1),
                        raw_response={"docs_retrieved": len(by_id), "tools_called": tools_called},
                    )

                elif tool == "get_steps":
                    steps = _parse_step_numbers(action.get("step_numbers"))[:8]
                    if not steps:
                        break
                    sig = f"steps:{sorted(steps)}"
                    if sig in called_signatures:
                        break
                    called_signatures.add(sig)
                    tools_called.append({"tool": "get_steps", "steps": steps})
                    new_docs, _ = await memory.async_retrieve_by_steps(
                        steps, retrieval_query, k=self.k,
                        user_id=user_id, query_timestamp=query_timestamp,
                        compact=False,
                    )
                    for doc in new_docs:
                        by_id[doc.id] = doc

                elif tool == "get_step_headers":
                    lo = action.get("range_from")
                    hi = action.get("range_to")
                    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                        break
                    lo, hi = int(lo), max(int(lo), int(hi))
                    hi = min(hi, lo + _MAX_HEADER_RANGE)
                    sig = f"headers:{lo}-{hi}"
                    if sig in called_signatures:
                        break
                    called_signatures.add(sig)
                    tools_called.append({"tool": "get_step_headers", "from": lo, "to": hi})
                    new_docs, _ = await memory.async_retrieve_by_steps(
                        list(range(lo, hi + 1)), retrieval_query, k=self.k,
                        user_id=user_id, query_timestamp=query_timestamp,
                        compact=True,
                    )
                    for doc in new_docs:
                        if doc.id not in by_id:  # don't overwrite full verbatim
                            by_id[doc.id] = doc

                elif tool == "search_steps":
                    search_query = action.get("query", "")
                    if not search_query:
                        break
                    sig = f"search:{search_query[:80]}"
                    if sig in called_signatures:
                        break
                    called_signatures.add(sig)
                    tools_called.append({"tool": "search_steps", "query": search_query})
                    new_docs, _ = await memory.async_search_steps(search_query, user_id=user_id, max_tokens=_SEARCH_MAX_TOKENS)
                    for doc in new_docs:
                        by_id[doc.id] = doc

                context = _rebuild_context()

        retrieve_ms = (time.perf_counter() - t0) * 1000

        if task_type == "open":
            result = await asyncio.to_thread(self._answer_open, query, context)
        else:
            result = await asyncio.to_thread(self._rag.answer_from_context, query, context, task_type)

        return AnswerResult(
            answer=result.answer,
            reasoning=result.reasoning,
            context=context,
            retrieve_time_ms=round(retrieve_ms, 1),
            raw_response={"docs_retrieved": len(by_id), "tools_called": tools_called},
        )

    def _answer_open(self, query: str, context: str) -> AnswerResult:
        from ..llm.base import Schema as _Schema
        schema = _Schema(
            properties={"answer": {"type": "string", "description": "Detailed, complete answer citing exact step numbers, element IDs, and verbatim values from the context."}},
            required=["answer"],
        )
        prompt = f"""You are an expert analyst of agent trajectory logs.

{context}

Question: {query}

Rules:
1. "Visited page X" = FIRST arrival after being on a different page.
2. LOOP DETECTION: Loop START = first step of the SECOND occurrence of the repeating cycle.
3. Element IDs like [29575] must come exactly from the FULL step content retrieved above.
4. Turn N = Step N exactly.
5. Give a COMPLETE and DETAILED answer. Include all relevant step numbers, element IDs, and verbatim values the question asks about. Do not omit sub-parts of a multi-part question.

ANSWER:"""
        data = self._llm.generate(prompt, schema)
        answer = data.get("answer", "") if isinstance(data, dict) else str(data)
        return AnswerResult(answer=answer, reasoning="", context=context, retrieve_time_ms=0.0)

    def answer_from_context(self, query: str, context: str, task_type: str = "open") -> AnswerResult:
        if task_type == "open":
            return self._answer_open(query, context)
        return self._rag.answer_from_context(query, context, task_type)
