from .llm.base import LLM, Schema
from .llm.gemini import GeminiLLM
from .models import JudgeResult

_PROMPT = """\
You are a strict evaluator judging whether an AI system correctly answered a question.

Question:
{query}

Gold answers (at least one must be substantially matched):
{gold_answers}

System's answer:
{answer}

Evaluation rules — mark correct=false if ANY of these apply:
- The system says it cannot answer, doesn't know, or lacks enough information
- The system gives a vague or evasive answer instead of a concrete one
- The system's answer contradicts or omits key facts present in the gold answers
- The system hedges heavily without providing the actual answer

Mark correct=true only if the system's answer captures the essential facts from the gold answer. \
Minor wording differences and reasonable paraphrasing are fine.\
"""

_SCHEMA = Schema(
    properties={
        "correct": {"type": "boolean"},
        "reason":  {"type": "string"},
    },
    required=["correct", "reason"],
)


class GeminiJudge:
    def __init__(self, llm: LLM | None = None):
        from .llm import get_judge_llm
        self._llm = llm or get_judge_llm()

    def score(self, query: str, answer: str, gold_answers: list[str], prompt_fn=None) -> JudgeResult:
        prompt = prompt_fn(query, gold_answers, answer) if prompt_fn else None
        if not prompt:
            gold_str = "\n".join(f"- {a}" for a in gold_answers)
            prompt = _PROMPT.format(query=query, gold_answers=gold_str, answer=answer)
        data = self._llm.generate(prompt, _SCHEMA)
        return JudgeResult(correct=bool(data["correct"]), reason=data["reason"])
