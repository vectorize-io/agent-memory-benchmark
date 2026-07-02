"""Coding response mode: build the task repo, run a coding agent with test-feedback interventions,
grade by pytest. Reuses the proven sdebench harness (`sdebench/harness/run.py`) verbatim.

Memory flows through the OMB memory provider: the runner ingests the dataset's git+chat documents
into the provider, and this mode queries it per task (Hindsight `reflect` via direct_answer, else
`retrieve`) and injects the surfaced decision into the agent via run.py's `provided` arm. Only the
no-memory baseline (`none`) and Hindsight providers are supported; other providers raise for coding.
The harness result is returned as an AnswerResult (the runner's `coding` branch reads `solved` etc.).
"""
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from .base import ResponseMode
from ..memory.base import MemoryProvider
from ..models import AnswerResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_PY = _REPO_ROOT / "sdebench" / "harness" / "run.py"


class CodingMode(ResponseMode):
    name = "coding"
    description = "Build the task repo, run a coding agent with test-feedback interventions, grade by pytest."

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("SDE_MODEL", "google/gemini-3.5-flash")

    @property
    def llm_id(self) -> str | None:
        return self._model

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "coding", user_id: str | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id))

    def answer_from_context(self, query: str, context: str, task_type: str = "coding") -> AnswerResult:
        raise NotImplementedError("coding mode grades by running the agent; --skip-retrieval is not supported")

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "coding",
                           user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        task_json = meta.get("task_json")
        task_id = user_id or meta.get("task_id") or "task"
        run_id = f"omb-{uuid.uuid4().hex[:8]}"

        if not task_json:
            return AnswerResult(answer="unsolved", reasoning="no task_json in meta", context="",
                                retrieve_time_ms=0.0, raw_response={"solved": False})

        # Map the OMB memory provider to a harness arm. The provider (populated by the runner's ingest)
        # is queried here for the decision, which is injected via run.py's `provided` arm. Only the
        # no-memory baseline and Hindsight are supported; other providers raise for the coding task.
        external_memory_file = None
        surfaced = ""
        if memory.name == "none":
            arm = "full"
        elif memory.name.startswith("hindsight"):
            arm = "provided"
            try:
                answer, ctx, _ = await memory.async_direct_answer(query)  # Hindsight reflect
                surfaced = (answer or ctx or "").strip()
            except NotImplementedError:
                docs, _ = await memory.async_retrieve(query, k=5)
                surfaced = "\n\n".join(d.content for d in docs).strip()
            if surfaced:
                external_memory_file = Path("/tmp/sdebench/omb-mem") / f"{task_id}_{run_id}.txt"
                external_memory_file.parent.mkdir(parents=True, exist_ok=True)
                external_memory_file.write_text(surfaced)
        else:
            raise NotImplementedError(
                f"coding mode supports memory providers 'none' and 'hindsight*'; got '{memory.name}'")

        cmd = ["uv", "run", "python", str(_RUN_PY), "--task", str(task_json),
               "--history", arm, "--model", self._model, "--run-id", run_id]
        if external_memory_file:
            cmd += ["--external-memory", str(external_memory_file)]
        t0 = time.perf_counter()
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), env={**os.environ},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        work = Path("/tmp/sdebench/run") / f"{task_id}_{arm}_{run_id}"
        result_path = work / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            # harness crashed before grading — surface stderr tail so it's debuggable
            result = {"solved": False, "interventions": None,
                      "final_pytest": (proc.stderr or proc.stdout or "")[-400:], "error": "no result.json"}

        solved = bool(result.get("solved"))
        return AnswerResult(
            answer="solved" if solved else "unsolved",
            reasoning=f"arm={arm} interventions={result.get('interventions')} "
                      f"cost=${result.get('cost_usd')} turns={result.get('turns')}",
            context=f"memory={memory.name} arm={arm}",   # non-empty; coding scoring ignores context
            retrieve_time_ms=float(result.get("wall_s", 0.0)) * 1000 or elapsed_ms,
            raw_response=result,                          # runner's coding branch reads solved/interventions/…
        )
