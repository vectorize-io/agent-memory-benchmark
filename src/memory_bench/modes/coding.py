"""Coding response mode: build the task repo, run a coding agent with test-feedback interventions,
grade by pytest. Reuses the sdebench harness (`sdebench/harness/run.py`) verbatim.

Memory flows through the STANDARD provider pipeline: the runner ingests the dataset's per-task
corpus into the selected provider (isolation_unit = task), then this mode dispatches:
  - `none`     => the no-memory baseline (`full` arm).
  - `hscoding` => the Hindsight plugin runs INSIDE the agent (reflect+inject); its provider ingested
    via the plugin's own deepen engine. This mode only passes the bank name through.
  - any other provider => generic arm: `provider.retrieve(bug_report)` and the top memories are
    injected into the task prompt (`provided` arm) — how any AMB memory system runs the benchmark.

Env: SDE_AGENT (opencode|claude-code|codex), SDE_HINDSIGHT_URL (Hindsight server, default :8888),
SDE_HSCODING_PLUGIN_DIR (plugin dir with dist/ built; hscoding only), SDE_MODEL.
"""
import asyncio
import json
import os
import shutil
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

    _AGENT_MODEL = {"opencode": "google/gemini-3.5-flash", "claude-code": "claude-sonnet-5",
                    "codex": "gpt-5.4-mini"}

    def __init__(self, model: str | None = None):
        self._agent = os.environ.get("SDE_AGENT", "opencode")   # --agent, so claude runs land in the UI
        self._model = model or os.environ.get("SDE_MODEL") or self._AGENT_MODEL.get(self._agent, "google/gemini-3.5-flash")

    @property
    def llm_id(self) -> str | None:
        return f"{self._agent}:{self._model}"

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

        # Dispatch by provider. Ingestion already happened through the RUNNER's standard pipeline
        # (dataset documents -> provider.async_ingest, per task unit) — nothing is ingested here.
        #   none      -> vanilla arm (full repo, no memory)
        #   hscoding  -> the plugin runs INSIDE the agent (reflect+inject); pass it the bank
        #   any other -> generic arm: provider.retrieve() -> context injected into the task prompt
        env = {**os.environ}
        retrieve_ms = 0.0
        external_memory_file = None
        if memory.name == "vanilla":
            arm = "full"
        elif memory.name == "hindsight-coding":
            arm = "hscoding"
            from ..memory.hscoding import bank_for
            env["SDE_HSCODING_BANK"] = bank_for(task_id)  # run.py -> plugin config (reflect+inject)
            env["SDE_HINDSIGHT_URL"] = os.environ.get("SDE_HINDSIGHT_URL", "http://localhost:8888")
        else:
            arm = "provided"
            t0 = time.perf_counter()
            docs, _raw = await memory.async_retrieve(query, k=10, user_id=task_id)
            retrieve_ms = (time.perf_counter() - t0) * 1000
            block = "\n\n---\n\n".join(d.content for d in docs if d.content) or "(no relevant memories found)"
            ext = Path("/tmp/sdebench/omb-context") / f"{task_id}_{run_id}.txt"
            ext.parent.mkdir(parents=True, exist_ok=True)
            ext.write_text(block)
            external_memory_file = str(ext)

        cmd = ["uv", "run", "python", str(_RUN_PY), "--task", str(task_json),
               "--history", arm, "--agent", self._agent, "--model", self._model, "--run-id", run_id]
        if external_memory_file:
            cmd += ["--external-memory", external_memory_file]
        t0 = time.perf_counter()
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        work = Path("/tmp/sdebench/run") / f"{task_id}_{arm}_{run_id}"
        result_path = work / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            trace_path = work / "trace.json"
            if trace_path.exists():
                # Surface WHAT HAPPENED to the UI's agent view: flatten the per-round trace into one
                # step list (feedback rounds separated by 🔁 markers), plus the final patch and the
                # repo's git history. Injected memory (hscoding) comes from the plugin's diag trail.
                tr = json.loads(trace_path.read_text())
                flat: list = []
                for i, rnd in enumerate(tr.get("trace") or []):
                    if i:
                        flat.append({"k": "say", "text": "🔁 " + (rnd.get("prompt") or "")[:400]})
                    flat.extend(rnd.get("trajectory") or [])
                result["trajectory"] = flat
                result["git_history"] = tr.get("git_history")
                result["final_patch"] = tr.get("final_patch")
            mem_blocks = [d.get("answer") for d in (result.get("memory_diag") or [])
                          if d.get("event") == "reflect_ok" and d.get("answer")]
            if mem_blocks:
                result["memory_context"] = "\n".join(f"## Memory (reflect)\n{a}" for a in mem_blocks)
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
            retrieve_time_ms=retrieve_ms or (float(result.get("wall_s", 0.0)) * 1000 or elapsed_ms),
            raw_response=result,                          # runner's coding branch reads solved/interventions/…
        )
