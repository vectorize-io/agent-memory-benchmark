"""Coding response mode: build the task repo, run a coding agent with test-feedback interventions,
grade by pytest. Reuses the proven sdebench harness (`sdebench/harness/run.py`) verbatim.

AMB does ZERO memory work for the coding task — memory is entirely the plugin's domain:
  - `none`     => the no-memory baseline (`full` arm).
  - `hscoding` => the mode (a) builds the task repo, (b) triggers the PLUGIN's own backfill
    (`hindsight-coding-backfill`) over that repo + the task's conversations — the plugin decides
    what and how to ingest — then (c) runs opencode + the plugin (`hscoding` arm), which does
    reflect+inject. AMB never calls Hindsight retain or reflect itself.
Any other provider raises. The harness result is returned as an AnswerResult (the runner's `coding`
branch reads `solved` etc.).

Env: SDE_HINDSIGHT_URL (Hindsight server, default :8888), SDE_HSCODING_PLUGIN_DIR (the plugin package
dir holding dist/backfill.js, default ~/dev/hindsight-coding-opencode), SDE_HSCODING_GIT_LIMIT
(optional git scope passed to the plugin backfill; unset => the plugin decides), SDE_MODEL.
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

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("SDE_MODEL", "google/gemini-3.5-flash")

    @property
    def llm_id(self) -> str | None:
        return self._model

    def answer(self, query: str, memory: MemoryProvider, task_type: str = "coding", user_id: str | None = None) -> AnswerResult:
        return asyncio.run(self.async_answer(query, memory, task_type=task_type, user_id=user_id))

    def answer_from_context(self, query: str, context: str, task_type: str = "coding") -> AnswerResult:
        raise NotImplementedError("coding mode grades by running the agent; --skip-retrieval is not supported")

    async def _plugin_backfill(self, task_json: str, task_id: str, run_id: str, bank: str, url: str) -> None:
        """Trigger the PLUGIN's own backfill — AMB does not ingest. Build the task repo, then run the
        plugin's `hindsight-coding-backfill` over that repo + the task's conversations; the plugin
        decides what and how to ingest (extraction, strategies, git scope, pages)."""
        plugin_dir = Path(os.environ.get("SDE_HSCODING_PLUGIN_DIR",
                                         str(Path.home() / "dev" / "hindsight-coding-opencode")))
        backfill_js = plugin_dir / "dist" / "backfill.js"
        tj = Path(task_json)
        t = json.loads(tj.read_text())
        build_py = tj.parents[2] / t.get("build", "build.py")
        base = Path("/tmp/sdebench/omb-backfill") / f"{task_id}_{run_id}"
        src = base / "repo"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        # 1. build the task repo (the plugin backfill reads its git history)
        await asyncio.to_thread(subprocess.run, ["python", str(build_py), str(src)],
                                capture_output=True, text=True, env={**os.environ})
        # 2. run the plugin's backfill (it owns extraction/strategies/pages/git scope)
        bf = ["node", str(backfill_js), "--repo", str(src), "--bank", bank, "--api-url", url, "--reset"]
        conv = t.get("conversations") or []
        if conv:
            cf = base / "conversations.json"
            cf.write_text(json.dumps([{"id": task_id, "turns": conv}]))
            bf += ["--conversations", str(cf)]
        limit = os.environ.get("SDE_HSCODING_GIT_LIMIT")  # optional scope; unset => the plugin decides
        if limit:
            bf += ["--limit", limit]
        await asyncio.to_thread(subprocess.run, bf, capture_output=True, text=True, env={**os.environ})

    async def async_answer(self, query: str, memory: MemoryProvider, task_type: str = "coding",
                           user_id: str | None = None, meta: dict | None = None) -> AnswerResult:
        meta = meta or {}
        task_json = meta.get("task_json")
        task_id = user_id or meta.get("task_id") or "task"
        run_id = f"omb-{uuid.uuid4().hex[:8]}"

        if not task_json:
            return AnswerResult(answer="unsolved", reasoning="no task_json in meta", context="",
                                retrieve_time_ms=0.0, raw_response={"solved": False})

        # `none` => vanilla; `hscoding` => trigger the plugin's own backfill, then run opencode+plugin.
        env = {**os.environ}
        if memory.name == "none":
            arm = "full"
        elif memory.name == "hscoding":
            arm = "hscoding"
            bank = f"sde-coding-{task_id}"
            url = os.environ.get("SDE_HINDSIGHT_URL", "http://localhost:8888")
            await self._plugin_backfill(task_json, task_id, run_id, bank, url)  # PLUGIN ingests; AMB does not
            env["SDE_HSCODING_BANK"] = bank   # run.py -> HINDSIGHT_BANK_ID for the plugin (reflect)
            env["SDE_HINDSIGHT_URL"] = url     # run.py -> HINDSIGHT_API_URL for the plugin
        else:
            raise NotImplementedError(f"coding mode supports 'none' and 'hscoding'; got '{memory.name}'")

        cmd = ["uv", "run", "python", str(_RUN_PY), "--task", str(task_json),
               "--history", arm, "--model", self._model, "--run-id", run_id]
        t0 = time.perf_counter()
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env,
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
