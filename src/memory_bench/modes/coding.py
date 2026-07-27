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
dir holding dist/backfill.js, default = the hindsight monorepo's hindsight-coding-agents package), SDE_HSCODING_GIT_LIMIT
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

    _AGENT_MODEL = {"opencode": "google/gemini-3.5-flash", "claude-code": "claude-sonnet-5",
                    "codex": "gpt-5.1-codex-mini"}

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

    @staticmethod
    def _bank_has_memories(url: str, bank: str) -> bool:
        """True if the bank already holds memories — used to REUSE a backfilled bank across n runs."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{url}/v1/default/banks/{bank}/memories/list?limit=1", timeout=10) as r:
                d = json.loads(r.read())
            return bool(d.get("items") or d.get("memories") or d.get("total"))
        except Exception:
            return False

    @staticmethod
    def _delete_bank(url: str, bank: str) -> None:
        """Fresh-trial reset lives in the HARNESS now (the plugin has no user-facing reset — a fresh
        bank IS the reset path). 404/errors are fine (bank didn't exist)."""
        import urllib.request
        try:
            req = urllib.request.Request(f"{url}/v1/default/banks/{bank}", method="DELETE")
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass

    async def _plugin_backfill(self, task_json: str, task_id: str, run_id: str, bank: str, url: str) -> None:
        """Trigger the PLUGIN's own ingestion — AMB does not ingest. Build the task repo, then run the
        plugin's background `deepen` engine (the same one every session start fires) over that repo +
        the task's conversations, and POLL the plugin's `status` entry until `synced` — the exact
        observable contract a real deployment has (the backfill CLI no longer exists)."""
        # Reuse a bank already ingested by a previous run (n=3 without re-ingesting the same data).
        if os.environ.get("SDE_HSCODING_REUSE_BANK", "").lower() in ("1", "true") \
                and await asyncio.to_thread(self._bank_has_memories, url, bank):
            return
        await asyncio.to_thread(self._delete_bank, url, bank)
        plugin_dir = Path(os.path.expanduser(os.environ.get("SDE_HSCODING_PLUGIN_DIR",
                                         str(Path.home() / "dev" / "hs-coding-plugin-wt"
                                             / "hindsight-integrations" / "hindsight-coding-agents"))))
        deepen_js = plugin_dir / "dist" / "deepen.js"
        status_js = plugin_dir / "dist" / "status.js"
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
        # 2. run the plugin's deepen engine (it owns extraction/strategies/pages/git scope)
        bf = ["node", str(deepen_js), "--repo", str(src), "--bank", bank, "--api-url", url]
        conv = t.get("conversations") or []
        # chats to ingest = the task's own decision chat + a shared pool of DECOY conversations (noise:
        # long, codebase-related, no task policy) so chat retrieval is a real ranking problem, not a
        # 1-chat lookup. Decoys are pre-generated once from git history (gen/decoy_conversations.json).
        # a task may carry ONE chat (list of turns) or SEVERAL (list of lists — e.g. a rule set in an
        # early chat and AMENDED in a later one); seed_sessions on the vanilla side already handles both.
        if conv and isinstance(conv[0], list):
            chats = [{"id": f"{task_id}-chat{i}", "turns": c} for i, c in enumerate(conv) if c]
        else:
            chats = [{"id": task_id, "turns": conv}] if conv else []
        decoy_path = Path(os.environ.get("SDE_DECOY_CONVERSATIONS",
                          str(_REPO_ROOT / "sdebench" / "datasets" / "gen" / "decoy_conversations.json")))
        if os.environ.get("SDE_DECOYS", "1").lower() not in ("0", "false") and decoy_path.exists():
            for d in json.loads(decoy_path.read_text()):
                chats.append({"id": d.get("id", f"decoy-{len(chats)}"), "turns": d["turns"]})
        if chats:
            cf = base / "conversations.json"
            cf.write_text(json.dumps(chats))
            bf += ["--conversations", str(cf)]
        limit = os.environ.get("SDE_HSCODING_GIT_LIMIT")  # optional scope; unset => the plugin decides
        if limit:
            bf += ["--gitlog-limit", limit]
        await asyncio.to_thread(subprocess.run, bf, capture_output=True, text=True,
                                env={**os.environ}, timeout=1800)
        # 3. poll the plugin's sync status until the seeded memory is fully queryable — this is the
        # product's readiness contract (gitlog + pages present, extractions drained), not a guess.
        st_cmd = ["node", str(status_js), "--repo", str(src), "--bank", bank, "--api-url", url]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            p = await asyncio.to_thread(subprocess.run, st_cmd, capture_output=True, text=True,
                                        env={**os.environ}, timeout=120)
            try:
                if json.loads(p.stdout.strip().splitlines()[-1]).get("synced"):
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
        raise RuntimeError(f"hscoding ingest never reached synced for bank {bank}")

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
               "--history", arm, "--agent", self._agent, "--model", self._model, "--run-id", run_id]
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
