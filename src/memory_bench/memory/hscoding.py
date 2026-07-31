"""Hindsight coding-agents plugin as a MemoryProvider (the sdebench `hscoding` arm).

Conforms to the standard provider contract with one deliberate difference in each direction:

- INGEST is repo-native: instead of consuming the dataset's Document list, `async_ingest` builds the
  task's repo and runs the plugin's own background `deepen` engine over it + the task's
  conversations + the decoy pool — the exact ingestion a real deployment performs at session start —
  then polls the plugin's `status` entry until `synced` (the product's readiness contract). The
  Document list and the deepen inputs describe the SAME knowledge; the plugin simply owns its own
  extraction, strategies, and git scope.
- RETRIEVE is a no-op: delivery is agent-side (the plugin's reflect+inject inside the agent
  harness), so the coding mode does not inject anything for this provider. The retrieve() stub
  exists only to satisfy the interface for non-coding callers.

Per-task isolation: the sdebench dataset declares `isolation_unit = "task"`, so the runner ingests
one unit (task) at a time; the bank is `sde-coding-<task_id>`. `--skip-ingestion` reuses populated
banks across n-runs (replaces the old SDE_HSCODING_REUSE_BANK env, which is still honored).
"""
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import MemoryProvider
from ..models import Document

_REPO_ROOT = Path(__file__).resolve().parents[3]


def bank_for(task_id: str) -> str:
    return f"sde-coding-{task_id}"


class HsCodingProvider(MemoryProvider):
    name = "hindsight-coding"
    description = "Hindsight coding-agents plugin: deepen-engine ingestion, agent-side reflect+inject."
    kind = "local"
    provider = "hindsight"
    variant = "coding-plugin"
    link = "https://github.com/vectorize-io/hindsight"
    concurrency = int(os.environ.get("SDE_CONCURRENCY", "4"))

    def __init__(self) -> None:
        self._url = os.environ.get("SDE_HINDSIGHT_URL", "http://localhost:8888")
        self._skip = False

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def initialize(self) -> None:
        plugin_dir = Path(os.path.expanduser(os.environ.get("SDE_HSCODING_PLUGIN_DIR", "")))
        if not plugin_dir.name or not (plugin_dir / "dist" / "deepen.js").exists():
            raise RuntimeError("memory=hindsight-coding needs SDE_HSCODING_PLUGIN_DIR -> a "
                               "hindsight-coding-agents checkout with dist/ built")
        self._plugin_dir = plugin_dir

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        # --skip-ingestion (or the legacy env) => reuse populated banks; otherwise fresh-trial reset.
        self._skip = (not reset) or os.environ.get("SDE_HSCODING_REUSE_BANK", "").lower() in ("1", "true")
        if not self._skip:
            for uid in unit_ids or set():
                self._delete_bank(bank_for(uid))

    # ── ingestion (the plugin's own engine) ──────────────────────────────────────
    def ingest(self, documents: list[Document]) -> None:
        asyncio.run(self.async_ingest(documents))

    async def async_ingest(self, documents: list[Document]) -> None:
        if not documents:
            return
        task_id = documents[0].user_id
        if not task_id:
            return
        bank = bank_for(task_id)
        if self._skip and await asyncio.to_thread(self._bank_has_memories, bank):
            return
        from ..dataset.sdebench import task_json_path
        tj = task_json_path(task_id)
        t = json.loads(tj.read_text())
        build_py = tj.parents[2] / t.get("build", "build.py")
        base = Path("/tmp/sdebench/omb-backfill") / task_id
        src = base / "repo"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        # 1. build the task repo (deepen reads its git history — the same knowledge the dataset's
        #    git-commit Documents describe, in its native form)
        bp = await asyncio.to_thread(subprocess.run, ["python", str(build_py), str(src)],
                                     capture_output=True, text=True, env={**os.environ})
        if bp.returncode != 0 or not (src / ".git").exists():
            raise RuntimeError(f"task repo build failed for {task_id} (rc={bp.returncode}): "
                               f"{(bp.stderr or bp.stdout or '')[-200:]}")
        # 2. the plugin's deepen engine (it owns extraction/strategies/pages/git scope)
        cmd = ["node", str(self._plugin_dir / "dist" / "deepen.js"), "--repo", str(src),
               "--bank", bank, "--api-url", self._url, "--git-ingest", "full"]
        chats = [{"id": d.id, "turns": [{"role": m["role"], "text": m["content"]}
                                        for m in (d.messages or [])]}
                 for d in documents if d.messages]
        if chats:
            cf = base / "conversations.json"
            cf.write_text(json.dumps(chats))
            cmd += ["--conversations", str(cf)]
        limit = os.environ.get("SDE_HSCODING_GIT_LIMIT")
        if limit:
            cmd += ["--gitlog-limit", limit]
        p = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True,
                                    env={**os.environ}, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"deepen failed (rc={p.returncode}) for bank {bank}: "
                               f"{(p.stderr or p.stdout or '')[-300:]}")
        # 3. poll the plugin's sync status until seeded memory is fully queryable
        st = ["node", str(self._plugin_dir / "dist" / "status.js"), "--repo", str(src),
              "--bank", bank, "--api-url", self._url]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            sp = await asyncio.to_thread(subprocess.run, st, capture_output=True, text=True,
                                         env={**os.environ}, timeout=120)
            try:
                if json.loads(sp.stdout.strip().splitlines()[-1]).get("synced"):
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
        raise RuntimeError(f"hscoding ingest never reached synced for bank {bank}")

    # ── retrieval: agent-side (plugin reflect+inject); nothing to serve here ─────
    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        return [], None

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _bank_has_memories(self, bank: str) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(
                    f"{self._url}/v1/default/banks/{bank}/memories/list?limit=1", timeout=10) as r:
                d = json.loads(r.read())
            return bool(d.get("items") or d.get("memories") or d.get("total"))
        except Exception:
            return False

    def _delete_bank(self, bank: str) -> None:
        import urllib.request
        try:
            req = urllib.request.Request(f"{self._url}/v1/default/banks/{bank}", method="DELETE")
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass
