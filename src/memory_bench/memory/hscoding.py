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


def bank_path(path: str) -> str:
    """Quote the bank id (the first segment) — bank ids may carry characters a URL path can't."""
    from urllib.parse import quote
    bank, _, rest = path.partition("/")
    return quote(bank, safe="") + ("/" + rest if rest else "")


def bank_for(task_id: str) -> str:
    # SDE_HSCODING_BANK_PREFIX: a run of its own banks, so one campaign can never reset or reuse
    # another's — and a bank id a server has wedged can be stepped around.
    return f"{os.environ.get('SDE_HSCODING_BANK_PREFIX') or 'sde-coding'}-{task_id}"


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
            await asyncio.to_thread(self._refresh_pages, bank)
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
        # 2. the plugin's deepen engine (it owns extraction/strategies/pages/git scope).
        #    An empty config of its own: the runner's ~/.hindsight/coding-agent.json would otherwise
        #    win over the env (its apiToken, bank overrides, seed limits), so the ingest would depend
        #    on whose machine ran it — and could land in another tenant entirely.
        #    Pages are seeded "manual": their default cron trigger refreshes them up to an hour after
        #    the seed, so an agent starting right after `synced` would read pages built from nothing.
        #    Step 4 refreshes them once, explicitly, instead.
        cfg = base / "coding-agent.json"
        cfg.write_text(json.dumps({"pageTriggerType": "manual"}))
        cmd = ["node", str(self._plugin_dir / "dist" / "deepen.js"), "--repo", str(src),
               "--bank", bank, "--api-url", self._url, "--git-ingest", "full", "--config", str(cfg)]
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
        # deepen exits 0 even when items never reached the bank ("… (N items failed to enqueue)"),
        # and a bank missing part of its history scores the memory arm below what it is.
        if "failed to enqueue" in (p.stdout or "") + (p.stderr or ""):
            raise RuntimeError(f"deepen left items out of bank {bank}: "
                               f"{[l for l in (p.stdout or '').splitlines() if 'failed to enqueue' in l][-3:]}")
        # 3. poll the plugin's sync status until seeded memory is fully queryable
        st = ["node", str(self._plugin_dir / "dist" / "status.js"), "--repo", str(src),
              "--bank", bank, "--api-url", self._url, "--config", str(cfg)]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            sp = await asyncio.to_thread(subprocess.run, st, capture_output=True, text=True,
                                         env={**os.environ}, timeout=120)
            try:
                synced = json.loads(sp.stdout.strip().splitlines()[-1]).get("synced")
            except Exception:
                synced = False
            if synced:
                # 4. `synced` only means the pages EXIST (they are created before the history lands)
                await asyncio.to_thread(self._refresh_pages, bank)
                return
            await asyncio.sleep(5)
        raise RuntimeError(f"hscoding ingest never reached synced for bank {bank}")

    # ── retrieval: agent-side (plugin reflect+inject); nothing to serve here ─────
    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        return [], None

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        # Same token the plugin reads (HINDSIGHT_API_TOKEN) — an authenticated server otherwise 401s,
        # and the swallowed error would reuse a stale bank instead of resetting it.
        token = os.environ.get("HINDSIGHT_API_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _api(self, method: str, path: str) -> dict:
        import urllib.request
        req = urllib.request.Request(f"{self._url}/v1/default/banks/{bank_path(path)}", method=method,
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    def _refresh_pages(self, bank: str, timeout_s: int = 1800) -> None:
        """Refresh every knowledge page once over the seeded bank and wait for all of them.

        Fails the unit when a refresh fails or leaves a page empty: a memory arm injecting blank
        pages measures no memory at all, and must not be scored as if it had one.
        """
        pages = self._api("GET", f"{bank}/mental-models?limit=100").get("items") or []
        if not pages:
            raise RuntimeError(f"bank {bank} has no knowledge pages to refresh")
        ops = {m["id"]: self._api("POST", f"{bank}/mental-models/{m['id']}/refresh")["operation_id"]
               for m in pages}
        deadline = time.monotonic() + timeout_s
        pending = dict(ops)
        while pending and time.monotonic() < deadline:
            for mm_id, op_id in list(pending.items()):
                status = (self._api("GET", f"{bank}/operations/{op_id}").get("status") or "").lower()
                if status == "failed":
                    raise RuntimeError(f"knowledge page {mm_id} refresh failed on bank {bank}")
                if status in ("completed", "cancelled"):
                    del pending[mm_id]
            if pending:
                time.sleep(10)
        if pending:
            raise RuntimeError(f"{len(pending)} knowledge page refresh(es) still running on bank {bank}")
        full = self._api("GET", f"{bank}/mental-models?limit=100&detail=full").get("items") or []
        empty = [m["name"] for m in full if not (m.get("content") or "").strip()]
        if empty:
            raise RuntimeError(f"knowledge pages still empty after refresh on bank {bank}: {empty}")

    def _bank_has_memories(self, bank: str) -> bool:
        import urllib.request
        try:
            req = urllib.request.Request(f"{self._url}/v1/default/banks/{bank}/memories/list?limit=1",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            return bool(d.get("items") or d.get("memories") or d.get("total"))
        except Exception:
            return False

    def _delete_bank(self, bank: str) -> None:
        import urllib.error
        import urllib.request
        # Look before deleting: some deployments answer DELETE on a missing bank with a 500, not a
        # 404, and a missing bank is already reset.
        try:
            req = urllib.request.Request(f"{self._url}/v1/default/banks/{bank}/stats", headers=self._headers())
            urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return
            raise RuntimeError(f"could not check bank {bank}: HTTP {e.code}") from e
        try:
            req = urllib.request.Request(f"{self._url}/v1/default/banks/{bank}", method="DELETE",
                                         headers=self._headers())
            urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"could not reset bank {bank}: HTTP {e.code}") from e
