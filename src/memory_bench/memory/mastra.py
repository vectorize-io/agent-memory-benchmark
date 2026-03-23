import json
import os
import subprocess
import time
import uuid

from pathlib import Path

import httpx

from ..models import Document
from .base import MemoryProvider

_SERVER_DIR = Path.home() / ".cache" / "omb" / "mastra"

_PACKAGE_JSON = json.dumps(
    {
        "name": "mastra-bench-server",
        "version": "1.0.0",
        "type": "module",
        "dependencies": {
            "@mastra/core": "latest",
            "@mastra/memory": "latest",
            "@mastra/libsql": "latest",
            "@mastra/fastembed": "latest",
        },
        "devDependencies": {
            "mastra": "latest",
        },
    },
    indent=2,
)

# Agent with local FastEmbed embeddings + LibSQL vector store.
# Semantic recall (topK=10) over the full resource scope, no last-messages window.
_INDEX_TS = """\
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore, LibSQLVector } from '@mastra/libsql';
import { fastembed } from '@mastra/fastembed';

const storage = new LibSQLStore({ id: 'bench-storage', url: 'file:mastra.db' });
const vector = new LibSQLVector({ id: 'bench-vector', url: 'file:vectors.db' });

const memory = new Memory({
  storage,
  vector,
  embedder: fastembed,
  options: {
    semanticRecall: { topK: 10, messageRange: 2, scope: 'resource' },
    lastMessages: 10,
  },
});

const benchAgent = new Agent({
  name: 'bench-agent',
  instructions: 'You are a helpful assistant. Answer questions based only on what you remember from past conversations.',
  model: 'openai/gpt-4o-mini',
  memory,
});

export const mastra = new Mastra({
  agents: { 'bench-agent': benchAgent },
  storage,
});
"""


def _scaffold(server_dir: Path) -> None:
    (server_dir / "package.json").write_text(_PACKAGE_JSON)
    src = server_dir / "src" / "mastra"
    src.mkdir(parents=True, exist_ok=True)
    index_ts = src / "index.ts"
    # Only write if content changed, to avoid unnecessary Mastra rebuilds.
    if not index_ts.exists() or index_ts.read_text() != _INDEX_TS:
        index_ts.write_text(_INDEX_TS)


class MastraMemoryProvider(MemoryProvider):
    name = "mastra"
    description = "Mastra semantic recall with LibSQL store and FastEmbed embeddings. topK=10."
    kind = "local"
    link = "https://mastra.ai"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=mastra.ai"

    def __init__(self):
        self._base_url = os.environ.get("MASTRA_BASE_URL", "http://localhost:4111").rstrip("/")
        self._agent_id = os.environ.get("MASTRA_AGENT_ID", "bench-agent")
        self._api_key = os.environ.get("MASTRA_API_KEY")
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._proc: subprocess.Popen | None = None

    def initialize(self) -> None:
        if self._ping():
            return

        _SERVER_DIR.mkdir(parents=True, exist_ok=True)
        _scaffold(_SERVER_DIR)

        if not (_SERVER_DIR / "node_modules").exists():
            print("Installing Mastra dependencies (first run only)...")
            subprocess.run(["npm", "install", "--silent"], cwd=_SERVER_DIR, check=True)

        print(f"Starting Mastra dev server at {self._base_url} ...")
        self._proc = subprocess.Popen(
            ["npx", "mastra", "dev"],
            cwd=_SERVER_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 90
        while time.time() < deadline:
            if self._ping():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("Mastra server process exited unexpectedly")
            time.sleep(2)

        self._proc.terminate()
        raise RuntimeError("Mastra server did not become ready within 90 seconds")

    def cleanup(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _restart_server(self) -> None:
        """Terminate and restart the server to get a fresh libSQL read snapshot."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        # Brief pause to let libSQL checkpoint the WAL on shutdown
        time.sleep(2)
        self._proc = subprocess.Popen(
            ["npx", "mastra", "dev"],
            cwd=_SERVER_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self._ping():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("Mastra server failed to restart")
            time.sleep(2)
        self._proc.terminate()
        raise RuntimeError("Mastra server did not restart within 90 seconds")

    def _ping(self) -> bool:
        try:
            r = httpx.get(f"{self._base_url}/api/memory/status", timeout=2)
            return r.status_code < 500
        except httpx.RequestError:
            return False

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _generate(self, messages: list[dict] | str, resource_id: str, thread_id: str, read_only: bool = False) -> dict:
        """Call agent.generate() — the canonical way to interact with Mastra memory."""
        r = httpx.post(
            f"{self._base_url}/api/agents/{self._agent_id}/generate",
            headers=self._headers(),
            json={
                "messages": messages,
                "memory": {
                    "thread": thread_id,
                    "resource": resource_id,
                    **({"readOnly": True} if read_only else {}),
                },
                "modelSettings": {"temperature": 0},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def ingest(self, documents: list[Document]) -> None:
        # Mirror the Mastra benchmark: each document is its own thread under the user's resourceId.
        # agent.generate() triggers the memory processor which embeds + stores messages.
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            messages = doc.messages or [{"role": "user", "content": doc.content}]
            self._generate(messages, resource_id=uid, thread_id=doc.id)
        # Mastra embeds messages asynchronously; the libSQL read snapshot used by the
        # running server won't see new vectors until a fresh connection is opened.
        # Restart the server so it starts with a clean snapshot that includes all vectors.
        self._restart_server()

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        # Use the memory search endpoint directly — returns raw vector-recalled messages
        # without passing through the agent's LLM (which would give conversational replies).
        r = httpx.get(
            f"{self._base_url}/api/memory/search",
            headers=self._headers(),
            params={"agentId": self._agent_id, "searchQuery": query, "resourceId": uid, "limit": max(k, 30)},
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()

        results = raw.get("results", [])
        docs = []
        for entry in results:
            content = entry.get("content", "")
            if isinstance(content, dict):
                content = content.get("content") or " ".join(
                    p.get("text", "") for p in content.get("parts", []) if p.get("type") == "text"
                )
            if isinstance(content, str) and content.strip():
                docs.append(Document(id=entry.get("id", str(uuid.uuid4())), content=content))

        return docs, raw
