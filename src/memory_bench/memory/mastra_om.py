import json
import os
import subprocess
import time
import uuid

from pathlib import Path

import httpx

from ..models import Document
from .base import MemoryProvider

_SERVER_DIR = Path.home() / ".cache" / "omb" / "mastra-om"
_PORT = 4112

_PACKAGE_JSON = json.dumps(
    {
        "name": "mastra-om-bench-server",
        "version": "1.0.0",
        "type": "module",
        "dependencies": {
            "@mastra/core": "latest",
            "@mastra/memory": "latest",
            "@mastra/libsql": "latest",
            "@ai-sdk/openai": "latest",
        },
        "devDependencies": {
            "mastra": "latest",
        },
    },
    indent=2,
)

# ObservationalMemory agent — mirrors Mastra's LongMemEval 94.87% setup.
# Observer + Reflector both use Gemini 2.5 Flash.
# Main agent uses gpt-4o at eval time.
# No vector store — OM manages observations in LibSQL only.
_INDEX_TS = """\
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { createOpenAI } from '@ai-sdk/openai';

const storage = new LibSQLStore({ id: 'om-storage', url: 'file:mastra-om.db' });

const memory = new Memory({
  storage,
  options: {
    semanticRecall: false,
    lastMessages: 0,
    observationalMemory: {
      model: 'google/gemini-2.5-flash',
      scope: 'resource',
      observation: {
        messageTokens: 30000,
      },
      reflection: {
        observationTokens: 80000,
      },
    },
  },
});

const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Use a cheap model for ingest — the agent response doesn't matter,
// ObservationalMemory processors handle observation extraction.
const ingestAgent = new Agent({
  name: 'om-ingest-agent',
  instructions: 'You are a helpful assistant.',
  model: openai('gpt-4o-mini'),
  memory,
});

const evalAgent = new Agent({
  name: 'om-eval-agent',
  instructions: 'You are a helpful assistant. Answer questions based only on what you remember from past conversations. Be concise and direct.',
  model: openai('gpt-4o'),
  memory,
});

export const mastra = new Mastra({
  agents: {
    'om-ingest-agent': ingestAgent,
    'om-eval-agent': evalAgent,
  },
  storage,
  server: { port: 4112 },
});
"""


def _scaffold(server_dir: Path) -> None:
    (server_dir / "package.json").write_text(_PACKAGE_JSON)
    src = server_dir / "src" / "mastra"
    src.mkdir(parents=True, exist_ok=True)
    index_ts = src / "index.ts"
    if not index_ts.exists() or index_ts.read_text() != _INDEX_TS:
        index_ts.write_text(_INDEX_TS)


class MastraOMMemoryProvider(MemoryProvider):
    name = "mastra-om"
    description = "Mastra Observational Memory: observer/reflector pattern with Gemini 2.5 Flash + GPT-4o."
    kind = "local"
    link = "https://mastra.ai"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=mastra.ai"

    def __init__(self):
        self._base_url = os.environ.get("MASTRA_OM_BASE_URL", f"http://localhost:{_PORT}").rstrip("/")
        self._ingest_agent_id = "om-ingest-agent"
        self._eval_agent_id = "om-eval-agent"
        self._api_key = os.environ.get("MASTRA_OM_API_KEY")
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._proc: subprocess.Popen | None = None

    def initialize(self) -> None:
        if self._ping():
            return

        _SERVER_DIR.mkdir(parents=True, exist_ok=True)
        _scaffold(_SERVER_DIR)

        if not (_SERVER_DIR / "node_modules").exists():
            print("Installing Mastra OM dependencies (first run only)...")
            subprocess.run(["npm", "install", "--silent"], cwd=_SERVER_DIR, check=True)

        print(f"Starting Mastra OM dev server at {self._base_url} ...")
        self._proc = subprocess.Popen(
            ["npx", "mastra", "dev", "--port", str(_PORT)],
            cwd=_SERVER_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 120
        while time.time() < deadline:
            if self._ping():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("Mastra OM server process exited unexpectedly")
            time.sleep(2)

        self._proc.terminate()
        raise RuntimeError("Mastra OM server did not become ready within 120 seconds")

    def cleanup(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _restart_server(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        time.sleep(2)
        self._proc = subprocess.Popen(
            ["npx", "mastra", "dev", "--port", str(_PORT)],
            cwd=_SERVER_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._ping():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("Mastra OM server failed to restart")
            time.sleep(2)
        self._proc.terminate()
        raise RuntimeError("Mastra OM server did not restart within 120 seconds")

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

    def _generate(self, agent_id: str, messages: list[dict] | str, resource_id: str, thread_id: str) -> dict:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        r = httpx.post(
            f"{self._base_url}/api/agents/{agent_id}/generate",
            headers=self._headers(),
            json={
                "messages": messages,
                "memory": {
                    "thread": thread_id,
                    "resource": resource_id,
                },
                "modelSettings": {"temperature": 0},
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def ingest(self, documents: list[Document]) -> None:
        # Feed each document through the ingest agent so ObservationalMemory's
        # Observer processes and stores structured observations.
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            messages = doc.messages or [{"role": "user", "content": doc.content}]
            self._generate(self._ingest_agent_id, messages, resource_id=uid, thread_id=doc.id)
        # Restart so libSQL WAL is checkpointed and observations are visible.
        self._restart_server()

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        # Use a fresh thread per query so OM injects observations from the resource scope.
        thread_id = f"eval_{uuid.uuid4().hex}"
        raw = self._generate(self._eval_agent_id, query, resource_id=uid, thread_id=thread_id)

        text = raw.get("text", "")
        doc = Document(id=thread_id, content=text) if text.strip() else None
        return ([doc] if doc else []), raw
