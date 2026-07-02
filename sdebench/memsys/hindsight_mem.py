"""Hindsight-backed memory for sdebench — replicates BOTH stages of the memsys+agentic-rerank design
using native Hindsight primitives:

  coarse retrieval  ->  recall(types=["world","experience"])   # raw facts only, observations disabled
  agentic rerank    ->  reflect(response_schema={ref_id,...})  # root-cause selection

The bank is configured via missions (the user's ask: "use hindsight mission to parse the inputs as you
like"): retain_mission turns git commit rationales + dev-chat decisions into raw decision facts and
preserves a REF-ID tracer in every fact; reflect_mission is a debugging persona that must pick the
single commit whose rationale explains the ROOT CAUSE (not a lexical twin). Observations are OFF.
"""
import os, asyncio
from pathlib import Path

# Local Docker by default (spin up: docker run … vectorize/hindsight-api:latest on :8888).
# Override with HINDSIGHT_BASE_URL. Do NOT default to cloud.
BASE = os.environ.get("HINDSIGHT_BASE_URL", "http://localhost:8888")

RETAIN_MISSION = (
    "You are ingesting a software project's history: git commit rationales and developer chat "
    "decisions. For each item, extract the concrete technical DECISION and the CAUSE/INVARIANT it "
    "protects as raw facts. Preserve every code identifier verbatim (e.g. getlist, __setitem__, "
    "OrderedMultiDict) and preserve the 'REF-ID: <token>' marker verbatim in every fact you extract "
    "from that item, so a fact can be traced to its source. Ignore mechanical noise: formatting, "
    "version bumps, license headers, changelog housekeeping.")

REFLECT_MISSION = (
    "You are a debugging assistant with access to the project's past decisions. Given a bug's "
    "SYMPTOM, identify the SINGLE past decision/commit whose rationale explains the ROOT CAUSE — not "
    "one that merely shares vocabulary with the symptom. Return that decision's REF-ID.")

REFLECT_SCHEMA = {"type": "object",
                  "properties": {"ref_id": {"type": "string"},
                                 "why": {"type": "string"}},
                  "required": ["ref_id"]}


def _key():
    k = os.environ.get("HINDSIGHT_CLOUD_KEY")
    if k:
        return k
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        envf = root / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("HINDSIGHT_CLOUD_KEY") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def client(timeout=120):
    import hindsight_client
    return hindsight_client.Hindsight(base_url=BASE, api_key=_key(), timeout=timeout)


def setup_bank(c, bank):
    """(Re)configure the bank: decision-extraction retain mission, root-cause reflect mission, NO observations."""
    c.create_bank(bank_id=bank, name="sdebench git+chat memory",
                  retain_mission=RETAIN_MISSION, enable_observations=False,
                  reflect_mission=REFLECT_MISSION)


def _content(cid, entry):
    src = "git commit rationale" if entry["kind"] == "decision" else "developer chat decision"
    return f"REF-ID: {cid}\nSOURCE: {src}\n{entry['text']}"


async def _aingest(bank, items, conc):
    import hindsight_client
    c = hindsight_client.Hindsight(base_url=BASE, api_key=_key(), timeout=180)
    sem = asyncio.Semaphore(conc)
    done = [0]

    async def one(cid, entry):
        async with sem:
            ctx = "git commit" if entry["kind"] == "decision" else "developer chat"
            await c.aretain(bank_id=bank, content=_content(cid, entry), context=ctx,
                            document_id=cid, tags=[f"cid:{cid}"])
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  retained {done[0]}/{len(items)}")
    await asyncio.gather(*[one(cid, e) for cid, e in items])
    c.close()


def ingest(bank, items, conc=8):
    """items: list of (cid, store-entry). Retains each as its own document (document_id=cid)."""
    asyncio.run(_aingest(bank, items, conc))


def coarse(c, bank, bug):
    """Coarse retrieval = recall raw facts only (observations disabled). Returns RecallResult list;
    each .document_id is the source note's cid."""
    return c.recall(bank_id=bank, query=bug, types=["world", "experience"], budget="mid",
                    max_tokens=4096).results


def rerank(c, bank, bug):
    """Agentic rerank = reflect with a response_schema forcing a single ref_id pick."""
    r = c.reflect(bank_id=bank,
                  query="Identify the SINGLE past decision whose rationale explains the ROOT CAUSE of "
                        "this bug, and return its REF-ID.\n\nBUG:\n" + bug,
                  response_schema=REFLECT_SCHEMA, budget="mid")
    return r.structured_output or {}
