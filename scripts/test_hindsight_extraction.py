"""
Interactive test script for Hindsight extraction quality on AMA-bench.
Creates a test bank, ingests a small slice of a trajectory, inspects the extracted facts,
then recalls against the problematic queries to evaluate quality.

Usage:
    uv run python scripts/test_hindsight_extraction.py
"""
import json
import time
from pathlib import Path

from hindsight import HindsightEmbedded
from hindsight_client_api.models import BankConfigUpdate
from hindsight_client_api.api.banks_api import BanksApi
from hindsight_client.hindsight_client import _run_async

from memory_bench.dataset.ama_bench import AmaBenchDataset
from memory_bench.memory.hindsight import _enrich_trajectory

# ── Config ────────────────────────────────────────────────────────────────────
BANK_ID = "omb-test-extraction"
PROFILE = "omb-test-extraction"
STEPS_TO_INGEST = 50  # covers steps 0-50 (includes all 5 test queries)

# ── Queries from the failing cases ───────────────────────────────────────────
TEST_QUERIES = [
    ("step42_ball_vanish",
     "At step 41, the ball was 1 step to the right. After the agent moved right in step 42, the ball vanished from the observation. What was the exact position of the agent relative to the ball at the end of step 42?",
     "agent on same tile as ball (relative position zero)"),

    ("step47_counterfactual",
     "In step 46, where is the DOOR text block? If at step 47 the agent had moved right instead of left, what would be the new relative position of DOOR?",
     "DOOR at (-3,0) → would become (-4,0) because right move shifts all relative positions left by 1"),

    ("step20_21_inversion",
     "Between Step 20 and Step 21, the agent's logged action is right. The rule win block moved from 3 steps left to 2 steps left. What was the agent's actual direction of movement?",
     "agent actually moved LEFT — controls are inverted"),

    ("step39_42_bypass",
     "In steps 39-42 the agent does up, up, right, right. What are the active rules at step 39? Is BALL IS PUSH active?",
     "only 'baba is you' is active — BALL IS PUSH not active, ball is immovable obstacle"),

    ("step7_8_loop",
     "Steps 7 to 10: down, up, down, up. The observations at step 8 and step 10 are identical to step 6. What caused this state reversion?",
     "down then up are inverse operations cancelling out, returning to step 6 state"),
]


def truncate_to_steps(content: str, max_steps: int) -> str:
    """Keep only the first max_steps steps from the trajectory."""
    lines = content.split("\n")
    result = []
    step_count = 0
    for line in lines:
        if line.startswith("Step ") and line.endswith(":"):
            step_count += 1
            if step_count > max_steps:
                break
        result.append(line)
    return "\n".join(result)


def make_entity_labels() -> list[dict]:
    return [
        {
            "key": "step_number",
            "description": "The exact trajectory step number(s) this fact is about. Use the integer step index as shown in 'Step N:' headers (e.g., '7', '42'). For facts spanning multiple steps write them comma-separated (e.g., '7,8'). For pattern facts covering a range, write 'start-end' (e.g., '7-10').",
            "type": "text",
            "tag": True,
            "optional": False,
        },
        {
            "key": "action",
            "description": "The action taken by the agent at this step, exactly as written.",
            "type": "value",
            "tag": True,
            "optional": True,
            "values": [
                {"value": "up", "description": "Agent moved up"},
                {"value": "down", "description": "Agent moved down"},
                {"value": "left", "description": "Agent moved left"},
                {"value": "right", "description": "Agent moved right"},
                {"value": "idle", "description": "Agent did not move"},
            ],
        },
        {
            "key": "step_pattern",
            "description": "Behavioral patterns detected at this step.",
            "type": "multi-values",
            "tag": True,
            "optional": True,
            "values": [
                {"value": "state_revert", "description": "This step's observation is identical to a previous step — the state reverted"},
                {"value": "inverse_pair", "description": "This step and the adjacent step are inverse actions (up/down or left/right) that cancel each other"},
                {"value": "loop", "description": "This step is part of a non-productive oscillation loop"},
                {"value": "object_vanished", "description": "An object present in the previous step is absent — agent is on the same tile"},
                {"value": "object_reappeared", "description": "An object reappeared after being absent — agent moved off that tile"},
            ],
        },
    ]


def make_custom_instructions() -> str:
    return """You are indexing agent trajectory logs from Baba Is You-style grid puzzle games.

IMPORTANT GAME MECHANICS: Objects listed as "rule `X`" (e.g., "rule `ball`", "rule `is`", "rule `win`") are TEXT BLOCKS that form rules — they are NOT the game entity itself. The actual game entities are unlabeled objects like "ball", "door", "baba". If "ball" is absent from the object list but "rule `ball`" is present, the game entity "ball" has vanished (agent occupies same tile).

For EACH step, you MUST do the following:

1. STEP FACT (required for EVERY step): "Step N: action=[X]. Active rules: [verbatim list]. Object positions: [list every game entity and rule block with exact (dx,dy) coordinates, where dx=positive means RIGHT, negative means LEFT; dy=positive means DOWN, negative means UP. Example: ball at (1,0) means 1 step right. door at (-3,-2) means 3 left, 2 up]. Rule implications: [e.g., 'baba is you: agent controls baba. NO ball is push: ball is an IMMOVABLE OBSTACLE. NO win rule: no win condition active']. Movement effect: [e.g., 'agent moved right → all relative positions shift: dx becomes dx-1']."
   PROHIBITED: Never write "positions were adjusted" or "positions were updated". Always list actual coordinates.

2. ABSENT OBJECT FACT (MANDATORY when any game entity disappears): For EVERY step N, compare the object list with step N-1. If a named game entity (e.g., "ball", "door") was listed at step N-1 but is NOT listed at step N (only "rule `ball`" remains), extract: "Step N: [entity] has VANISHED from the observation — the agent is occupying the same tile as [entity] (relative position (0,0)). The agent was at position (dx,dy) relative to [entity] at step N-1. This means the agent has reached [entity]'s tile."
   Set step_pattern=object_vanished.

3. STATE REVERT FACT (when step N observation matches prior step M): "Steps N and M have IDENTICAL observations — state reverted. The actions [A at N-1] then [B at N] are inverse operations that cancelled each other. Net change: zero."
   Set step_pattern=state_revert.

4. INVERSE PAIR FACT (when consecutive actions cancel): "Step N action=[X] and Step N+1 action=[Y] are inverse operations (e.g., right/left, down/up) — no net progress."
   Set step_pattern=inverse_pair.

5. COORDINATE COUNTERFACTUAL (for EVERY step): "Step N: agent moved [dir] → all objects shift by (ddx,ddy). If agent had moved [opposite dir] instead, all objects would shift by (-ddx,-ddy). Example: at step N, DOOR was at (-3,0); if agent moved right instead of left, DOOR would be at (-4,0)."

6. TASK FACT (once, for the first chunk only): verbatim task description.

Entity label assignments:
- step_number: exact step index(es)
- action: action taken
- step_pattern: detected patterns (state_revert, inverse_pair, loop, object_vanished, object_reappeared)

Do NOT summarize. Every step must produce its own complete fact with explicit coordinates."""


def main():
    import os
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    print(f"[1/5] Loading dataset...")
    ds = AmaBenchDataset()
    queries = ds.load_queries("test", limit=1)
    q = queries[0]
    docs = ds.load_documents("test", ids=set(q.gold_ids))
    doc = docs[0]
    content = truncate_to_steps(doc.content, STEPS_TO_INGEST)
    content = _enrich_trajectory(content)
    print(f"    doc {doc.id}: {len(doc.content)} chars total, using {len(content)} chars ({STEPS_TO_INGEST} steps, enriched)")

    print(f"\n[2/5] Creating bank '{BANK_ID}'...")
    client = HindsightEmbedded(
        profile=PROFILE,
        llm_provider="gemini",
        llm_model="gemini-2.5-flash-lite",
        llm_api_key=api_key,
    )

    # Delete existing bank
    try:
        client.banks.delete(bank_id=BANK_ID)
        print("    deleted existing bank")
    except Exception:
        pass

    # Create bank with custom extraction
    client.create_bank(
        bank_id=BANK_ID,
        name="AMA-bench extraction test",
        enable_observations=False,
        retain_extraction_mode="custom",
        retain_custom_instructions=make_custom_instructions(),
    )

    # Apply entity labels via low-level update_bank_config
    banks_api = BanksApi(client._api_client)
    result = _run_async(banks_api.update_bank_config(
        bank_id=BANK_ID,
        bank_config_update=BankConfigUpdate(updates={
            "entity_labels": make_entity_labels(),
        }),
    ))
    print(f"    entity labels set: {[l['key'] for l in result.overrides.get('entity_labels', [])]}")

    print(f"\n[3/5] Ingesting {STEPS_TO_INGEST} steps...")
    from hindsight_client_api.api.operations_api import OperationsApi
    ops_api = OperationsApi(client._api_client)

    resp = client.retain_batch(
        bank_id=BANK_ID,
        items=[{"content": content, "tags": ["user:test"]}],
        document_id=doc.id,
        retain_async=True,
    )
    print(f"    retain_batch resp: operation_id={resp.operation_id}")
    if resp.operation_id:
        print(f"    waiting for operation {resp.operation_id}...")
        while True:
            status = _run_async(ops_api.get_operation_status(bank_id=BANK_ID, operation_id=resp.operation_id))
            print(f"    status: {status.status}")
            if status.status in ("completed", "failed"):
                break
            time.sleep(2)

    # Poll until extraction completes (LLM extraction runs async in background worker)
    print(f"    waiting for LLM extraction to complete...")
    for attempt in range(120):
        memories = client.list_memories(bank_id=BANK_ID, limit=10)
        units = getattr(memories, "items", None) or getattr(memories, "memories", None) or []
        if units:
            print(f"    extraction done after ~{attempt * 2}s: {len(units)} facts so far (total={memories.total})")
            break
        time.sleep(2)
    else:
        print("    WARNING: extraction did not complete after 240s")

    print(f"\n[4/5] Inspecting extracted facts...")
    memories = client.list_memories(bank_id=BANK_ID, limit=500)
    units = getattr(memories, "items", None) or getattr(memories, "memories", None) or []
    print(f"    total facts extracted: {len(units)} (total={memories.total})")

    # Show facts by step_number entity label
    step_facts = {}
    for unit in units:
        if isinstance(unit, dict):
            text = unit.get("text", str(unit))
            entities_str = unit.get("entities", "")
            tags = unit.get("tags", [])
        else:
            text = getattr(unit, "text", str(unit))
            entities_str = getattr(unit, "entities", "")
            tags = getattr(unit, "tags", [])
        # Extract step_number from tags (format: "step_number:N")
        step = "?"
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("step_number:"):
                step = tag.split(":", 1)[1]
                break
        step_facts.setdefault(step, []).append((text, entities_str))

    # Show steps 6-10 and 39-43 and 41-42 specifically
    for step_of_interest in ["6", "7", "8", "9", "10", "39", "40", "41", "42", "43", "46", "47", "48"]:
        facts = step_facts.get(step_of_interest, [])
        if facts:
            print(f"\n  --- Step {step_of_interest} facts ({len(facts)}) ---")
            for text, entities in facts[:3]:
                print(f"    entities: {entities}")
                print(f"    text: {text[:200]}")
        else:
            print(f"\n  --- Step {step_of_interest}: NO FACTS ---")

    import re
    from memory_bench.llm.gemini import GeminiLLM
    from memory_bench.modes.rag import RAGMode
    from memory_bench.models import Document as BenchDocument

    llm = GeminiLLM()
    rag = RAGMode(llm=llm)

    def extract_step_tags(query: str) -> list[str]:
        matches = re.findall(r'\bstep\s+(\d+)\b', query, re.IGNORECASE)
        return [f"step_number:{n}" for n in dict.fromkeys(matches)]

    print(f"\n[5/5] Recall quality check (with step-number tag filtering + RAG answer)...")
    for name, query, expected in TEST_QUERIES:
        step_tags = extract_step_tags(query)
        kwargs: dict = dict(
            bank_id=BANK_ID,
            query=query,
            budget="high",
            max_tokens=8192,
            types=["world", "experience"],
            include_chunks=True,
        )
        if step_tags:
            kwargs["tags"] = step_tags
            kwargs["tags_match"] = "any"

        response = client.recall(**kwargs)
        results = response.results or []
        chunks = response.chunks or {}

        # Build context for RAG
        docs = []
        for r in results:
            lines = [r.text]
            if r.chunk_id and r.chunk_id in chunks:
                lines.append(f"> {chunks[r.chunk_id].text}")
            docs.append(BenchDocument(id=r.id, content="\n".join(lines)))
        context = "\n\n---\n\n".join(f"## Memory {i+1}\n{d.content}" for i, d in enumerate(docs))

        rag_result = rag.answer_from_context(query, context, task_type="open")

        print(f"\n  [{name}] step_tags={step_tags}")
        print(f"  Q: {query[:90]}")
        print(f"  Expected: {expected[:80]}")
        print(f"  Got {len(results)} results → RAG answer: {rag_result.answer[:150]}")
        for r in results[:3]:
            chunk_text = ""
            if r.chunk_id and r.chunk_id in chunks:
                chunk_text = f" | chunk: {chunks[r.chunk_id].text[:60]}"
            print(f"    - {r.text[:120]}{chunk_text}")


if __name__ == "__main__":
    main()
