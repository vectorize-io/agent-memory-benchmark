"""Check that the retain-side ingest controls reach bank creation.

Ingest cost on a large split is dominated by fact extraction, and extraction cost is
``corpus_chars / retain_chunk_size`` LLM calls — for BEAM-10M (~468M characters) that is ~156,000
calls per run at the server-side default chunk size of 3000. These two env vars are the levers, so
they are worth a check that they actually arrive rather than being silently dropped.

    uv run python scripts/test_bank_kwargs.py
"""
import os

from memory_bench.memory.hindsight import _HindsightBase


def _kwargs(dataset: str | None, **env) -> dict:
    prev = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        p = _HindsightBase.__new__(_HindsightBase)
        p._dataset = dataset
        return p._bank_kwargs()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> None:
    # Default: neither control is sent, so an existing run is byte-for-byte the run it always was.
    k = _kwargs("beam", AMB_HINDSIGHT_EXTRACTION_MODE=None, AMB_HINDSIGHT_CHUNK_SIZE=None)
    assert "retain_extraction_mode" not in k, k
    assert "retain_chunk_size" not in k, k
    assert "retain_mission" in k, "BEAM must still get its extraction mission by default"
    print("default            -> unchanged, mission present  ok")

    # chunks mode skips the LLM entirely. It is a DIFFERENT measurement — the mission above is an
    # extraction prompt and chunks mode ignores it — so this is opt-in, never a default.
    k = _kwargs("beam", AMB_HINDSIGHT_EXTRACTION_MODE="chunks", AMB_HINDSIGHT_CHUNK_SIZE=None)
    assert k["retain_extraction_mode"] == "chunks", k
    print("mode=chunks        -> retain_extraction_mode=chunks  ok")

    # Chunk size is the gentler lever: it keeps extraction and halves the call count per doubling.
    k = _kwargs("beam", AMB_HINDSIGHT_EXTRACTION_MODE=None, AMB_HINDSIGHT_CHUNK_SIZE="12000")
    assert k["retain_chunk_size"] == 12000 and isinstance(k["retain_chunk_size"], int), k
    print("chunk_size=12000   -> retain_chunk_size=12000 (int)  ok")

    # Non-BEAM datasets get the controls too, just no BEAM mission.
    k = _kwargs("locomo", AMB_HINDSIGHT_EXTRACTION_MODE="chunks", AMB_HINDSIGHT_CHUNK_SIZE=None)
    assert k["retain_extraction_mode"] == "chunks" and "retain_mission" not in k, k
    print("non-beam           -> controls apply, no beam mission  ok")

    print("\nall ok")


if __name__ == "__main__":
    main()
