# Lab Notes: MemoryHub AMB Submission

## Reproduction Protocol

### Environment
- **Cluster**: OpenShift (mcp-rhoai context)
- **MemoryHub version**: Build 33 (commit c4b3c06, 2026-08-21)
- **AMB harness**: Fork of vectorize-io/agent-memory-benchmark (main branch)
- **Python**: 3.13 (onnxruntime requires <=3.13)
- **memoryhub SDK**: 0.15.3

### Configuration
- **Memory provider**: `memoryhub` (cloud kind)
- **Answer LLM**: `gemini-3.5-flash-lite` (via OMB_ANSWER_LLM=gemini, OMB_ANSWER_MODEL=gemini-3.5-flash-lite)
- **Judge LLM**: `gemini-3.5-flash-lite` (via OMB_JUDGE_MODEL=gemini-3.5-flash-lite)
- **Retrieval depth (k)**: 70
- **Project isolation**: `amb-upstream-repro`
- **Ingestion mode**: library (verbatim conversation storage)

### Key env vars
```
MEMORYHUB_URL=<cluster route>/mcp/
MEMORYHUB_API_KEY=<from ~/.config/memoryhub/credentials>
MEMORYHUB_PROJECT_ID=amb-upstream-repro
MEMORYHUB_K=70
OMB_ANSWER_LLM=gemini
OMB_ANSWER_MODEL=gemini-3.5-flash-lite
```

### Commands
```bash
# Install dependencies (Python 3.13 required for onnxruntime compat)
uv sync --python 3.13

# Verify provider discovery
uv run amb providers  # should list 'memoryhub'

# Run the benchmark
uv run amb run \
    --dataset personamem \
    --split 32k \
    --memory memoryhub \
    --name memoryhub \
    --output-dir outputs
```

### Checkpointing
The AMB harness supports several resume modes:
- `--skip-ingestion`: Skip ingest, query existing memory state
- `--skip-ingested`: Resume mode for unit-sequential datasets
- `--skip-retrieval`: Re-run answer generation using cached contexts
- `--skip-answer`: Re-judge cached answers from previous run
- `--only-failed`: Restrict to queries that failed in the previous run

For re-runs against the same data, use `--skip-ingestion` to avoid
re-ingesting 195 documents.

## Issues Encountered

### 1. logical_id NOT NULL violation (resolved)
- **Symptom**: Write fails with IntegrityError on chunk creation
- **Root cause**: Migration 027 made `logical_id` NOT NULL, but
  `_create_chunk_children` and `create_fact_children` did not set it
- **Fix**: Set `logical_id=chunk_id` / `logical_id=fact_id` on child nodes
- **Commit**: c4b3c06

### 2. Large repo clone (738 MB)
- **Symptom**: `git clone` times out on slow connections
- **Workaround**: Use `--filter=blob:limit=1m` to skip large result blobs
- **Note**: Clone the fork on fast connections before benchmark sessions

### 3. Python 3.14 incompatibility
- **Symptom**: `onnxruntime` (required by cognee) has no cp314 wheels
- **Workaround**: Use `uv sync --python 3.13`

## Comparison with original result
- **Original (2026-07-16)**: 84.9% (500/589) with gemini-3.1-pro-preview
- **This run**: gemini-3.5-flash-lite (different answer model, expect score variance)
- **Note**: gemini-2.5-flash-lite deprecated mid-session (404); gemini-3.1-pro-preview
  depleted Gemini prepaid credits after ~540 queries with retries
- The retrieval quality is what we're measuring; answer accuracy depends on the LLM

## Differences from internal adapter
The upstream adapter (`src/memory_bench/memory/memoryhub.py`, ~130 lines) is
derived from our internal adapter (`benchmarks/amb-harness/.../memoryhub.py`,
503 lines). Key simplifications:

| Feature | Internal | Upstream |
|---------|----------|----------|
| Env vars | 20+ (ablation config) | 4 (URL, key, project, k) |
| Ingestion modes | library, dreaming, combined | library only |
| Search routing | pooled, split | pooled only |
| DB reset | Raw SQL DELETE | None (use fresh project) |
| Token budget | Configurable cap | None |
| Preflight search | Yes | No |
