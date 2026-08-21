# Issue Draft

**Title:** Add MemoryHub as an AMB memory provider

**Body:**

MemoryHub is a Kubernetes-native agent memory component for OpenShift AI. It stores verbatim conversations and retrieves via hybrid vector + keyword search with cross-encoder reranking and reciprocal-rank fusion. PostgreSQL + pgvector backend.

- **Repository:** https://github.com/redhat-ai-americas/memory-hub
- **Architecture:** Verbatim storage, Granite embeddings, hybrid search (vector + BM25 + keyword), cross-encoder reranking, reciprocal-rank fusion
- **License:** Apache 2.0
- **Kind:** cloud (hosted service, accessed via MCP SDK)

I'd like to contribute a provider adapter and PersonaMem 32k results. The adapter is ready and tested against the upstream harness.

---

# PR Draft

**Title:** Add MemoryHub memory provider

**Body:**

## Summary

Adds [MemoryHub](https://github.com/redhat-ai-americas/memory-hub) as a memory provider for AMB.

- **Architecture:** Verbatim conversation storage in PostgreSQL + pgvector. Hybrid search via vector + keyword + BM25 with cross-encoder reranking and reciprocal-rank fusion.
- **License:** Apache 2.0, self-hostable on OpenShift/Kubernetes
- **PersonaMem 32k result:** TBD% accuracy (TBD/589)
  - Reader: Gemini 3.5 Flash Lite
  - Judge: Gemini 3.5 Flash Lite
  - Embeddings: Granite (via server-side pipeline)

## Files

- `src/memory_bench/memory/memoryhub.py` -- provider adapter (~130 lines)
- `src/memory_bench/memory/__init__.py` -- registration
- `catalog.json` -- provider metadata
- `pyproject.toml` -- `memoryhub>=0.15` dependency
- `outputs/personamem/memoryhub/rag/32k.json.gz` -- results
- `results-manifest.json` -- manifest entry

## Setup

Requires a running MemoryHub instance (self-hosted or cloud):

```bash
pip install memoryhub
export MEMORYHUB_URL=https://your-instance/mcp/
export MEMORYHUB_API_KEY=your-api-key
export MEMORYHUB_PROJECT_ID=amb-benchmark  # optional, default
```

## Usage

```bash
amb run --dataset personamem --split 32k --memory memoryhub
```
