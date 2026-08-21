# AMB — Agent Memory Benchmark

We built AMB because we wanted to be honest about how Hindsight performs — and because no existing benchmark gave us the full picture. AMB is fully open: datasets, prompts, scoring logic, and results.

Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**

## The problem with existing benchmarks

LoComo and LongMemEval are solid datasets, but they were designed for an era of 32k context windows. State-of-the-art models now have million-token context windows — on most instances, a naive "dump everything into context" approach scores competitively, not because it's a good memory architecture, but because retrieval has become the easy part. The benchmarks can no longer tell them apart.

Both datasets were also built around chatbot use cases. Agents today don't just answer questions about conversation history — they research, plan, execute multi-step tasks, and build knowledge across many interactions. AMB adds datasets that focus on agentic tasks: memory across tool calls, knowledge built from document research, preferences applied to multi-step decisions.

## What AMB measures

A memory system that scores 90% accuracy but costs $10 per user per day is not better than one that scores 82% and costs $0.10. AMB starts from accuracy because it's the hardest to fake, and tracks speed and token cost alongside it.

The only credible benchmark result is one you can reproduce yourself. AMB publishes everything: the evaluation harness, judge prompts, answer generation prompts, and the exact models used. Small changes to any of these can swing accuracy scores by double digits — we publish all of them.

## How it works

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — a Gemini model produces an answer from the retrieved context
4. **Judge** — a second Gemini call scores the answer against gold answers

Retrieval time is tracked separately from generation; ingestion time is also recorded.

## Setup

```bash
# Copy and fill in your API key
cp .env.example .env   # or just create .env with:
# GEMINI_API_KEY=...
```

## Usage

```bash
# List available datasets, memory providers, and modes
uv run amb providers

# List domains for a dataset
uv run amb domains --dataset personamem

# Run a benchmark
uv run amb run --dataset personamem --domain 32k --memory bm25

# Limit scale for a quick test
uv run amb run --dataset personamem --domain 32k --memory bm25 --query-limit 20

# Oracle mode: ingest only gold documents (tests generation quality in isolation)
uv run amb run --dataset personamem --domain 32k --memory bm25 --oracle

# Dataset statistics
uv run amb dataset-stats --dataset personamem

# Browse results in the browser
uv run amb view
```

## Results

Results are saved to `outputs/{dataset}/{memory}/{mode}/{domain}.json` and can be explored with `uv run amb view`.

## Requirements

- Python ≥ 3.11
- `GEMINI_API_KEY` in `.env` or environment
- For MemBench: set `MEMBENCH_DATA_PATH` to your local data directory

---

## ValorBrain memory provider

**ValorBrain** is a hybrid memory engine for AI agents: BM25 + dense vectors (pgvector) + reciprocal-rank fusion + entity-graph reranking + cross-encoder rerank, on PostgreSQL with row-level security, multi-tenant. The provider talks to a running engine over its REST API — retrieval goes through `/api/v1/memory/prepare`, the same production path agents use at runtime (full funnel: consolidation + timeline + snippet delivery), falling back to `/search` (hybrid) when prepare returns nothing. On ingest, ~100k-char AMB chunks are re-windowed into 8k-char windows with 800-char overlap, matching production conversational windows.

### BEAM-100K result

| | |
|---|---|
| **Accuracy (avg score per criterion)** | **71.4%** |
| Binary correct | 304/400 (76.0%) |
| Reader (answer) | Gemini 3.6 Flash (AGY gateway) |
| Judge | GLM-5.2 (Z.ai) — different family from the reader |
| Avg retrieve time | 5.2s |
| Avg context tokens | 12.6k |

Run file: `outputs/beam/valorbrain/rag/100k.json` (400/400 queries, `--split 100k --memory valorbrain`).

Reproduction: set `VALORBRAIN_URL`, `VALORBRAIN_TOKEN`, `VALORBRAIN_BENCHMARK_TENANT_ID` to a running engine + benchmark tenant, plus reader/judge keys, then `uv run amb run --dataset beam --split 100k --memory valorbrain --name valorbrain`. The judge is overridable via `OMB_JUDGE_LLM`/`OMB_JUDGE_MODEL` (see Known patches below) — we pin a judge from a different model family than the reader to avoid self-preference bias.

The result was independently reproduced with a different reader/judge pair (stealth "Ox Alpha" reader via OpenCode Zen + GLM judge: 70.2% avg score, 76.0% binary on the same 400 queries), supporting that the score measures the memory system rather than one particular reader.
