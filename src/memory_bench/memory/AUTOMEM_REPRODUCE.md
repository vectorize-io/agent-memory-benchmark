# Reproducing the AutoMem results

AutoMem is a graph + vector memory service (FalkorDB + Qdrant behind a Flask API). The
provider is **self-spinning**: it brings the whole stack up via Docker, runs the benchmark,
and tears it down. You need **Docker** and a **GEMINI_API_KEY** (the shared answer/judge
key every provider uses) — **no embedding API keys**.

## One command per split

```bash
GEMINI_API_KEY=... \
OMB_ANSWER_LLM=gemini OMB_ANSWER_MODEL=gemini-3.1-pro-preview \
OMB_JUDGE_LLM=gemini  OMB_JUDGE_MODEL=gemini-2.5-flash-lite \
uv run omb run --memory automem --dataset locomo --split locomo10
```

Swap `--dataset/--split` for the others (`longmemeval/s`, `personamem/32k`,
`beam/{100k,500k,1m,10m}`); the env block stays identical. `initialize()` runs
`docker compose up` on `automem_compose.yml` (AutoMem + FalkorDB + Qdrant), waits for
`/health`, and an atexit-registered `cleanup()` runs `docker compose down -v`. Ports are
chosen per run, so concurrent/repeat runs don't collide; a crash also tears the stack down.

## Pinned configuration

| Knob | Value |
|---|---|
| AutoMem image | `ghcr.io/verygoodplugins/automem:amb-v1` (override with `AUTOMEM_IMAGE`) |
| FalkorDB | `falkordb/falkordb:v4.18.3` (pinned) |
| Qdrant | `qdrant/qdrant:v1.11.3` (pinned) |
| Embeddings | FastEmbed local, `BAAI/bge-base-en-v1.5`, 768-dim (`EMBEDDING_PROVIDER=local`, no API key) |
| Answer LLM | `gemini-3.1-pro-preview` |
| Judge LLM | `gemini-2.5-flash-lite` |
| Mode | `rag` |

## How the provider uses AutoMem

- **Ingest** → `POST /memory/batch`, one memory per document, chunked at `AUTOMEM_MAX_CHARS`
  (default 1800) on sentence/paragraph boundaries with timestamps backdated to the source.
  After ingest the provider waits for AutoMem's enrichment queue to settle so the graph it
  queries is fully built.
- **Retrieve** → `GET /recall`, scoped to the run's tags, with graph relation expansion on
  (`expand_relations` + `expand_respect_tags`). Content is read from
  `result["memory"]["content"]`.

## Reported metrics

Each run writes per-query `retrieve_time_ms`, `context_tokens`, and `correct`/`score` to
`outputs/{dataset}/{name}/rag/{split}.json`. Recall latency is wall-clock around
`memory.retrieve()` on local hardware (FastEmbed in-process, single-query/RAG mode) and is
environment-relative — summarize with the median (P50), and treat it as a per-run figure,
not a cross-system axis.

## Tuning knobs (env)

| Env | Default | Purpose |
|---|---|---|
| `AUTOMEM_IMAGE` | `ghcr.io/verygoodplugins/automem:amb-v1` | AutoMem image tag |
| `AUTOMEM_MAX_CHARS` | `1800` | chunk size (under AutoMem's 2000 hard limit) |
| `AUTOMEM_RECALL_K` | (harness `k`) | override retrieval depth |
| `AUTOMEM_ENRICH_SETTLE_SECONDS` | `120` | max wait for enrichment to drain after ingest |
| `AUTOMEM_HOST` / `AUTOMEM_*_PORT` | localhost / free | override where the provider reaches the stack |

## Tests

```bash
uv run --with pytest pytest tests/test_automem_provider.py
```
(The provider's HTTP contract and chunking/extraction helpers are unit-tested without Docker.)
