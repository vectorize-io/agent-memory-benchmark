# Reproduction and Verification Notes

## Current Submission PRs

- LongMemEval:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/18
- MemSim:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/17

Both were verified on 2026-06-02 as open, mergeable, and not merged.

## Manifest Files

The public benchmark entries are recorded in:

- `results-manifest.json`
- `blob-manifest.json`
- `.blob_manifest.json`

LongMemEval artifact:

```text
outputs/longmemeval/Fufront-RyanX/rag/s.json.gz
sha256: bc692b10877d44a8669bbd1c10eef09ae333530c06235217170389820497ef1a
```

## Official vs Local Boundary

Use official judge artifacts for public claims. Local replay is useful for
debugging typed solvers and composer behavior, but it is not equivalent to
official OpenAI judge scoring.

## What Should Be Added Before a Standalone Paper or Repo

1. Full official command transcript.
2. Environment snapshot.
3. Warm local answer latency.
4. Official end-to-end latency.
5. Per-question trace with secrets and private memory removed.
6. Ablation table:
   - retrieval baseline
   - CKB typed cards
   - typed solvers
   - absence guard
   - deterministic composer

