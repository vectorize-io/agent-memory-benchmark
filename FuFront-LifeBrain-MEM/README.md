# FuFront-LifeBrain-MEM

Public evidence folder for Fufront-RyanX / LifeBrain memory benchmark results.

This folder is intentionally scoped. It contains public benchmark evidence,
reproduction boundaries, and open-source staging notes. It does not contain
private memory, API keys, raw user data, unreleased model weights, or production
write-back configuration.

## Current Evidence

### LongMemEval S

- Run name: `Fufront-RyanX`
- Memory provider: `ckb`
- Answer path: `corebrain:ckb-body-v1`
- Judge: `openai:gpt-4o`
- Oracle: `false`
- Total queries: `500`
- Correct: `500`
- Accuracy: `100.0%`
- Artifact: `outputs/longmemeval/Fufront-RyanX/rag/s.json.gz`
- Artifact sha256:
  `bc692b10877d44a8669bbd1c10eef09ae333530c06235217170389820497ef1a`
- Submission PR:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/18

### MemSim

- Run name: `Fufront-RyanX`
- Memory provider: `ckb`
- Oracle: `false`
- Submission PR:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/17

| Split | Correct | Accuracy |
| --- | ---: | ---: |
| simple | 200/200 | 100.0% |
| conditional | 200/200 | 100.0% |
| comparative | 294/294 | 100.0% |
| aggregative | 275/275 | 100.0% |
| post_processing | 200/200 | 100.0% |
| noisy | 200/200 | 100.0% |

## Public Claim Boundary

Safe claim:

```text
Fufront-RyanX CKB reached 500/500 on LongMemEval S using a local CoreBrain
plus causal memory bank, with OpenAI GPT-4o used only as the official judge.
```

Unsafe claims:

- Do not claim AGI from these benchmark results.
- Do not claim upstream leaderboard deployment before the PR is merged.
- Do not treat local replay scores as official judge evidence.
- Do not claim production shared-memory write-back is unlocked.
- Do not publish private memory, raw traces, API keys, or credentials.

## Architecture Summary

FuFront-LifeBrain-MEM is not ordinary RAG. The intended architecture is:

```text
question
-> target memory schema
-> causal memory bank typed cards
-> real evidence guard
-> typed solver proof
-> deterministic answer composer
-> official judge
```

The core invariant is that solver-generated intermediate objects are not
evidence. Evidence must come from source-grounded memory cards.

## Read Order

1. `EVIDENCE_PACKET.json`
2. `PUBLIC_REPORT.md`
3. `REPRODUCTION.md`
4. `OPEN_SOURCE_PLAN.md`
5. `MANIFEST_SHA256.txt`

