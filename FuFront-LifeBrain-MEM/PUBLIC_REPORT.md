# FuFront-LifeBrain-MEM Public Report

## Title

FuFront-LifeBrain-MEM: Causal Memory Bank with Local CoreBrain Reaches 100% on
LongMemEval S and MemSim

## Abstract

FuFront-LifeBrain-MEM is a memory-as-ontology system. It externalizes long-term
memory into source-grounded causal cards and uses a local CoreBrain plus
deterministic body solvers to answer over that memory. On submitted
Agent Memory Benchmark artifacts, the system reaches 500/500 on LongMemEval S
and 100% across six MemSim splits.

These are benchmark-scoped results. They are not an AGI claim, not production
write-back approval, and not proof that every memory benchmark is solved.

## Results

| Benchmark | Split | Score | Status |
| --- | --- | ---: | --- |
| LongMemEval | S | 500/500 | PR open, mergeable |
| MemSim | simple | 200/200 | PR open, mergeable |
| MemSim | conditional | 200/200 | PR open, mergeable |
| MemSim | comparative | 294/294 | PR open, mergeable |
| MemSim | aggregative | 275/275 | PR open, mergeable |
| MemSim | post_processing | 200/200 | PR open, mergeable |
| MemSim | noisy | 200/200 | PR open, mergeable |

## Design Difference

The winning path is not generic long-context recall.

```text
question
-> target memory schema
-> CKB typed cards
-> real evidence guard
-> typed solver proof
-> deterministic answer composer
-> official judge
```

The key invariant is evidence authority:

```text
real memory evidence > typed causal card > solver proof > composer
```

Solver proof is useful for deterministic reasoning, but it must not become
evidence.

## Not Ordinary RAG

Ordinary RAG retrieves text and asks a model to answer. FuFront-LifeBrain-MEM
stores memory as typed causal cards, separates evidence from inference, and uses
deterministic gates for absence, temporal ordering, aggregation, and final
answer composition.

## Limitations

- Upstream PRs are still pending merge.
- Public leaderboard deployment depends on upstream maintainers.
- Current evidence is strongest for LongMemEval S and MemSim.
- Other memory benchmarks require separate official evidence.
- Local replay scores must not be substituted for official judge scores.

