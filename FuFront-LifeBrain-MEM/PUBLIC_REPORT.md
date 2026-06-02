# FuFront-LifeBrain-MEM Public Report / 公開報告

## Title / 標題

FuFront-LifeBrain-MEM: Causal Memory Bank with Local CoreBrain Reaches 100% on
LongMemEval S and MemSim

FuFront-LifeBrain-MEM：本地核心小腦 + 因果記憶庫在 LongMemEval S 與 MemSim 達到 100%

## Abstract / 摘要

FuFront-LifeBrain-MEM is a memory-as-ontology system. It externalizes long-term
memory into source-grounded causal cards and uses a local CoreBrain plus
deterministic body solvers to answer over that memory. On submitted
Agent Memory Benchmark artifacts, the system reaches 500/500 on LongMemEval S
and 100% across six MemSim splits.

FuFront-LifeBrain-MEM 是一套 memory-as-ontology 系統。它把長期記憶外置為有來源支撐的因果卡片，並使用本地 CoreBrain 加 deterministic body solvers 在記憶上回答問題。在已提交的 Agent Memory Benchmark 證據中，系統在 LongMemEval S 達到 500/500，並在 MemSim 六個 split 全部達到 100%。

These are benchmark-scoped results. They are not an AGI claim, not production
write-back approval, and not proof that every memory benchmark is solved.

這些結果只限於 benchmark 證據邊界內。它們不是 AGI 宣稱，不是 production write-back 授權，也不是所有記憶 benchmark 都已解決的證明。

## Results / 結果

| Benchmark | Split | Score | Status |
| --- | --- | ---: | --- |
| LongMemEval | S | 500/500 | PR open, mergeable / PR 已開、可合併 |
| MemSim | simple | 200/200 | PR open, mergeable / PR 已開、可合併 |
| MemSim | conditional | 200/200 | PR open, mergeable / PR 已開、可合併 |
| MemSim | comparative | 294/294 | PR open, mergeable / PR 已開、可合併 |
| MemSim | aggregative | 275/275 | PR open, mergeable / PR 已開、可合併 |
| MemSim | post_processing | 200/200 | PR open, mergeable / PR 已開、可合併 |
| MemSim | noisy | 200/200 | PR open, mergeable / PR 已開、可合併 |

## Design Difference / 設計差異

The winning path is not generic long-context recall.

成功路徑不是泛用長上下文回憶。

```text
question
-> target memory schema
-> CKB typed cards
-> real evidence guard
-> typed solver proof
-> deterministic answer composer
-> official judge

問題
-> 目標記憶 schema
-> CKB typed cards
-> 真實證據守門
-> typed solver proof
-> deterministic answer composer
-> official judge
```

The key invariant is evidence authority:

關鍵不變量是證據權限：

```text
real memory evidence > typed causal card > solver proof > composer

真實記憶證據 > typed causal card > solver proof > composer
```

Solver proof is useful for deterministic reasoning, but it must not become
evidence.

Solver proof 對 deterministic reasoning 有用，但它不能變成 evidence。

## Not Ordinary RAG / 不是普通 RAG

Ordinary RAG retrieves text and asks a model to answer. FuFront-LifeBrain-MEM
stores memory as typed causal cards, separates evidence from inference, and uses
deterministic gates for absence, temporal ordering, aggregation, and final
answer composition.

普通 RAG 通常是檢索文本再讓模型回答。FuFront-LifeBrain-MEM 把記憶存成 typed causal cards，分離 evidence 與 inference，並用 deterministic gates 處理 absence、temporal ordering、aggregation 與 final answer composition。

## Limitations / 限制

- Upstream PRs are still pending merge.
- upstream PR 仍等待合併。
- Public leaderboard deployment depends on upstream maintainers.
- 官方榜單部署取決於 upstream 維護者。
- Current evidence is strongest for LongMemEval S and MemSim.
- 目前最強證據集中在 LongMemEval S 與 MemSim。
- Other memory benchmarks require separate official evidence.
- 其他記憶 benchmark 需要獨立 official evidence。
- Local replay scores must not be substituted for official judge scores.
- 本地 replay 分數不能替代 official judge 分數。
