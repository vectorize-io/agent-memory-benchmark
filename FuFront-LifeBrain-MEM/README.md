# FuFront-LifeBrain-MEM

Public evidence folder for Fufront-RyanX / LifeBrain memory benchmark results.

FuFront-LifeBrain-MEM 是 Fufront-RyanX / LifeBrain 記憶基準測試的公開證據資料夾。

This folder is intentionally scoped. It contains public benchmark evidence,
reproduction boundaries, and open-source staging notes. It does not contain
private memory, API keys, raw user data, unreleased model weights, or production
write-back configuration.

這個資料夾刻意保持邊界清楚：只放公開 benchmark 證據、重現邊界與分階段開源說明。不包含私有記憶、API key、原始使用者資料、未公開模型權重或 production write-back 設定。

## Current Evidence / 目前證據

### LongMemEval S

- Run name / 運行名稱: `Fufront-RyanX`
- Memory provider / 記憶系統: `ckb`
- Answer path / 回答路徑: `corebrain:ckb-body-v1`
- Judge / 裁判: `openai:gpt-4o`
- Oracle / 是否 oracle: `false`
- Total queries / 題數: `500`
- Correct / 正確: `500`
- Accuracy / 準確率: `100.0%`
- Artifact / 結果檔: `outputs/longmemeval/Fufront-RyanX/rag/s.json.gz`
- Artifact sha256:
  `bc692b10877d44a8669bbd1c10eef09ae333530c06235217170389820497ef1a`
- Submission PR / 提交 PR:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/18

### MemSim

- Run name / 運行名稱: `Fufront-RyanX`
- Memory provider / 記憶系統: `ckb`
- Oracle / 是否 oracle: `false`
- Submission PR / 提交 PR:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/17

| Split | Correct | Accuracy |
| --- | ---: | ---: |
| simple | 200/200 | 100.0% |
| conditional | 200/200 | 100.0% |
| comparative | 294/294 | 100.0% |
| aggregative | 275/275 | 100.0% |
| post_processing | 200/200 | 100.0% |
| noisy | 200/200 | 100.0% |

## Public Claim Boundary / 對外宣稱邊界

Safe claim / 安全宣稱:

```text
Fufront-RyanX CKB reached 500/500 on LongMemEval S using a local CoreBrain
plus causal memory bank, with OpenAI GPT-4o used only as the official judge.

Fufront-RyanX CKB 使用本地核心小腦 + 因果記憶庫，在 LongMemEval S 達到
500/500；OpenAI GPT-4o 僅作為官方裁判使用。
```

Unsafe claims / 不安全宣稱:

- Do not claim AGI from these benchmark results.
- 不要用這些 benchmark 結果宣稱 AGI。
- Do not claim upstream leaderboard deployment before the PR is merged.
- upstream PR 合併前，不要宣稱官方榜單已部署。
- Do not treat local replay scores as official judge evidence.
- 不要把本地 replay 分數當成 official judge 證據。
- Do not claim production shared-memory write-back is unlocked.
- 不要宣稱 production shared-memory write-back 已解鎖。
- Do not publish private memory, raw traces, API keys, or credentials.
- 不要發布私有記憶、原始 trace、API key 或憑證。

## Architecture Summary / 架構摘要

FuFront-LifeBrain-MEM is not ordinary RAG.

FuFront-LifeBrain-MEM 不是普通 RAG。

The intended architecture is:

目標架構是：

```text
question
-> target memory schema
-> causal memory bank typed cards
-> real evidence guard
-> typed solver proof
-> deterministic answer composer
-> official judge

問題
-> 目標記憶 schema
-> 因果記憶庫 typed cards
-> 真實證據守門
-> typed solver proof
-> deterministic answer composer
-> official judge
```

The core invariant is that solver-generated intermediate objects are not
evidence. Evidence must come from source-grounded memory cards.

核心不變量：solver 生成的中間物不能反過來當 evidence。證據必須來自有來源支撐的記憶卡片。

## Read Order / 閱讀順序

1. `EVIDENCE_PACKET.json`
2. `PUBLIC_REPORT.md`
3. `REPRODUCTION.md`
4. `OPEN_SOURCE_PLAN.md`
5. `MANIFEST_SHA256.txt`
