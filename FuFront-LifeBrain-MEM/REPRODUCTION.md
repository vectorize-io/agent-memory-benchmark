# Reproduction and Verification Notes / 重現與驗證說明

## Current Submission PRs / 目前提交 PR

- LongMemEval:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/18
- MemSim:
  https://github.com/vectorize-io/agent-memory-benchmark/pull/17

Both were verified on 2026-06-02 as open, mergeable, and not merged.

兩個 PR 在 2026-06-02 驗證為 open、mergeable、not merged。

## Manifest Files / Manifest 檔案

The public benchmark entries are recorded in:

公開 benchmark 條目記錄於：

- `results-manifest.json`
- `blob-manifest.json`
- `.blob_manifest.json`

LongMemEval artifact / LongMemEval 結果檔：

```text
outputs/longmemeval/Fufront-RyanX/rag/s.json.gz
sha256: bc692b10877d44a8669bbd1c10eef09ae333530c06235217170389820497ef1a
```

## Official vs Local Boundary / 官方與本地邊界

Use official judge artifacts for public claims. Local replay is useful for
debugging typed solvers and composer behavior, but it is not equivalent to
official OpenAI judge scoring.

對外宣稱必須使用 official judge artifact。本地 replay 適合用來 debug typed solvers 與 composer 行為，但不等同於 official OpenAI judge scoring。

## What Should Be Added Before a Standalone Paper or Repo

## 獨立論文或 repo 前應補齊的證據

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

中文：

1. 完整 official command transcript。
2. 環境快照。
3. warm local answer latency。
4. official end-to-end latency。
5. 已移除 secrets 與 private memory 的 per-question trace。
6. Ablation table：
   - retrieval baseline
   - CKB typed cards
   - typed solvers
   - absence guard
   - deterministic composer
