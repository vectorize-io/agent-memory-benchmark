# Staged Open-Source Plan / 分階段開源計畫

## Verdict / 裁決

ALLOW staged open-source.

允許分階段開源。

BLOCK naked repository dump.

禁止裸開源整包亂丟。

## Why Staging Is Required / 為什麼必須分階段

The value is not just code. The value is the architecture discipline:

價值不只是代碼，而是整套架構紀律：

- memory as ontology / 記憶即本體
- typed causal cards / typed causal cards
- evidence and inference separation / evidence 與 inference 分離
- deterministic body solvers / deterministic body solvers
- absence authority guard / absence authority guard
- official judge parity / official judge 對齊
- anti-overclaim gates / 反過度宣稱 gate

If released as a loose repo, the design can be misread as ordinary RAG or a
benchmark-specific patch collection.

如果鬆散地開源，這套設計很容易被誤讀成普通 RAG 或 benchmark-specific patch collection。

## Stage 0: Evidence Freeze / 階段 0：證據凍結

Freeze:

凍結：

- benchmark table / benchmark 表格
- artifact sha256 / artifact sha256
- PR links / PR 連結
- command transcript / command transcript
- limitation text / 限制聲明
- no-secret audit / 無 secrets 審計

## Stage 1: Reference Implementation / 階段 1：參考實作

Open:

可公開：

- card schema
- edge schema
- evidence guard
- typed solver examples
- benchmark harness adapter
- trace visualizer

Do not open:

暫不公開：

- private memory banks / 私有記憶庫
- raw user traces / 原始使用者 trace
- API keys / API keys
- production write-back config / production write-back 設定
- unreleased model weights / 未公開模型權重
- benchmark-specific cleanup scripts without context / 沒有上下文的 benchmark-specific cleanup scripts

## Stage 2: Multi-Benchmark Expansion / 階段 2：多 benchmark 擴展

Next public targets:

下一批公開目標：

1. Maintained LongMemEval-style leaderboard or report.
2. LoCoMo.
3. LifeBench.
4. PersonaMem.
5. Any active memory leaderboard with reproducible submission rules.

中文：

1. 有維護的 LongMemEval 類 leaderboard 或 report。
2. LoCoMo。
3. LifeBench。
4. PersonaMem。
5. 任何有清楚提交規則、可重現的活躍記憶排行榜。

## Stage 3: Full Public Package / 階段 3：完整公開包

Publish a clean reference package only after the evidence and no-secret gates
are frozen.

只有在 evidence gate 與 no-secret gate 凍結後，才發布乾淨的 reference package。
