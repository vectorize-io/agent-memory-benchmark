// What a headline number is. Judged and MCQ datasets score "accuracy"; knowledge-bank
// retrieval datasets score a ranked metric ("ndcg@10", "recall@10"). The number lives in the
// same `accuracy` field either way, so every place that shows it labels it with this.

export const isAccuracy = type => !type || type === 'accuracy'

export const scoreLabel = type =>
  isAccuracy(type) ? 'Accuracy' : type.replace(/^ndcg/i, 'NDCG').replace(/^recall/i, 'Recall')

// Accuracy as a percentage; ranked metrics on the 0–100 scale leaderboards print (62.25), no "%".
export const fmtScore = (v, type) =>
  v == null ? '—' : isAccuracy(type) ? (v * 100).toFixed(1) + '%' : (v * 100).toFixed(2)

export const isKnowledge = catalogDataset => catalogDataset?.section === 'knowledge'

// Leaderboards that publish one number over several splits (ViDoRe's 8 tasks, MIRACL's 18
// languages) compare to the mean over the splits. Add that mean as the dataset's average split,
// one row per run that covers every split; a run missing a split gets no average.
export function withAverageSplit(entries, catalogDataset) {
  const avg = catalogDataset?.average_split
  const splits = catalogDataset?.splits ?? []
  if (!avg || !splits.length) return entries
  const runs = {}
  for (const e of entries) {
    if (!splits.includes(e.split) || e.accuracy == null) continue
    ;(runs[e.run_name + '||' + e.mode] ??= {})[e.split] = e
  }
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length
  const averages = Object.values(runs)
    .filter(bySplit => splits.every(s => bySplit[s]))
    .map(bySplit => {
      const parts = splits.map(s => bySplit[s])
      const times = parts.map(p => p.avg_retrieve_time_ms).filter(v => v != null)
      return {
        ...parts[0],
        split: avg,
        path: null, // no single result file: open a split to drill into queries
        average_of: parts.length,
        accuracy: mean(parts.map(p => p.accuracy)),
        total_queries: parts.reduce((a, p) => a + (p.total_queries ?? 0), 0),
        correct: parts.reduce((a, p) => a + (p.correct ?? 0), 0),
        ingestion_time_ms: parts.reduce((a, p) => a + (p.ingestion_time_ms ?? 0), 0),
        ingested_docs: parts.reduce((a, p) => a + (p.ingested_docs ?? 0), 0),
        avg_retrieve_time_ms: times.length ? mean(times) : null,
      }
    })
  return [...entries, ...averages]
}
