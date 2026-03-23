export const fmtTokens = n =>
  n >= 1_000_000 ? (n / 1_000_000).toFixed(1) + 'M' :
  n >= 1_000     ? (n / 1_000).toFixed(0) + 'k' : String(n)

export const avg   = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
export const pct50 = s   => s[Math.floor(s.length * 0.50)] ?? 0
export const pct99 = s   => s[Math.floor(s.length * 0.99)] ?? 0

export const accuracyColor = pct =>
  pct >= 0.7 ? '#34d399' : pct >= 0.4 ? '#fbbf24' : '#f87171'
