export async function fetchExternalResults() {
  try {
    const res = await fetch('/api/external-results')
    if (!res.ok) return {}
    return res.json()
  } catch { return {} }
}

export async function fetchManifest() {
  const res = await fetch('/api/results')
  if (!res.ok) throw new Error('Could not reach /api/results — run omb view to start the viewer.')
  return res.json()
}

export async function fetchDatasetInfo(dataset) {
  try {
    const res = await fetch(`/api/dataset/${encodeURIComponent(dataset)}/info`)
    if (!res.ok) return {}
    const d = await res.json()
    return d.error ? {} : d
  } catch { return {} }
}

export async function fetchSplitStats(dataset, split) {
  try {
    const res = await fetch(`/api/split-stats?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}`)
    if (!res.ok) return null
    const d = await res.json()
    return d.error ? null : d
  } catch { return null }
}

export async function fetchRunData(path) {
  const res = await fetch('/' + path)
  if (!res.ok) throw new Error(`Failed to load ${path}`)
  return res.json()
}

export async function fetchQueries(dataset, split, { search = '', category = '', limit = 50, offset = 0 } = {}) {
  const p = new URLSearchParams({ search, category, limit, offset })
  const res = await fetch(`/api/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}/queries?${p}`)
  if (!res.ok) throw new Error(`Failed to load queries`)
  return res.json()
}

export async function fetchDocuments(dataset, split, { search = '', category = '', limit = 20, offset = 0 } = {}) {
  const p = new URLSearchParams({ search, category, limit, offset })
  const res = await fetch(`/api/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}/documents?${p}`)
  if (!res.ok) throw new Error(`Failed to load documents`)
  return res.json()
}

export async function fetchSplitCategoryBreakdown(dataset, split) {
  try {
    const res = await fetch(`/api/split-category-breakdown?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}`)
    if (!res.ok) return []
    return res.json()
  } catch { return [] }
}

export async function fetchDocument(dataset, split, docId) {
  const res = await fetch(`/api/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}/documents/${encodeURIComponent(docId)}`)
  if (!res.ok) throw new Error(`Failed to load document ${docId}`)
  return res.json()
}
