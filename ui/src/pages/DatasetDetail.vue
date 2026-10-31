<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useHead } from '@unhead/vue'
import { fetchManifest, fetchSplitStats, fetchExternalResults, fetchDatasetInfo, fetchSplitCategoryBreakdown } from '../api.js'
import { fmtTokens } from '../utils.js'
import Card from '@/components/ui/card.vue'
import Badge from '@/components/ui/badge.vue'
import BarChart from '@/components/ui/bar-chart.vue'
import CodingScatter from '@/components/ui/coding-scatter.vue'
import TopNav from '@/components/ui/top-nav.vue'
import UiTable from '@/components/ui/table.vue'
import TableHeader from '@/components/ui/table-header.vue'
import TableBody from '@/components/ui/table-body.vue'
import TableRow from '@/components/ui/table-row.vue'
import TableHead from '@/components/ui/table-head.vue'
import TableCell from '@/components/ui/table-cell.vue'

const route   = useRoute()
const router  = useRouter()
const dataset = computed(() => decodeURIComponent(route.params.name))

useHead(computed(() => ({
  title: `${dataset.value} — Agent Memory Benchmark`,
  meta: [
    { name: 'description', content: `Benchmark results for the ${dataset.value} dataset — accuracy, recall speed, and context token comparisons across memory providers.` },
    { property: 'og:title', content: `${dataset.value} — Agent Memory Benchmark` },
    { property: 'og:description', content: `Benchmark results for the ${dataset.value} dataset — accuracy, recall speed, and context token comparisons across memory providers.` },
    { property: 'og:url', content: `https://agentmemorybenchmark.ai/dataset/${encodeURIComponent(dataset.value)}` },
  ],
  link: [{ rel: 'canonical', href: `https://agentmemorybenchmark.ai/dataset/${encodeURIComponent(dataset.value)}` }],
})))

const manifest        = ref([])
const splitStats      = ref({})
const externalResults = ref({})
const datasetLinks    = ref([])
const catalog         = ref({ providers: {} })
const loading         = ref(true)
const error           = ref(null)
const activeSplit         = ref(null)
const splitViewMode       = ref({}) // { [split]: 'overall' | 'category' }
const splitCatBreakdown   = ref({}) // { [split]: breakdown[] } — per-result category breakdowns
const splitCatLoading     = ref({}) // { [split]: bool }

const providerByKey = computed(() => {
  const map = {}
  for (const [family, p] of Object.entries(catalog.value.providers ?? {})) {
    if (p.variants) {
      for (const [variant, v] of Object.entries(p.variants)) {
        map[v.key] = { family, variant, logo: p.logo, link: p.link }
      }
    } else {
      const entry = { family, variant: null, logo: p.logo, link: p.link }
      map[family] = entry
      for (const alias of (p.aliases ?? [])) map[alias] = entry
    }
  }
  return map
})

async function load() {
  loading.value = true; error.value = null
  try {
    const [all, extAll, info, cat] = await Promise.all([
      fetchManifest(),
      fetchExternalResults(),
      fetchDatasetInfo(dataset.value),
      fetch('/api/catalog').then(r => r.json()),
    ])
    manifest.value = all.filter(m => m.dataset === dataset.value)
    externalResults.value = extAll[dataset.value] ?? {}
    datasetLinks.value = info.links ?? []
    catalog.value = cat
    if (!manifest.value.length && !Object.keys(externalResults.value).length) {
      error.value = `No results found for dataset "${dataset.value}"`
      return
    }
    const canonicalSplits = cat?.datasets?.[dataset.value]?.splits ?? []
    const allSplitKeys = [...new Set([...manifest.value.map(m => m.split), ...Object.keys(extAll[dataset.value] ?? {})])]
    const splits = [...canonicalSplits.filter(s => allSplitKeys.includes(s)), ...allSplitKeys.filter(s => !canonicalSplits.includes(s))]
    if (!activeSplit.value || !splits.includes(activeSplit.value)) activeSplit.value = splits[0] ?? null
    splits.forEach(async split => {
      const s = await fetchSplitStats(dataset.value, split)
      if (s) splitStats.value = { ...splitStats.value, [split]: s }
    })
  } catch (e) { error.value = e.message }
  finally { loading.value = false }
}

onMounted(load)
watch(dataset, load)

const bySplit = computed(() => {
  const map = {}
  manifest.value.forEach(item => {
    const sp = item.split || '—'
    if (!map[sp]) map[sp] = { local: [], external: [] }
    map[sp].local.push(item)
  })
  Object.entries(externalResults.value).forEach(([sp, items]) => {
    if (!map[sp]) map[sp] = { local: [], external: [] }
    map[sp].external.push(...items)
  })
  const canonicalOrder = catalog.value?.datasets?.[dataset.value]?.splits ?? []
  const allSplits = [...canonicalOrder.filter(sp => map[sp]), ...Object.keys(map).filter(sp => !canonicalOrder.includes(sp))]
  return allSplits.map(sp => [sp, map[sp]])
})

// Retrieval-scored datasets carry precision/recall/active-pass counts; every other dataset
// leaves them null, and the columns stay hidden rather than showing a row of dashes.
const hasRetrievalMetrics = rows => (rows || []).some(r => r.mean_precision != null || r.mean_recall != null)

const localSortCol = ref('accuracy'); const localSortDir = ref('desc')
const extSortCol   = ref('accuracy'); const extSortDir   = ref('desc')

function _sortVal(item, col) {
  if (col === 'memory') return (providerByKey.value[item.memory]?.family ?? item.memory).toLowerCase()
  if (col === 'ingest') return (item.ingestion_time_ms != null && item.ingested_docs) ? item.ingestion_time_ms / item.ingested_docs : null
  return item[col] ?? null
}
function sortItems(items, col, dir) {
  return [...items].sort((a, b) => {
    const av = _sortVal(a, col), bv = _sortVal(b, col)
    if (av == null && bv == null) return 0
    if (av == null) return 1; if (bv == null) return -1
    if (typeof av === 'string') return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
    return dir === 'asc' ? av - bv : bv - av
  })
}
function toggleSort(col, sortCol, sortDir) {
  if (sortCol.value === col) sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  else { sortCol.value = col; sortDir.value = col === 'memory' || col === 'mode' ? 'asc' : 'desc' }
}

function _chartData(items, key, sortAsc = false) {
  const rows = items.filter(i => i[key] != null).map(item => {
    const info = providerByKey.value[item.memory]
    return { label: (info?.family ?? item.memory) + (info?.variant ? ' · ' + info.variant : ''), logo: info?.logo ?? null, value: item[key] }
  }).sort((a, b) => sortAsc ? a.value - b.value : b.value - a.value)
  const max = rows.reduce((m, r) => Math.max(m, r.value), 0) || 1
  return { rows, max }
}

function fmtSplitStats(s) {
  if (!s) return '…'
  const p = []
  if (s.queries != null) p.push(`${s.queries.toLocaleString()} queries`)
  if (s.docs != null) p.push(`${s.docs.toLocaleString()} docs`)
  if (s.total_tokens != null) p.push(`${fmtTokens(s.total_tokens)} tokens`)
  return p.join(' · ')
}

// ── Category view helpers ──────────────────────────────────────────────

// Single-category runs only (no comma-separated multi-category)
function singleCatRuns(local) {
  return local.filter(i => i.category && !i.category.includes(','))
}

function splitCategories(local) {
  return [...new Set(singleCatRuns(local).map(i => i.category))].sort()
}

// Pivot: rows = unique (run_name+memory+mode), cols = categories
function categoryMatrix(local) {
  const cats = splitCategories(local)
  if (!cats.length) return null
  const runs = singleCatRuns(local)
  const rowMap = {}
  runs.forEach(item => {
    const key = item.run_name + '||' + item.mode
    if (!rowMap[key]) rowMap[key] = { key, item, cats: {} }
    rowMap[key].cats[item.category] = item
  })
  const rows = Object.values(rowMap).sort((a, b) => {
    const vals = c => Object.values(c.cats).map(i => i.accuracy ?? 0)
    const avg = c => vals(c).reduce((s, v) => s + v, 0) / (vals(c).length || 1)
    return avg(b) - avg(a)
  })
  return { cats, rows }
}

// Per-category charts: one chart per category, all providers
function categoryCharts(local) {
  const cats = splitCategories(local)
  return cats.map(cat => {
    const items = singleCatRuns(local).filter(i => i.category === cat && i.accuracy != null)
    const rows = items.map(item => {
      const info = providerByKey.value[item.memory]
      return {
        label: (info?.family ?? item.memory) + (info?.variant ? ' · ' + info.variant : ''),
        logo: info?.logo ?? null,
        value: item.accuracy,
        item,
      }
    }).sort((a, b) => b.value - a.value)
    const max = rows.reduce((m, r) => Math.max(m, r.value), 0) || 1
    return { cat, rows, max }
  })
}

function providerLabel(item) {
  const info = providerByKey.value[item.memory]
  return (info?.family ?? item.memory) + (info?.variant ? ' · ' + info.variant : '')
}

const openExt = url => window.open(url, '_blank', 'noopener')
const sortLocal = items => sortItems(items, localSortCol.value, localSortDir.value)
const sortExt   = items => sortItems(items, extSortCol.value, extSortDir.value)
const toggleLocalSort = col => toggleSort(col, localSortCol, localSortDir)
const toggleExtSort   = col => toggleSort(col, extSortCol, extSortDir)
const chartAccuracy = local => _chartData(local, 'accuracy')
const chartRecall   = local => _chartData(local, 'avg_retrieve_time_ms', true)
const chartTokens   = local => _chartData(local, 'avg_context_tokens', true)
// coding datasets (sdebench) report agent metrics (interventions/cost/turns/tokens), not QA metrics
const isCoding = rows => rows.length > 0 && rows.every(r => r.coding)
// Memory arms: no memory, Hindsight, Hindsight with a decision model (runs record whether the
// server's decision model was called; older runs fall back to an "hsdm" run name).
const ARMS = [
  { key: 'vanilla',     label: 'No memory' },
  { key: 'hindsight',   label: 'Hindsight' },
  { key: 'hindsight-dm', label: 'Hindsight + decision model' },
]
const codingArm = item => {
  const m = (item.memory || '').toLowerCase(); const rn = (item.run_name || '').toLowerCase()
  if (m === 'none' || m === 'vanilla' || /(^|[-_])(none|vanilla|full)([-_]|$)/.test(rn)) return 'vanilla'
  if (m === 'hscoding' || m === 'hindsight-coding')
    return item.decision_model || /(^|[-_])hsdm([-_]|$)/.test(rn) ? 'hindsight-dm' : 'hindsight'
  const info = providerByKey.value[item.memory]
  return info?.family ?? item.memory ?? 'memory'
}
const armLabel = key => ARMS.find(a => a.key === key)?.label ?? key
const fmtUsd = v => v == null ? '—' : '$' + (v < 0.1 ? v.toFixed(3) : v.toFixed(2))
const fmtPct = (v, base) => (v == null || !base) ? '' : ((v - base) / base * 100 >= 0 ? '+' : '') + Math.round((v - base) / base * 100) + '%'
// Head-to-head groups: runs of one agent+model over the same task set, one row per memory arm,
// each metric per task and averaged over every run of that arm — so 3 repeats read as one line.
// Total cost = the agent's bill + the memory server's (unknown memory cost keeps total unknown).
const shortAgent = a => ({ 'claude-code': 'Claude Code', codex: 'Codex CLI', opencode: 'opencode' }[a] ?? a)
function codingHeadToHead(local) {
  const groups = {}
  for (const item of local) {
    if (!item.tasks) continue
    const key = `${item.agent}|${item.model}|${item.task_set ?? item.tasks}`
    const g = (groups[key] ??= { key, agent: item.agent, model: item.model, tasks: item.tasks, task_ids: item.task_ids, arms: {} })
    // one row per arm AND memory-server model: Hindsight on gemma and on gemini are different setups
    const models = [item.memory_llm, item.memory_decision].filter(Boolean).join(' + ')
    const arm = codingArm(item) + (models ? '|' + models : '')
    ;(g.arms[arm] ??= []).push(item)
  }
  const mean = (runs, f) => { const vs = runs.map(f); return vs.some(v => v == null) ? null : vs.reduce((a, b) => a + b, 0) / vs.length }
  return Object.values(groups).map(g => {
    const rows = Object.entries(g.arms).map(([armKey, runs]) => {
      const [arm, memoryLlm] = armKey.split('|')
      const per = k => r => r[k] == null ? null : r[k] / r.tasks
      const agentCost = mean(runs, per('cost_usd'))
      const memCost = arm === 'vanilla' ? 0 : mean(runs, per('memory_query_usd'))
      return {
        arm, memoryLlm, key: armKey, runs: [...runs].sort((a, b) => a.run_name.localeCompare(b.run_name)),
        solved: mean(runs, r => r.solved / r.tasks),
        interventions: mean(runs, per('interventions')),
        agent_cost: agentCost, memory_cost: memCost,
        ingest_cost: arm === 'vanilla' ? 0 : mean(runs, per('memory_ingest_usd')),
        total_cost: agentCost == null || memCost == null ? null : agentCost + memCost,
        wall_s: mean(runs, per('wall_s')),
      }
    }).sort((a, b) => (ARMS.findIndex(x => x.key === a.arm) - ARMS.findIndex(x => x.key === b.arm))
                      || (a.memoryLlm ?? '').localeCompare(b.memoryLlm ?? ''))
    const base = rows.find(r => r.arm === 'vanilla')
    return { ...g, rows, base, comparable: !!base && rows.length > 1 }
  }).sort((a, b) => (b.comparable - a.comparable) || (b.tasks - a.tasks) || a.key.localeCompare(b.key))
}
const openArms = ref({})
const toggleArm = k => { openArms.value = { ...openArms.value, [k]: !openArms.value[k] } }
// Scatter: one group per agent+model+task set that has a no-memory baseline; x = total cost.
const CODING_METRIC_KEYS = ['interventions', 'agent_cost', 'total_cost', 'wall_s']
function codingGroups(local) {
  return codingHeadToHead(local).filter(g => g.comparable).map(g => ({
    label: `${g.agent} · ${g.model ?? '?'} · ${g.tasks}`, agent: g.agent, model: g.model, tasks: g.tasks,
    arms: g.rows.map(r => ({ arm: r.arm, key: r.key, llm: r.memoryLlm, runs: r.runs.length,
                             metrics: Object.fromEntries(CODING_METRIC_KEYS.map(k => [k, r[k]])) })),
  }))
}
const sortIcon = (col, active, dir) => active === col ? (dir === 'asc' ? ' ↑' : ' ↓') : ''
const getViewMode = split => splitViewMode.value[split] ?? 'overall'
async function setViewMode(split, mode) {
  splitViewMode.value = { ...splitViewMode.value, [split]: mode }
  if (mode === 'category' && !splitCatBreakdown.value[split]) {
    splitCatLoading.value = { ...splitCatLoading.value, [split]: true }
    splitCatBreakdown.value = { ...splitCatBreakdown.value, [split]: await fetchSplitCategoryBreakdown(dataset.value, split) }
    splitCatLoading.value = { ...splitCatLoading.value, [split]: false }
  }
}

// Merge file-level categories (ama-bench style) + per-result breakdowns (locomo style)
function allCategoryRows(local, split) {
  // File-level: each single-category run is its own data point
  const fileRows = singleCatRuns(local).map(item => ({
    run_name: item.run_name, memory: item.memory, mode: item.mode, path: item.path,
    axis: 'category',
    categories: { [item.category]: item.accuracy },
  }))
  // Per-result: from server breakdown
  const breakdown = splitCatBreakdown.value[split] ?? []
  return [...fileRows, ...breakdown]
}

// Map category value → type ('doc' | 'query' | null) from split stats
function catTypeMap(split) {
  return Object.fromEntries(
    Object.entries(splitStats.value[split]?.categories ?? {}).map(([k, v]) => [k, v.type ?? null])
  )
}

// Build provider rows (grouped by run_name+mode) with cats map
function buildProviderRows(allRows) {
  const rowMap = {}
  allRows.forEach(r => {
    const key = r.run_name + '||' + r.mode
    if (!rowMap[key]) rowMap[key] = { key, run_name: r.run_name, memory: r.memory, mode: r.mode, path: r.path, cats: {} }
    Object.assign(rowMap[key].cats, r.categories)
  })
  return Object.values(rowMap).sort((a, b) => {
    const avg = r => { const vs = Object.values(r.cats).filter(v => v != null); return vs.length ? vs.reduce((s, v) => s + v, 0) / vs.length : 0 }
    return avg(b) - avg(a)
  })
}

// Return sections: [{ type, label, cats, rows, charts }]
function categorySections(local, split) {
  const allRows = allCategoryRows(local, split)
  if (!allRows.length) return null

  const typeMap = catTypeMap(split)
  const catSet = new Set()
  allRows.forEach(r => Object.keys(r.categories).forEach(c => catSet.add(c)))
  const allCats = [...catSet].sort()

  // Group cats by type
  const byType = {}
  allCats.forEach(cat => {
    const t = typeMap[cat] ?? 'other'
    ;(byType[t] ??= []).push(cat)
  })

  const typeOrder = ['query', 'doc', 'other']
  const typeLabel = { query: 'Query categories', doc: 'Document categories', other: 'Categories' }

  const providerRows = buildProviderRows(allRows)

  return typeOrder.filter(t => byType[t]?.length).map(t => {
    const cats = byType[t]
    const charts = cats.map(cat => {
      const entries = allRows.filter(r => r.categories[cat] != null).map(r => {
        const info = providerByKey.value[r.memory]
        return { label: (info?.family ?? r.memory) + (info?.variant ? ' · ' + info.variant : ''), logo: info?.logo ?? null, value: r.categories[cat] }
      }).sort((a, b) => b.value - a.value)
      const max = entries.reduce((m, r) => Math.max(m, r.value), 0) || 1
      return { cat, rows: entries, max }
    })
    return { type: t, label: typeLabel[t], cats, rows: providerRows, charts }
  })
}

function hasCategoryData(local, split) {
  return splitCategories(local).length > 0 || (splitCatBreakdown.value[split]?.length > 0)
}
</script>

<template>
  <div class="min-h-screen bg-background">

    <TopNav :crumbs="[{ label: dataset }]" />

    <div class="px-6 py-8">

      <div class="mb-10">
        <div class="flex items-center gap-3 flex-wrap mb-2">
          <h1 class="font-display text-2xl font-bold text-foreground tracking-tight">{{ dataset }}</h1>
          <Badge v-if="catalog.datasets?.[dataset]?.task"
                 :variant="catalog.datasets[dataset].task === 'MCQ' ? 'mcq' : 'open'">
            {{ catalog.datasets[dataset].task === 'MCQ' ? 'Multiple Choice' : 'Open-ended' }}
          </Badge>
          <a v-for="link in datasetLinks" :key="link.label" :href="link.url" target="_blank" rel="noopener"
             class="text-sm text-primary hover:text-primary/80 transition-colors border border-primary/30 rounded px-2 py-0.5" @click.stop>
            {{ link.label }} ↗
          </a>
        </div>
        <p v-if="catalog.datasets?.[dataset]?.description"
           class="text-muted-foreground text-sm max-w-2xl leading-relaxed">
          {{ catalog.datasets[dataset].description }}
        </p>
      </div>

      <div v-if="loading" class="text-center mt-24"><p class="text-muted-foreground text-sm animate-pulse">Loading…</p></div>
      <div v-else-if="error" class="text-destructive text-sm mt-10">{{ error }}</div>

      <div v-else>
        <!-- Split tabs -->
        <div class="flex gap-1 mb-8 border-b border-border">
          <button v-for="[split] in bySplit" :key="split"
                  @click="activeSplit = split"
                  :class="activeSplit === split
                    ? 'border-b-2 border-primary text-foreground font-semibold'
                    : 'text-muted-foreground hover:text-foreground'"
                  class="px-4 py-2 text-sm transition-colors -mb-px">
            {{ split }}
          </button>
        </div>

        <div v-for="[split, { local, external }] in bySplit" :key="split" v-show="activeSplit === split">

          <!-- Split header -->
          <div class="flex items-center justify-between mb-6">
            <div>
              <p class="text-muted-foreground/85 text-sm">{{ fmtSplitStats(splitStats[split]) }}</p>
            </div>
            <div class="flex items-center gap-4">
              <!-- View mode toggle — always show if split has any category data (lazy-detected) -->
              <div class="flex text-sm border border-border overflow-hidden rounded">
                <button @click="setViewMode(split, 'overall')"
                        :class="getViewMode(split) === 'overall' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'"
                        class="px-3 py-1.5 transition-colors">Overall</button>
                <button @click="setViewMode(split, 'category')"
                        :class="getViewMode(split) === 'category' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'"
                        class="px-3 py-1.5 transition-colors border-l border-border">By Category</button>
              </div>
              <button @click="router.push(`/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}`)"
                      class="text-primary hover:text-primary/80 text-sm transition-colors">Browse →</button>
            </div>
          </div>

          <!-- ── OVERALL VIEW ── -->
          <template v-if="getViewMode(split) === 'overall'">

            <!-- Chart (coding datasets): corrections vs cost/time, vanilla → hindsight per agent -->
            <Card v-if="isCoding(local) && codingGroups(local).length" class="p-4 mb-4">
              <CodingScatter :groups="codingGroups(local)" />
            </Card>

            <!-- ── CODING dataset (sdebench): head-to-head per agent · model · task set ── -->
            <template v-if="isCoding(local)">
              <Card v-for="g in codingHeadToHead(local)" :key="g.key" class="overflow-hidden mb-4">
                <div class="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-3 pb-2">
                  <div class="flex items-baseline gap-2">
                    <span class="font-semibold">{{ shortAgent(g.agent) }}</span>
                    <span class="text-muted-foreground/85">{{ g.model ?? '?' }}</span>
                    <Badge variant="secondary">{{ g.tasks }} task{{ g.tasks === 1 ? '' : 's' }}</Badge>
                  </div>
                  <span v-if="g.task_ids" class="text-muted-foreground/70 text-xs font-mono truncate">{{ g.task_ids.map(t => t.replace(/^boltons-|-001$/g, '')).join(' · ') }}</span>
                </div>
                <UiTable>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Memory</TableHead>
                      <TableHead :right="true">Runs</TableHead>
                      <TableHead :right="true">Solved</TableHead>
                      <TableHead :right="true" title="Human-like feedback rounds needed per task (lower is better)">Corrections / task</TableHead>
                      <TableHead :right="true" title="The coding agent's own bill per task">Agent $ / task</TableHead>
                      <TableHead :right="true" title="The memory server's LLM + decision-model bill for the agent's reflects/recalls">Memory $ / task</TableHead>
                      <TableHead :right="true" title="Agent + memory during the task">Total $ / task</TableHead>
                      <TableHead :right="true" title="Seeding the bank from the repo's history and chats: paid once per repo, not per task">One-time load $</TableHead>
                      <TableHead :right="true">Wall / task</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <template v-for="r in g.rows" :key="r.key">
                      <TableRow :clickable="true" @click="toggleArm(g.key + r.key)">
                        <TableCell :primary="true">
                          <span class="inline-block w-3 text-muted-foreground/70">{{ openArms[g.key + r.key] ? '▾' : '▸' }}</span>
                          <Badge :variant="r.arm === 'vanilla' ? 'secondary' : 'default'">{{ armLabel(r.arm) }}</Badge>
                          <span v-if="r.memoryLlm" class="ml-2 text-muted-foreground/85 text-xs font-mono">{{ r.memoryLlm }}</span>
                        </TableCell>
                        <TableCell :right="true">{{ r.runs.length }}</TableCell>
                        <TableCell :right="true">{{ r.solved != null ? Math.round(r.solved * 100) + '%' : '—' }}</TableCell>
                        <TableCell :right="true" class="font-semibold">
                          {{ r.interventions != null ? r.interventions.toFixed(2) : '—' }}
                          <span v-if="r !== g.base" class="delta">{{ fmtPct(r.interventions, g.base?.interventions) }}</span>
                        </TableCell>
                        <TableCell :right="true">{{ fmtUsd(r.agent_cost) }}</TableCell>
                        <TableCell :right="true" class="text-muted-foreground/85">{{ r.arm === 'vanilla' ? '—' : fmtUsd(r.memory_cost) }}</TableCell>
                        <TableCell :right="true" class="font-semibold">
                          {{ fmtUsd(r.total_cost) }}
                          <span v-if="r !== g.base" class="delta">{{ fmtPct(r.total_cost, g.base?.total_cost) }}</span>
                        </TableCell>
                        <TableCell :right="true" class="text-muted-foreground/85">{{ r.arm === 'vanilla' ? '—' : fmtUsd(r.ingest_cost) }}</TableCell>
                        <TableCell :right="true">{{ r.wall_s != null ? Math.round(r.wall_s) + 's' : '—' }}</TableCell>
                      </TableRow>
                      <TableRow v-for="item in (openArms[g.key + r.key] ? r.runs : [])" :key="item.path" :clickable="true"
                                class="text-sm" @click="router.push('/run/' + item.path.replace(/\.json(\.gz)?$/, ''))">
                        <TableCell class="pl-10 text-muted-foreground/85">{{ item.run_name }} →</TableCell>
                        <TableCell></TableCell>
                        <TableCell :right="true">{{ item.solved }}/{{ item.tasks }}</TableCell>
                        <TableCell :right="true">{{ item.interventions }}</TableCell>
                        <TableCell :right="true">{{ fmtUsd(item.cost_usd) }}</TableCell>
                        <TableCell :right="true" class="text-muted-foreground/85">{{ r.arm === 'vanilla' ? '—' : fmtUsd(item.memory_query_usd) }}</TableCell>
                        <TableCell :right="true">{{ item.memory_query_usd != null || r.arm === 'vanilla' ? fmtUsd(item.cost_usd + (item.memory_query_usd ?? 0)) : '—' }}</TableCell>
                        <TableCell :right="true" class="text-muted-foreground/85">{{ r.arm === 'vanilla' ? '—' : fmtUsd(item.memory_ingest_usd) }}</TableCell>
                        <TableCell :right="true">{{ item.wall_s != null ? Math.round(item.wall_s) + 's' : '—' }}</TableCell>
                      </TableRow>
                    </template>
                  </TableBody>
                </UiTable>
              </Card>
              <p class="text-muted-foreground/70 text-xs mb-4">Per task, mean over runs. Memory $: what the Hindsight server's own models (LLM + decision model) spent answering the agent; One-time load $: seeding the bank from the repo's history (each task has its own bank here, a real repo pays it once). "—" when a run predates that measurement. Percentages are vs. no memory.</p>
            </template>

            <!-- Charts (QA datasets only) -->
            <div v-if="local.length && !isCoding(local)" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              <Card class="p-4">
                <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-4">Accuracy</p>
                <BarChart :rows="chartAccuracy(local).rows" :max="1"
                          :format="v => (v * 100).toFixed(1) + '%'"
                          variant="accuracy" />
              </Card>
              <Card class="p-4">
                <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-1">Recall speed</p>
                <p class="text-muted-foreground/70 text-sm mb-4">lower is better</p>
                <BarChart :rows="chartRecall(local).rows"
                          :format="v => Math.round(v) + 'ms'"
                          variant="recall" :lower-better="true" />
              </Card>
              <Card class="p-4">
                <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-1">Context tokens</p>
                <p class="text-muted-foreground/70 text-sm mb-4">lower is better</p>
                <BarChart :rows="chartTokens(local).rows"
                          :format="v => v >= 1000 ? (v / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : Math.round(v)"
                          variant="tokens" :lower-better="true" />
              </Card>
            </div>

            <!-- Overall table (QA datasets only) -->
            <Card v-if="local.length && !isCoding(local)" class="overflow-hidden mb-4">
              <UiTable>
                <TableHeader>
                  <TableRow>
                    <TableHead :sortable="true" @click="toggleLocalSort('memory')">Memory{{ sortIcon('memory', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" @click="toggleLocalSort('mode')">Mode{{ sortIcon('mode', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('total_queries')">Queries{{ sortIcon('total_queries', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('correct')">Correct{{ sortIcon('correct', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('accuracy')">Accuracy{{ sortIcon('accuracy', localSortCol, localSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(local)" :sortable="true" :right="true" @click="toggleLocalSort('mean_precision')">Precision{{ sortIcon('mean_precision', localSortCol, localSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(local)" :sortable="true" :right="true" @click="toggleLocalSort('mean_recall')">Recall{{ sortIcon('mean_recall', localSortCol, localSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(local)" :sortable="true" :right="true" @click="toggleLocalSort('active_passes')">Active{{ sortIcon('active_passes', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('ingest')">Ingest/doc{{ sortIcon('ingest', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('avg_retrieve_time_ms')">Recall avg{{ sortIcon('avg_retrieve_time_ms', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('avg_context_tokens')">Ctx tokens{{ sortIcon('avg_context_tokens', localSortCol, localSortDir) }}</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow v-for="item in sortLocal(local)" :key="item.path" :clickable="true"
                            @click="router.push('/run/' + item.path.replace(/\.json(\.gz)?$/, ''))">
                    <TableCell :primary="true">
                      <div class="flex items-center gap-2">
                        <img v-if="providerByKey[item.memory]?.logo" :src="providerByKey[item.memory].logo"
                             class="w-4 h-4 rounded object-contain shrink-0" @error="$event.target.style.display='none'" />
                        <span>{{ providerByKey[item.memory]?.family ?? item.memory }}</span>
                        <Badge v-if="providerByKey[item.memory]?.variant" variant="secondary">{{ providerByKey[item.memory].variant }}</Badge>
                        <Badge v-if="item.run_name && item.run_name !== item.memory" variant="default">{{ item.run_name }}</Badge>
                      </div>
                    </TableCell>
                    <TableCell>{{ item.mode }}</TableCell>
                    <TableCell :right="true">{{ item.total_queries ?? '—' }}</TableCell>
                    <TableCell :right="true">{{ item.correct ?? '—' }}</TableCell>
                    <TableCell :right="true" class="font-semibold">
                      {{ item.accuracy != null ? (item.accuracy * 100).toFixed(1) + '%' : '—' }}
                    </TableCell>
                    <TableCell v-if="hasRetrievalMetrics(local)" :right="true">{{ item.mean_precision != null ? item.mean_precision.toFixed(2) : '—' }}</TableCell>
                    <TableCell v-if="hasRetrievalMetrics(local)" :right="true">{{ item.mean_recall != null ? item.mean_recall.toFixed(2) : '—' }}</TableCell>
                    <TableCell v-if="hasRetrievalMetrics(local)" :right="true">{{ item.active_passes != null && item.active_total ? item.active_passes + '/' + item.active_total : '—' }}</TableCell>
                    <TableCell :right="true">{{ (item.ingestion_time_ms != null && item.ingested_docs) ? Math.round(item.ingestion_time_ms / item.ingested_docs) + 'ms' : '—' }}</TableCell>
                    <TableCell :right="true">{{ item.avg_retrieve_time_ms != null ? Math.round(item.avg_retrieve_time_ms) + 'ms' : '—' }}</TableCell>
                    <TableCell :right="true">{{ item.avg_context_tokens != null ? Math.round(item.avg_context_tokens).toLocaleString() : '—' }}</TableCell>
                    <TableCell :right="true" class="text-muted-foreground/70 text-sm">→</TableCell>
                  </TableRow>
                </TableBody>
              </UiTable>
            </Card>

          </template>

          <!-- ── CATEGORY VIEW ── -->
          <template v-else>
            <div v-if="splitCatLoading[split]" class="text-muted-foreground text-sm animate-pulse py-8 text-center">Loading categories…</div>
            <div v-else-if="!categorySections(local, split)" class="text-muted-foreground text-sm py-8 text-center">No category data available for this split.</div>
            <template v-else>

              <div v-for="section in categorySections(local, split)" :key="section.type" class="mb-10">
                <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-4">{{ section.label }}</p>

                <!-- Per-category bar charts -->
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                  <Card v-for="{ cat, rows } in section.charts" :key="cat" class="p-4">
                    <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-3">{{ cat }}</p>
                    <BarChart :rows="rows" :max="1"
                              :format="v => (v * 100).toFixed(1) + '%'"
                              variant="accuracy" />
                  </Card>
                </div>

                <!-- Pivot table -->
                <Card class="overflow-hidden">
                  <UiTable>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Provider</TableHead>
                        <TableHead>Mode</TableHead>
                        <TableHead v-for="cat in section.cats" :key="cat" :right="true">{{ cat }}</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow v-for="row in section.rows" :key="row.key">
                        <TableCell :primary="true">
                          <div class="flex items-center gap-2">
                            <img v-if="providerByKey[row.memory]?.logo" :src="providerByKey[row.memory].logo"
                                 class="w-4 h-4 rounded object-contain shrink-0" @error="$event.target.style.display='none'" />
                            <span>{{ providerByKey[row.memory]?.family ?? row.memory }}</span>
                            <Badge v-if="providerByKey[row.memory]?.variant" variant="secondary">{{ providerByKey[row.memory].variant }}</Badge>
                            <Badge v-if="row.run_name && row.run_name !== row.memory" variant="default">{{ row.run_name }}</Badge>
                          </div>
                        </TableCell>
                        <TableCell>{{ row.mode }}</TableCell>
                        <TableCell v-for="cat in section.cats" :key="cat" :right="true"
                                   class="font-semibold tabular-nums">
                          {{ row.cats[cat] != null ? (row.cats[cat] * 100).toFixed(1) + '%' : '—' }}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </UiTable>
                </Card>
              </div>

            </template>
          </template>

          <!-- Scoring note -->
          <div v-if="catalog.datasets?.[dataset]?.scoring_note" class="mt-6 mb-2 rounded-md bg-muted/50 px-4 py-3 max-w-3xl">
            <p class="text-sm font-medium text-foreground/80 mb-1">How is this scored?</p>
            <p class="text-sm text-muted-foreground leading-relaxed">{{ catalog.datasets[dataset].scoring_note }}</p>
          </div>

          <!-- Unverified -->
          <template v-if="external.length">
            <div class="mb-3 mt-8">
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-ca mb-1">Unverified</p>
              <p class="text-sm text-muted-foreground max-w-2xl leading-relaxed">Sourced from external papers or leaderboards — not independently reproduced here. Scores are <strong class="text-foreground">not directly comparable</strong>. Click a row to view the source.</p>
            </div>
            <Card class="overflow-hidden border-ca/15">
              <UiTable>
                <TableHeader>
                  <TableRow>
                    <TableHead :sortable="true" @click="toggleExtSort('memory')">Memory{{ sortIcon('memory', extSortCol, extSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleExtSort('accuracy')">Accuracy{{ sortIcon('accuracy', extSortCol, extSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(external)" :sortable="true" :right="true" @click="toggleExtSort('mean_precision')">Precision{{ sortIcon('mean_precision', extSortCol, extSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(external)" :sortable="true" :right="true" @click="toggleExtSort('mean_recall')">Recall{{ sortIcon('mean_recall', extSortCol, extSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(external)" :sortable="true" :right="true" @click="toggleExtSort('active_passes')">Active{{ sortIcon('active_passes', extSortCol, extSortDir) }}</TableHead>
                    <TableHead v-if="hasRetrievalMetrics(external)" :sortable="true" :right="true" @click="toggleExtSort('avg_retrieve_time_ms')">Recall avg{{ sortIcon('avg_retrieve_time_ms', extSortCol, extSortDir) }}</TableHead>
                    <TableHead>Source</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow v-for="item in sortExt(external)" :key="item.memory + (item.source_label ?? '')"
                            :clickable="true" @click="openExt(item.source_url)">
                    <TableCell :primary="true">
                      <div class="flex items-center gap-2">
                        <img v-if="providerByKey[item.memory]?.logo" :src="providerByKey[item.memory].logo"
                             class="w-4 h-4 rounded object-contain shrink-0" @error="$event.target.style.display='none'" />
                        <span>{{ providerByKey[item.memory]?.family ?? item.memory }}</span>
                        <Badge v-if="providerByKey[item.memory]?.variant" variant="secondary">{{ providerByKey[item.memory].variant }}</Badge>
                      </div>
                      <div v-if="item.comment" class="text-sm text-muted-foreground/85 mt-0.5">{{ item.comment }}</div>
                    </TableCell>
                    <TableCell :right="true" class="font-semibold">
                      {{ item.accuracy != null ? (item.accuracy * 100).toFixed(1) + '%' : '—' }}
                    </TableCell>
                    <TableCell v-if="hasRetrievalMetrics(external)" :right="true">{{ item.mean_precision != null ? item.mean_precision.toFixed(2) : '—' }}</TableCell>
                    <TableCell v-if="hasRetrievalMetrics(external)" :right="true">{{ item.mean_recall != null ? item.mean_recall.toFixed(2) : '—' }}</TableCell>
                    <TableCell v-if="hasRetrievalMetrics(external)" :right="true">{{ item.active_passes != null && item.active_total ? item.active_passes + '/' + item.active_total : '—' }}</TableCell>
                    <TableCell v-if="hasRetrievalMetrics(external)" :right="true">{{ item.avg_retrieve_time_ms != null ? Math.round(item.avg_retrieve_time_ms) + 'ms' : '—' }}</TableCell>
                    <TableCell class="text-ca text-sm">
                      <span v-if="item.source_label">{{ item.source_label }} ↗</span>
                      <span v-else class="text-muted-foreground">View source ↗</span>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </UiTable>
            </Card>
          </template>

        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.delta { margin-left: 0.35rem; font-size: 0.75rem; font-weight: 500; color: var(--muted-foreground); }
</style>
