<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchManifest, fetchSplitStats, fetchExternalResults, fetchDatasetInfo, fetchSplitCategoryBreakdown } from '../api.js'
import { fmtTokens, accuracyColor } from '../utils.js'
import Card from '@/components/ui/card.vue'
import Badge from '@/components/ui/badge.vue'
import BarChart from '@/components/ui/bar-chart.vue'
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
             class="text-xs text-primary hover:text-primary/80 transition-colors border border-primary/30 rounded px-2 py-0.5" @click.stop>
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
              <p class="text-muted-foreground/60 text-xs">{{ fmtSplitStats(splitStats[split]) }}</p>
            </div>
            <div class="flex items-center gap-4">
              <!-- View mode toggle — always show if split has any category data (lazy-detected) -->
              <div class="flex text-xs border border-border overflow-hidden rounded">
                <button @click="setViewMode(split, 'overall')"
                        :class="getViewMode(split) === 'overall' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'"
                        class="px-3 py-1.5 transition-colors">Overall</button>
                <button @click="setViewMode(split, 'category')"
                        :class="getViewMode(split) === 'category' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'"
                        class="px-3 py-1.5 transition-colors border-l border-border">By Category</button>
              </div>
              <button @click="router.push(`/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}`)"
                      class="text-primary hover:text-primary/80 text-xs transition-colors">Browse →</button>
            </div>
          </div>

          <!-- ── OVERALL VIEW ── -->
          <template v-if="getViewMode(split) === 'overall'">

            <!-- Charts -->
            <div v-if="local.length" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              <Card class="p-4">
                <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-4">Accuracy</p>
                <BarChart :rows="chartAccuracy(local).rows" :max="1"
                          :format="v => (v * 100).toFixed(1) + '%'"
                          variant="accuracy" />
              </Card>
              <Card class="p-4">
                <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-1">Recall speed</p>
                <p class="text-muted-foreground/40 text-xs mb-4">lower is better</p>
                <BarChart :rows="chartRecall(local).rows"
                          :format="v => Math.round(v) + 'ms'"
                          variant="recall" />
              </Card>
              <Card class="p-4">
                <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-1">Context tokens</p>
                <p class="text-muted-foreground/40 text-xs mb-4">lower is better</p>
                <BarChart :rows="chartTokens(local).rows"
                          :format="v => Math.round(v).toLocaleString()"
                          variant="tokens" />
              </Card>
            </div>

            <!-- Overall table -->
            <Card v-if="local.length" class="overflow-hidden mb-4">
              <UiTable>
                <TableHeader>
                  <TableRow>
                    <TableHead :sortable="true" @click="toggleLocalSort('memory')">Memory{{ sortIcon('memory', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" @click="toggleLocalSort('mode')">Mode{{ sortIcon('mode', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('total_queries')">Queries{{ sortIcon('total_queries', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('correct')">Correct{{ sortIcon('correct', localSortCol, localSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleLocalSort('accuracy')">Accuracy{{ sortIcon('accuracy', localSortCol, localSortDir) }}</TableHead>
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
                    <TableCell :right="true" class="font-semibold" :style="{ color: item.accuracy != null ? accuracyColor(item.accuracy) : '' }">
                      {{ item.accuracy != null ? (item.accuracy * 100).toFixed(1) + '%' : '—' }}
                    </TableCell>
                    <TableCell :right="true">{{ (item.ingestion_time_ms != null && item.ingested_docs) ? Math.round(item.ingestion_time_ms / item.ingested_docs) + 'ms' : '—' }}</TableCell>
                    <TableCell :right="true">{{ item.avg_retrieve_time_ms != null ? Math.round(item.avg_retrieve_time_ms) + 'ms' : '—' }}</TableCell>
                    <TableCell :right="true">{{ item.avg_context_tokens != null ? Math.round(item.avg_context_tokens).toLocaleString() : '—' }}</TableCell>
                    <TableCell :right="true" class="text-muted-foreground/40 text-xs">→</TableCell>
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
                <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-4">{{ section.label }}</p>

                <!-- Per-category bar charts -->
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                  <Card v-for="{ cat, rows } in section.charts" :key="cat" class="p-4">
                    <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-3">{{ cat }}</p>
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
                                   class="font-semibold tabular-nums"
                                   :style="{ color: row.cats[cat] != null ? accuracyColor(row.cats[cat]) : '' }">
                          {{ row.cats[cat] != null ? (row.cats[cat] * 100).toFixed(1) + '%' : '—' }}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </UiTable>
                </Card>
              </div>

            </template>
          </template>

          <!-- Unverified -->
          <template v-if="external.length">
            <div class="mb-3 mt-8">
              <p class="font-display text-xs font-semibold uppercase tracking-wider text-ca mb-1">Unverified</p>
              <p class="text-xs text-muted-foreground max-w-2xl leading-relaxed">Sourced from external papers or leaderboards — not independently reproduced here. Scores are <strong class="text-foreground">not directly comparable</strong>. Click a row to view the source.</p>
            </div>
            <Card class="overflow-hidden border-ca/15">
              <UiTable>
                <TableHeader>
                  <TableRow>
                    <TableHead :sortable="true" @click="toggleExtSort('memory')">Memory{{ sortIcon('memory', extSortCol, extSortDir) }}</TableHead>
                    <TableHead :sortable="true" :right="true" @click="toggleExtSort('accuracy')">Accuracy{{ sortIcon('accuracy', extSortCol, extSortDir) }}</TableHead>
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
                      <div v-if="item.comment" class="text-xs text-muted-foreground/60 mt-0.5">{{ item.comment }}</div>
                    </TableCell>
                    <TableCell :right="true" class="font-semibold" :style="{ color: item.accuracy != null ? accuracyColor(item.accuracy) : '' }">
                      {{ item.accuracy != null ? (item.accuracy * 100).toFixed(1) + '%' : '—' }}
                    </TableCell>
                    <TableCell class="text-ca text-xs">
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
