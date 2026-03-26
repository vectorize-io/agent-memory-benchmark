<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useHead } from '@unhead/vue'
import { fetchManifest, fetchExternalResults } from '../api.js'

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

useHead({
  title: 'Agent Memory Benchmark — AMB',
  meta: [
    { name: 'description', content: 'An open, reproducible leaderboard for evaluating AI agent memory and retrieval systems on real-world long-context tasks.' },
    { property: 'og:title', content: 'Agent Memory Benchmark — AMB' },
    { property: 'og:description', content: 'An open, reproducible leaderboard for evaluating AI agent memory and retrieval systems on real-world long-context tasks.' },
    { property: 'og:url', content: 'https://agentmemorybenchmark.ai/' },
    { name: 'twitter:title', content: 'Agent Memory Benchmark — AMB' },
    { name: 'twitter:description', content: 'An open, reproducible leaderboard for evaluating AI agent memory and retrieval systems on real-world long-context tasks.' },
  ],
  link: [{ rel: 'canonical', href: 'https://agentmemorybenchmark.ai/' }],
})

const ABOUT_OPEN_KEY = 'omb_about_open'

const router   = useRouter()
const manifest         = ref([])
const catalog          = ref({ datasets: {}, providers: {}, modes: {} })
const externalResults  = ref({})
const loading  = ref(true)
const error    = ref(null)
let pollTimer  = null

const aboutOpen = ref(localStorage.getItem(ABOUT_OPEN_KEY) !== 'false')

async function load() {
  try {
    const [manifestData, catalogData, extData] = await Promise.all([
      fetchManifest(),
      fetch('/api/catalog').then(r => r.json()),
      fetchExternalResults(),
    ])
    manifest.value        = manifestData
    catalog.value         = catalogData
    externalResults.value = extData
    error.value = null
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(() => { load(); pollTimer = setInterval(load, 5000) })
onUnmounted(() => clearInterval(pollTimer))

const providerByKey = computed(() => {
  const map = {}
  for (const [family, p] of Object.entries(catalog.value.providers ?? {})) {
    if (p.variants) {
      for (const [variant, v] of Object.entries(p.variants)) {
        map[v.key] = { family, variant, logo: p.logo }
      }
    } else {
      const entry = { family, variant: null, logo: p.logo }
      map[family] = entry
      for (const alias of (p.aliases ?? [])) map[alias] = entry
    }
  }
  return map
})

function computeChart(items) {
  const best = {}
  items.forEach(item => {
    if (item.accuracy == null) return
    const info = providerByKey.value[item.memory]
    const label = info?.family ?? item.memory
    if (!best[label] || item.accuracy > best[label].value) {
      best[label] = { label, logo: info?.logo ?? null, value: item.accuracy }
    }
  })
  const rows = Object.values(best).sort((a, b) => b.value - a.value).slice(0, 6)
  return { rows }
}

const datasets = computed(() => {
  const map = {}
  manifest.value.forEach(item => {
    const ds = item.dataset || '(unknown)'
    if (!map[ds]) map[ds] = { name: ds, splits: new Set(), runs: 0, items: [] }
    map[ds].splits.add(item.split)
    map[ds].runs++
    map[ds].items.push(item)
  })
  // Include datasets that only have external results
  Object.entries(externalResults.value).forEach(([ds, splitMap]) => {
    if (!map[ds]) map[ds] = { name: ds, splits: new Set(), runs: 0, items: [] }
    Object.keys(splitMap).forEach(split => map[ds].splits.add(split))
  })
  return Object.values(map).map(ds => ({ ...ds, chart: computeChart(ds.items) }))
})

// ── Cross-provider comparison table ──────────────────────────────────
const showUnverified = ref(false)
const cmpSortCol = ref(null) // null = default (avg accuracy), or "dataset||split" colKey
const cmpSortDir = ref('desc')

function toggleCmpSort(colKey) {
  if (cmpSortCol.value === colKey) cmpSortDir.value = cmpSortDir.value === 'asc' ? 'desc' : 'asc'
  else { cmpSortCol.value = colKey; cmpSortDir.value = 'desc' }
}
function cmpSortIcon(colKey) {
  return cmpSortCol.value === colKey ? (cmpSortDir.value === 'asc' ? ' ↑' : ' ↓') : ''
}

const comparisonTable = computed(() => {
  const colSet = new Map()
  const catalogDatasets = catalog.value.datasets ?? {}

  manifest.value.forEach(item => {
    const key = item.dataset + '||' + item.split
    if (!colSet.has(key)) colSet.set(key, { dataset: item.dataset, split: item.split })
  })
  if (showUnverified.value) {
    Object.entries(externalResults.value).forEach(([ds, splitMap]) => {
      Object.keys(splitMap).forEach(split => {
        const key = ds + '||' + split
        if (!colSet.has(key)) colSet.set(key, { dataset: ds, split })
      })
    })
  }

  const columns = [...colSet.values()]
  columns.sort((a, b) => {
    const aOrder = Object.keys(catalogDatasets).indexOf(a.dataset)
    const bOrder = Object.keys(catalogDatasets).indexOf(b.dataset)
    const ai = aOrder >= 0 ? aOrder : 999
    const bi = bOrder >= 0 ? bOrder : 999
    if (ai !== bi) return ai - bi
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset)
    const splits = catalogDatasets[a.dataset]?.splits ?? []
    const asi = splits.indexOf(a.split), bsi = splits.indexOf(b.split)
    if (asi >= 0 && bsi >= 0) return asi - bsi
    if (asi >= 0) return -1
    if (bsi >= 0) return 1
    return a.split.localeCompare(b.split)
  })

  const providerMap = {}

  const getProviderInfo = (memory) => {
    const info = providerByKey.value[memory]
    const label = (info?.family ?? memory) + (info?.variant ? ' · ' + info.variant : '')
    return { label, logo: info?.logo ?? null }
  }

  manifest.value.forEach(item => {
    if (item.accuracy == null) return
    const { label, logo } = getProviderInfo(item.memory)
    if (!providerMap[label]) providerMap[label] = { label, logo, cells: {} }
    const colKey = item.dataset + '||' + item.split
    const existing = providerMap[label].cells[colKey]
    if (!existing || item.accuracy > existing.accuracy) {
      providerMap[label].cells[colKey] = { accuracy: item.accuracy, unverified: false }
    }
  })

  if (showUnverified.value) {
    Object.entries(externalResults.value).forEach(([ds, splitMap]) => {
      Object.entries(splitMap).forEach(([split, items]) => {
        const colKey = ds + '||' + split
        items.forEach(item => {
          if (item.accuracy == null) return
          const { label, logo } = getProviderInfo(item.memory)
          if (!providerMap[label]) providerMap[label] = { label, logo, cells: {} }
          const existing = providerMap[label].cells[colKey]
          if (!existing || (existing.unverified && item.accuracy > existing.accuracy)) {
            providerMap[label].cells[colKey] = {
              accuracy: item.accuracy, unverified: true,
              source_url: item.source_url, source_label: item.source_label, comment: item.comment,
            }
          }
        })
      })
    })
  }

  // Sort providers
  const col = cmpSortCol.value
  const dir = cmpSortDir.value
  const providers = Object.values(providerMap).sort((a, b) => {
    let av, bv
    if (col === null) {
      // Default: average accuracy
      const avgAcc = p => {
        const vals = Object.values(p.cells).map(c => c.accuracy)
        return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
      }
      av = avgAcc(a); bv = avgAcc(b)
    } else if (col === 'provider') {
      av = a.label.toLowerCase(); bv = b.label.toLowerCase()
      if (av === bv) return 0
      const cmp = av < bv ? -1 : 1
      return dir === 'asc' ? cmp : -cmp
    } else {
      av = a.cells[col]?.accuracy ?? null
      bv = b.cells[col]?.accuracy ?? null
    }
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return dir === 'asc' ? av - bv : bv - av
  })

  // Best per column (only among verified if unverified shown)
  const bestPerCol = {}
  columns.forEach(c => {
    const colKey = c.dataset + '||' + c.split
    let best = -1
    providers.forEach(p => {
      const cell = p.cells[colKey]
      if (cell && cell.accuracy > best) best = cell.accuracy
    })
    if (best >= 0) bestPerCol[colKey] = best
  })

  const datasetGroups = []
  let current = null
  columns.forEach(c => {
    if (!current || current.dataset !== c.dataset) {
      current = { dataset: c.dataset, splits: [] }
      datasetGroups.push(current)
    }
    current.splits.push(c.split)
  })

  const hasExternal = Object.keys(externalResults.value).length > 0

  return { columns, providers, bestPerCol, datasetGroups, hasExternal }
})

function toggleAbout() {
  aboutOpen.value = !aboutOpen.value
  localStorage.setItem(ABOUT_OPEN_KEY, aboutOpen.value)
}

// ── Scroll-reveal directive ───────────────────────────────────────────────
const vReveal = {
  mounted(el, binding) {
    const delay = binding.value ?? 0
    el.style.transitionDelay = delay + 'ms'
    el.classList.add('reveal-item')
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        el.classList.add('reveal-visible')
        obs.disconnect()
      }
    }, { threshold: 0.05 })
    obs.observe(el)
    el._revealObs = obs
  },
  unmounted(el) { el._revealObs?.disconnect() },
}
</script>

<template>
  <div class="min-h-screen bg-background">

    <TopNav />

    <div class="px-6 py-12 space-y-20">

      <!-- Hero -->
      <section class="pt-4">
        <div class="max-w-3xl">
          <h1 class="font-display text-3xl font-bold text-foreground tracking-tight mb-4">
            Agent Memory Benchmark
          </h1>
          <p class="text-base text-muted-foreground leading-relaxed mb-8">
            An open, reproducible benchmark for evaluating memory and retrieval systems on real-world long-context tasks —
            personal conversations, agent trajectories, and time-sensitive knowledge.
            <a href="https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark" target="_blank" rel="noopener"
               class="text-primary hover:text-primary/80 transition-colors whitespace-nowrap">Learn more →</a>
          </p>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div class="inset-section sm:col-span-2">
              <p class="text-xs font-display font-semibold uppercase tracking-wider text-ca mb-2">The problem</p>
              <p class="text-xs text-muted-foreground leading-relaxed">Every memory provider ships with its own paper, proprietary methodology, and internal benchmark — making apples-to-apples comparison nearly impossible. The goal of this project is to establish a shared, neutral ground where providers are evaluated under the same conditions.</p>
            </div>
            <div class="inset-section">
              <p class="text-xs font-display font-semibold uppercase tracking-wider text-primary mb-2">Reproducible</p>
              <p class="text-xs text-muted-foreground leading-relaxed">All datasets, prompts, and scoring logic are open source. Anyone can run the benchmark locally or extend it with new providers and datasets.</p>
            </div>
            <div class="inset-section">
              <p class="text-xs font-display font-semibold uppercase tracking-wider text-primary mb-2">Comprehensive</p>
              <p class="text-xs text-muted-foreground leading-relaxed">Covers multiple-choice and open-ended tasks across multiple domains — from personal preference tracking to multi-hop agent trajectory reasoning.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Results -->
      <section>
        <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-8">Results</p>

        <div v-if="loading" class="text-center mt-16">
          <p class="text-muted-foreground text-sm animate-pulse">Loading…</p>
        </div>
        <div v-else-if="error" class="text-destructive text-sm">{{ error }}</div>
        <div v-else-if="datasets.length === 0">
          <p class="text-muted-foreground text-sm">No results yet. Run <code class="font-mono text-xs bg-secondary px-2 py-0.5 rounded">omb run …</code> to get started.</p>
        </div>
        <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <Card
            v-for="(ds, i) in datasets" :key="ds.name"
            v-reveal="i * 70"
            @click="router.push(`/dataset/${encodeURIComponent(ds.name)}`)"
            class="p-5 flex flex-col gap-4 cursor-pointer hover:border-primary/30 hover:bg-secondary/30 transition-colors"
          >
            <!-- Card header -->
            <div class="flex items-start justify-between gap-2">
              <h2 class="font-display font-semibold text-foreground text-base leading-tight">{{ ds.name }}</h2>
              <Badge v-if="catalog.datasets[ds.name]"
                     :variant="catalog.datasets[ds.name].task === 'MCQ' ? 'mcq' : 'open'"
                     class="shrink-0 mt-0.5">
                {{ catalog.datasets[ds.name].task === 'MCQ' ? 'Multiple Choice' : 'Open-ended' }}
              </Badge>
            </div>

            <!-- Description -->
            <p v-if="catalog.datasets[ds.name]" class="text-sm text-muted-foreground leading-relaxed">
              {{ catalog.datasets[ds.name].description }}
            </p>

            <!-- Mini accuracy chart -->
            <div v-if="ds.chart.rows.length" class="mt-1">
              <p class="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground/40 mb-2">Accuracy</p>
              <BarChart :rows="ds.chart.rows" :max="1"
                        :format="v => (v * 100).toFixed(1) + '%'"
                        variant="accuracy" label-width="w-20" />
            </div>

            <!-- Card footer -->
            <div class="flex items-center justify-between mt-auto pt-2 border-t border-border/50">
              <span class="text-xs text-muted-foreground/50">
                {{ ds.splits.size }} split{{ ds.splits.size !== 1 ? 's' : '' }} ·
                {{ ds.runs }} run{{ ds.runs !== 1 ? 's' : '' }}
              </span>
              <span class="text-primary text-xs font-medium">Explore →</span>
            </div>
          </Card>
        </div>
      </section>

      <!-- Comparison table -->
      <section v-if="!loading && !error && comparisonTable.providers.length">
        <div class="flex items-center justify-between mb-8">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60">Comparison</p>
          <label v-if="comparisonTable.hasExternal" class="flex items-center gap-2 cursor-pointer select-none">
            <span class="text-xs text-muted-foreground">Show unverified</span>
            <button @click="showUnverified = !showUnverified"
                    class="relative w-8 h-4 rounded-full transition-colors"
                    :class="showUnverified ? 'bg-primary' : 'bg-border'">
              <span class="absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform"
                    :class="showUnverified ? 'translate-x-4' : ''"></span>
            </button>
          </label>
        </div>

        <Card class="overflow-x-auto">
          <UiTable>
            <TableHeader>
              <!-- Dataset group header -->
              <TableRow>
                <TableHead class="border-b-0"></TableHead>
                <template v-for="group in comparisonTable.datasetGroups" :key="group.dataset">
                  <TableHead :colspan="group.splits.length" class="text-center border-l border-border/50 border-b-0 text-foreground font-semibold">
                    {{ group.dataset }}
                  </TableHead>
                </template>
              </TableRow>
              <!-- Split header (sortable) -->
              <TableRow>
                <TableHead :sortable="true" @click="toggleCmpSort('provider')">Provider{{ cmpSortIcon('provider') }}</TableHead>
                <TableHead v-for="col in comparisonTable.columns" :key="col.dataset + col.split"
                           :right="true" :sortable="true" class="border-l border-border/50"
                           @click="toggleCmpSort(col.dataset + '||' + col.split)">
                  {{ col.split }}{{ cmpSortIcon(col.dataset + '||' + col.split) }}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow v-for="provider in comparisonTable.providers" :key="provider.label">
                <TableCell :primary="true">
                  <div class="flex items-center gap-2 whitespace-nowrap">
                    <img v-if="provider.logo" :src="provider.logo"
                         class="w-4 h-4 rounded object-contain shrink-0" @error="$event.target.style.display='none'" />
                    <span>{{ provider.label }}</span>
                  </div>
                </TableCell>
                <TableCell v-for="col in comparisonTable.columns"
                           :key="col.dataset + col.split"
                           :right="true"
                           class="tabular-nums border-l border-border/50"
                           :class="provider.cells[col.dataset + '||' + col.split]?.accuracy === comparisonTable.bestPerCol[col.dataset + '||' + col.split]
                             ? 'font-bold text-primary bg-primary/5'
                             : ''">
                  <template v-if="provider.cells[col.dataset + '||' + col.split]">
                    <span class="inline-flex items-center gap-0.5">
                      {{ (provider.cells[col.dataset + '||' + col.split].accuracy * 100).toFixed(1) + '%' }}
                      <a v-if="provider.cells[col.dataset + '||' + col.split].unverified && provider.cells[col.dataset + '||' + col.split].source_url"
                         :href="provider.cells[col.dataset + '||' + col.split].source_url"
                         target="_blank" rel="noopener" @click.stop
                         class="text-ca hover:text-ca/80 cursor-pointer"
                         :title="[provider.cells[col.dataset + '||' + col.split].source_label, provider.cells[col.dataset + '||' + col.split].comment].filter(Boolean).join(' — ')">*</a>
                      <span v-else-if="provider.cells[col.dataset + '||' + col.split].unverified"
                            class="text-ca"
                            :title="provider.cells[col.dataset + '||' + col.split].comment || 'Unverified'">*</span>
                    </span>
                  </template>
                  <template v-else>
                    <span class="text-muted-foreground/30">—</span>
                  </template>
                </TableCell>
              </TableRow>
            </TableBody>
          </UiTable>
        </Card>

        <div class="flex items-center justify-between mt-3">
          <p v-if="showUnverified" class="text-xs text-muted-foreground/50"><span class="text-ca">*</span> Unverified — sourced from external papers, not independently reproduced.</p>
          <span v-else></span>
          <a href="https://github.com/vectorize-io/open-memory-benchmark" target="_blank" rel="noopener"
             class="text-xs text-muted-foreground hover:text-foreground transition-colors">
            Want to add your memory system? Contribute on GitHub →
          </a>
        </div>
      </section>

      <!-- About -->
      <section>
        <button @click="toggleAbout" class="flex items-center gap-2 mb-8 group">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 group-hover:text-muted-foreground transition-colors">About</p>
          <span class="text-muted-foreground/40 text-xs transition-transform" :class="aboutOpen ? 'rotate-180' : ''">▼</span>
        </button>

        <div v-if="aboutOpen" class="space-y-10">
          <div v-reveal class="space-y-2">
            <p class="text-sm text-muted-foreground leading-relaxed max-w-2xl">
              AMB evaluates how well memory and retrieval systems can store, index, and surface relevant information
              from long-context personal conversations and agent trajectories.
              A benchmark <strong class="text-foreground font-semibold">run</strong> ingests documents into a memory provider,
              then answers queries using retrieved context and scores the results.
            </p>
          </div>

          <div v-reveal="80">
            <p class="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground/60 mb-3">Splits</p>
            <p class="text-xs text-muted-foreground leading-relaxed max-w-2xl">
              A well-defined subset of a dataset — often by context-window size or QA type, giving a controlled axis for comparison.
              e.g. <code class="bg-secondary px-1 rounded text-xs">32k</code> <code class="bg-secondary px-1 rounded text-xs">128k</code> in personamem;
              <code class="bg-secondary px-1 rounded text-xs">bitcoin</code> in tempo.
            </p>
          </div>

          <div v-reveal="160">
            <p class="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground/60 mb-3">Categories</p>
            <p class="text-xs text-muted-foreground leading-relaxed max-w-2xl">
              Optional sub-filters within a split — drill into a query type or document partition without re-running.
              e.g. <code class="bg-secondary px-1 rounded text-xs">single-hop</code> vs <code class="bg-secondary px-1 rounded text-xs">temporal</code> in locomo.
            </p>
          </div>

          <div v-reveal="240">
            <p class="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground/60 mb-4">Modes</p>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div v-for="(m, mName) in catalog.modes" :key="mName" class="inset-section">
                <span class="text-xs font-semibold text-primary block mb-1">{{ mName }}</span>
                <p class="text-xs text-muted-foreground leading-relaxed">{{ m.description }}</p>
              </div>
            </div>
          </div>
        </div>
      </section>

    </div>
  </div>
</template>
