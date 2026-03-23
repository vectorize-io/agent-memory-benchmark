<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { fetchManifest, fetchExternalResults } from '../api.js'
import { accuracyColor } from '../utils.js'
import Card from '@/components/ui/card.vue'
import Badge from '@/components/ui/badge.vue'
import BarChart from '@/components/ui/bar-chart.vue'
import TopNav from '@/components/ui/top-nav.vue'

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
            <p class="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground/60 mb-4">Memory Providers</p>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div v-for="(p, pName) in catalog.providers" :key="pName" class="inset-section">
                <div class="flex items-center gap-1.5 mb-2">
                  <img v-if="p.logo" :src="p.logo" class="w-4 h-4 rounded object-contain shrink-0" @error="$event.target.style.display='none'" />
                  <span class="text-xs font-semibold text-primary">{{ pName }}</span>
                  <a v-if="p.link" :href="p.link" target="_blank" rel="noopener" @click.stop class="ml-auto text-muted-foreground/50 hover:text-muted-foreground text-xs">↗</a>
                </div>
                <template v-if="!p.variants">
                  <Badge :variant="p.kind === 'cloud' ? 'cloud' : 'local'" class="mr-1">{{ p.kind }}</Badge>
                  <span class="text-xs text-muted-foreground">{{ p.description }}</span>
                </template>
                <template v-else>
                  <div v-for="(v, vName) in p.variants" :key="vName" class="mt-2 first:mt-0 flex items-center gap-2">
                    <span class="text-xs text-muted-foreground/60 w-10 shrink-0">{{ vName }}</span>
                    <Badge :variant="v.kind === 'cloud' ? 'cloud' : 'local'">{{ v.kind }}</Badge>
                    <span class="text-xs text-muted-foreground">{{ v.description }}</span>
                  </div>
                </template>
              </div>
            </div>
          </div>

          <div v-reveal="320">
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
