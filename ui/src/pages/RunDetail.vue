<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchRunData } from '../api.js'
import { avg, pct50 } from '../utils.js'
import Badge from '@/components/ui/badge.vue'
import Input from '@/components/ui/input.vue'
import Button from '@/components/ui/button.vue'
import Card from '@/components/ui/card.vue'
import TopNav from '@/components/ui/top-nav.vue'

const route   = useRoute()
const router  = useRouter()
const runPath = computed(() => {
  const p = route.params.path
  return /\.json(\.gz)?$/.test(p) ? p : p + '.json'
})

const data        = ref(null)
const loading     = ref(true)
const error       = ref(null)
const filter      = ref('all')
const catFilter   = ref({})
const search      = ref('')
const activeIndex = ref(null)
const copied      = ref(false)

async function load() {
  loading.value = true; error.value = null; activeIndex.value = null
  data.value = null; catFilter.value = {}; filter.value = 'all'; search.value = ''
  try { data.value = await fetchRunData(runPath.value) }
  catch (e) { error.value = e.message }
  finally { loading.value = false }
}

onMounted(load)
watch(runPath, load)

const results = computed(() => data.value?.results ?? [])

watch(results, val => {
  if (!val.length) return
  const id = route.query.id
  const idx = id ? val.findIndex(r => r.query_id === id) : -1
  activeIndex.value = idx >= 0 ? idx : 0
})

watch(activeIndex, idx => {
  const id = idx != null ? results.value[idx]?.query_id : undefined
  router.replace({ query: { ...route.query, id: id || undefined } })
})

const axisStats = computed(() => {
  const map = {}
  results.value.forEach(r => {
    Object.entries(r.category_axes ?? {}).forEach(([axis, vals]) => {
      if (!map[axis]) map[axis] = {}
      ;(vals ?? []).forEach(v => {
        if (!map[axis][v]) map[axis][v] = { correct: 0, total: 0 }
        map[axis][v].total++
        if (r.correct) map[axis][v].correct++
      })
    })
  })
  return map
})

const allAxes = computed(() => Object.keys(axisStats.value).sort())

const filtered = computed(() =>
  results.value.map((r, i) => ({ r, i })).filter(({ r }) => {
    for (const [axis, vals] of Object.entries(catFilter.value))
      if (vals?.size && !(r.category_axes?.[axis] ?? []).some(v => vals.has(v))) return false
    if (filter.value === 'correct' && !r.correct) return false
    if (filter.value === 'wrong'   &&  r.correct) return false
    if (search.value) {
      const s = search.value.toLowerCase()
      if (!r.query?.toLowerCase().includes(s) && !r.query_id?.toLowerCase().includes(s)) return false
    }
    return true
  })
)

const active = computed(() => activeIndex.value != null ? results.value[activeIndex.value] : null)

const headerTitle = computed(() => {
  if (!data.value) return '—'
  const d = data.value
  const mem = d.run_name && d.run_name !== d.memory_provider ? `${d.memory_provider} (${d.run_name})` : d.memory_provider
  return `${d.dataset ? d.dataset + ' · ' : ''}${d.split || d.domain || '—'} · ${mem} · ${d.mode}`
})

const summary = computed(() => {
  if (!data.value) return null
  const pass = results.value.filter(r => r.correct).length
  const total = results.value.length
  return { pass, total, pct: total ? pass / total : 0 }
})

const perf = computed(() => {
  if (!data.value) return null
  const recTimes  = results.value.map(r => r.retrieve_time_ms).filter(v => v != null).sort((a, b) => a - b)
  const ctxTokens = results.value.map(r => r.context_tokens).filter(v => v != null)
  return { recTimes, ctxTokens, ingestAvg: data.value.ingested_docs ? data.value.ingestion_time_ms / data.value.ingested_docs : 0 }
})

function navigate(dir) {
  if (activeIndex.value == null) return
  const next = activeIndex.value + dir
  if (next >= 0 && next < results.value.length) activeIndex.value = next
}

function handleKey(e) {
  if (!data.value || document.activeElement.tagName === 'INPUT') return
  if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); navigate(1) }
  if (e.key === 'ArrowUp'   || e.key === 'k') { e.preventDefault(); navigate(-1) }
}

onMounted(() => document.addEventListener('keydown', handleKey))
onUnmounted(() => document.removeEventListener('keydown', handleKey))

function goBack() {
  data.value?.dataset ? router.push(`/dataset/${encodeURIComponent(data.value.dataset)}`) : router.push('/')
}

const queriesLink = computed(() => {
  if (!data.value?.dataset) return null
  const sp = data.value.split || data.value.domain
  if (!sp) return null
  return `/dataset/${encodeURIComponent(data.value.dataset)}/${encodeURIComponent(sp)}/queries`
})

function accuracyColor(pct) {
  return pct >= 0.7 ? '#34d399' : pct >= 0.4 ? '#fbbf24' : '#f87171'
}

let copyTimer = null
function copyId(id) {
  navigator.clipboard.writeText(id)
  copied.value = true
  clearTimeout(copyTimer)
  copyTimer = setTimeout(() => { copied.value = false }, 1500)
}

function toggleCat(axis, cat) {
  const s = new Set(catFilter.value[axis])
  s.has(cat) ? s.delete(cat) : s.add(cat)
  catFilter.value = { ...catFilter.value, [axis]: s }
}
</script>

<template>
  <div v-if="loading" class="fixed inset-0 flex items-center justify-center bg-background">
    <p class="text-muted-foreground text-sm animate-pulse">Loading…</p>
  </div>
  <div v-else-if="error" class="fixed inset-0 flex items-center justify-center bg-background">
    <p class="text-destructive text-sm max-w-md text-center">{{ error }}</p>
  </div>

  <div v-else class="flex flex-col h-screen overflow-hidden bg-background">
    <TopNav :crumbs="data ? [
      { label: data.dataset, to: `/dataset/${encodeURIComponent(data.dataset)}` },
      { label: data.split || data.domain, to: data.dataset && (data.split || data.domain) ? `/dataset/${encodeURIComponent(data.dataset)}/${encodeURIComponent(data.split || data.domain)}` : undefined },
      { label: data.memory_provider + (data.mode ? ' · ' + data.mode : '') },
    ] : []" />
    <!-- Sidebar -->
    <div class="flex flex-1 overflow-hidden">
    <aside class="sidebar w-[300px] shrink-0 flex flex-col overflow-hidden">

      <!-- Header -->
      <div class="sidebar-section px-4 pt-4 pb-3 shrink-0">
        <div class="flex items-center gap-2 mb-3">
          <h1 class="font-semibold text-sm text-foreground truncate" :title="headerTitle">{{ headerTitle }}</h1>
        </div>

        <div v-if="summary" class="flex items-center gap-3 mb-3">
          <span class="font-display text-2xl font-bold tracking-tight" :style="{ color: accuracyColor(summary.pct) }">
            {{ (summary.pct * 100).toFixed(1) }}%
          </span>
          <span class="text-muted-foreground text-xs">{{ summary.pass }} / {{ summary.total }} correct</span>
          <Badge v-if="data?.oracle" variant="default" class="ml-auto">oracle</Badge>
        </div>

        <div v-if="data?.answer_llm || data?.judge_llm" class="text-xs text-muted-foreground mb-2 space-y-0.5">
          <div v-if="data.answer_llm"><span class="text-foreground/70">Answer LLM</span> {{ data.answer_llm }}</div>
          <div v-if="data.judge_llm"><span class="text-foreground/70">Judge LLM</span> {{ data.judge_llm }}</div>
        </div>

        <div v-if="perf" class="grid grid-cols-3 gap-1.5">
          <div v-for="[label, val] in [
            ['Ingest/doc', perf.ingestAvg.toFixed(1) + 'ms'],
            ['Recall p50', pct50(perf.recTimes).toFixed(0) + 'ms'],
            ['Ctx tokens', Math.round(avg(perf.ctxTokens)).toLocaleString()],
          ]" :key="label" class="stat-box">
            <p class="text-muted-foreground/60 text-xs mb-0.5">{{ label }}</p>
            <p class="font-semibold text-foreground text-xs">{{ val }}</p>
          </div>
        </div>
      </div>

      <!-- Category filters -->
      <div v-if="allAxes.length" class="sidebar-section shrink-0">
        <div v-for="axis in allAxes" :key="axis" class="px-4 py-2.5 border-b border-border last:border-0">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-1.5">{{ axis }}</p>
          <table class="w-full text-xs">
            <tbody>
              <tr v-for="(stats, cat) in axisStats[axis]" :key="cat"
                  @click="toggleCat(axis, cat)"
                  :class="catFilter[axis]?.has(cat) ? 'cat-row-active' : 'cat-row'">
                <td class="py-0.5 pr-2 text-muted-foreground truncate max-w-[110px]">{{ cat }}</td>
                <td class="py-0.5 pr-2 text-right text-muted-foreground/50 w-10">{{ stats.total }}</td>
                <td class="py-0.5 text-right font-semibold w-10" :style="{ color: accuracyColor(stats.correct / stats.total) }">
                  {{ (stats.correct / stats.total * 100).toFixed(1) }}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="Object.values(catFilter).some(v => v?.size)" class="px-4 py-2">
          <button @click="catFilter = {}" class="text-xs text-muted-foreground hover:text-foreground transition-colors">✕ clear filters</button>
        </div>
      </div>

      <!-- Search + filter -->
      <div class="sidebar-section px-3 py-2.5 shrink-0">
        <Input v-model="search" placeholder="Search queries…" class="mb-2 h-8 text-xs" />
        <div class="flex items-center justify-between">
          <div class="flex gap-1">
            <button v-for="[val, label, cls] in [['all','All','hover:bg-secondary'], ['correct','✓ Pass','hover:bg-cg/10 hover:text-cg'], ['wrong','✗ Fail','hover:bg-cr/10 hover:text-cr']]"
                    :key="val" @click="filter = val"
                    :class="[
                      'px-2.5 py-1 rounded text-xs font-medium transition-colors',
                      filter === val && val === 'all'     && 'bg-secondary text-foreground',
                      filter === val && val === 'correct' && 'bg-cg/15 text-cg',
                      filter === val && val === 'wrong'   && 'bg-cr/15 text-cr',
                      filter !== val && 'text-muted-foreground ' + cls,
                    ]">{{ label }}</button>
          </div>
          <span class="text-xs text-muted-foreground/50">{{ filtered.length }} / {{ results.length }}</span>
        </div>
      </div>

      <!-- Result list -->
      <div class="flex-1 overflow-y-auto">
        <p v-if="!filtered.length" class="p-4 text-xs text-muted-foreground">No results match this filter.</p>
        <button v-for="{ r, i } in filtered" :key="i"
                @click="activeIndex = i"
                :class="i === activeIndex ? 'item-active' : 'hover:bg-secondary/30'"
                class="w-full text-left px-4 py-2.5 border-b border-border/50 last:border-0 transition-colors">
          <div class="flex items-start gap-2">
            <span :class="r.correct ? 'text-cg' : 'text-cr'" class="font-bold text-xs shrink-0 mt-0.5">{{ r.correct ? '✓' : '✗' }}</span>
            <div class="min-w-0">
              <p class="text-sm text-foreground leading-snug line-clamp-2">{{ (r.query || '').split('\n')[0].trim().slice(0, 80) }}</p>
              <div class="flex items-center gap-1 mt-0.5 flex-wrap">
                <span class="text-xs font-mono text-muted-foreground/50">{{ r.query_id.slice(0, 8) }}</span>
                <Badge v-for="c in Object.values(r.category_axes ?? {}).flat()" :key="c" variant="default" class="text-xs">{{ c }}</Badge>
              </div>
            </div>
          </div>
        </button>
      </div>
    </aside>

    <!-- Main panel -->
    <main class="flex-1 overflow-y-auto bg-background">
      <div class="p-6 max-w-3xl mx-auto">
        <p v-if="!active" class="text-muted-foreground text-sm mt-20 text-center">← Select a query</p>

        <div v-else class="space-y-5 pb-10">
          <div class="flex items-center justify-between text-xs text-muted-foreground">
            <button @click="navigate(-1)" :disabled="activeIndex === 0"
                    class="hover:text-foreground transition-colors disabled:opacity-25 disabled:pointer-events-none">← prev</button>
            <span class="font-mono flex items-center gap-2.5">
              {{ activeIndex + 1 }} / {{ results.length }}
              <span class="pill" title="Memory recall latency">⏱ recall {{ active.retrieve_time_ms?.toFixed(0) }}ms</span>
              <span class="pill" title="Context tokens fed to the LLM">🔤 {{ active.context_tokens?.toLocaleString() }} tok</span>
            </span>
            <button @click="navigate(1)" :disabled="activeIndex === results.length - 1"
                    class="hover:text-foreground transition-colors disabled:opacity-25 disabled:pointer-events-none">next →</button>
          </div>

          <section>
            <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2 flex items-center gap-2">
              Query
              <span class="font-mono text-muted-foreground/40 font-normal normal-case tracking-normal">{{ active.query_id }}</span>
              <button @click="copyId(active.query_id)" class="text-muted-foreground/40 hover:text-muted-foreground transition-colors" title="Copy ID">
                <svg v-if="!copied" xmlns="http://www.w3.org/2000/svg" class="inline w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="inline w-3 h-3 text-cg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              </button>
              <router-link v-if="queriesLink" :to="queriesLink + '?id=' + encodeURIComponent(active.query_id)"
                class="text-primary hover:text-primary/80 normal-case tracking-normal font-sans font-normal transition-colors"
                title="Explore this query in the dataset view">explore query ↗</router-link>
            </p>
            <Card class="p-4 text-sm text-foreground leading-relaxed whitespace-pre-wrap">{{ active.query }}</Card>
          </section>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <section class="flex flex-col">
              <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">System answer</p>
              <Card class="p-4 flex-1 text-sm text-foreground leading-relaxed">
                <span v-if="active.answer">{{ active.answer }}</span>
                <em v-else class="text-muted-foreground">empty</em>
              </Card>
            </section>
            <section class="flex flex-col">
              <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Gold answer</p>
              <Card class="p-4 flex-1 text-sm text-foreground leading-relaxed">
                {{ active.gold_answers?.[0] ?? '' }}
                <Badge v-for="a in (active.gold_answers?.slice(1) ?? [])" :key="a" variant="secondary" class="ml-2 font-mono">{{ a }}</Badge>
              </Card>
            </section>
          </div>

          <section>
            <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Judge verdict</p>
            <div :class="active.correct ? 'verdict-pass' : 'verdict-fail'" class="text-sm">
              <span class="font-semibold">{{ active.correct ? '✓ Correct' : '✗ Incorrect' }}</span>
              <span v-if="active.judge_reason" class="ml-2 opacity-75 font-normal">— {{ active.judge_reason }}</span>
            </div>
          </section>

          <hr class="border-border/40" />

          <section v-if="active.reasoning">
            <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Reasoning</p>
            <Card class="p-4 text-sm text-muted-foreground leading-relaxed italic">{{ active.reasoning }}</Card>
          </section>

          <section>
            <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Injected context</p>
            <div class="code-block"><pre>{{ active.context || '(no context)' }}</pre></div>
          </section>

          <section v-if="active.raw_response">
            <details>
              <summary class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 cursor-pointer hover:text-muted-foreground transition-colors">Raw provider response</summary>
              <div class="mt-2 code-block"><pre>{{ JSON.stringify(active.raw_response, null, 2) }}</pre></div>
            </details>
          </section>
        </div>
      </div>
    </main>
    </div>
  </div>
</template>
