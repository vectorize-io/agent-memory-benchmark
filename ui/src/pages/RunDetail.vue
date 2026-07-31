<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useHead } from '@unhead/vue'
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
const catalog     = ref({ datasets: {} })
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

onMounted(() => {
  load()
  fetch('/api/catalog').then(r => r.json()).then(d => catalog.value = d).catch(() => {})
})
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
        if (!map[axis][v]) map[axis][v] = { correct: 0, total: 0, scoreSum: 0, scoreCount: 0 }
        map[axis][v].total++
        if (r.correct) map[axis][v].correct++
        if (r.score != null) { map[axis][v].scoreSum += r.score; map[axis][v].scoreCount++ }
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

useHead(computed(() => {
  if (!data.value) return { title: 'Run — Agent Memory Benchmark' }
  const d = data.value
  const mem = d.run_name && d.run_name !== d.memory_provider ? `${d.memory_provider} (${d.run_name})` : d.memory_provider
  const title = `${d.dataset ? d.dataset + ' · ' : ''}${d.split || d.domain || ''} · ${mem} — AMB`
  return {
    title,
    meta: [
      { property: 'og:title', content: title },
      { name: 'robots', content: 'noindex' },
    ],
  }
}))

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
  const hasScores = results.value.some(r => r.score != null)
  let pct
  if (hasScores) {
    const scored = results.value.filter(r => r.score != null)
    pct = scored.length ? scored.reduce((s, r) => s + r.score, 0) / scored.length : 0
  } else {
    pct = total ? pass / total : 0
  }
  return { pass, total, pct, hasScores }
})

const perf = computed(() => {
  if (!data.value) return null
  const recTimes  = results.value.map(r => r.retrieve_time_ms).filter(v => v != null).sort((a, b) => a - b)
  const ctxTokens = results.value.map(r => r.context_tokens).filter(v => v != null)
  return { recTimes, ctxTokens, ingestAvg: data.value.ingested_docs ? data.value.ingestion_time_ms / data.value.ingested_docs : 0 }
})

// ── Multi-turn agent view (declared by `view: "agent"` in the result JSON) ──
const isAgent = computed(() => data.value?.view === 'agent')

function parseSteps(reasoning) {  // fallback for old string-based traces
  if (!reasoning) return []
  return reasoning.split('\n').filter(l => l.trim()).map(l => {
    const t = l.match(/^(\S+)\s*(.*)$/)
    if (t && /^(grep|read|glob|bash|edit|write|apply_patch|hindsight\w*|list|webfetch)$/.test(t[1]))
      return { k: 'tool', tool: t[1], arg: t[2].trim() }
    return { k: 'say', text: l }
  })
}
const steps = computed(() => {
  if (!isAgent.value) return []
  const tj = active.value?.trajectory
  const raw = Array.isArray(tj) && tj.length ? tj : parseSteps(active.value?.reasoning)
  let prevIn = 0
  return raw.map((s, i) => {
    const o = { ...s, i, mem: s.tool ? String(s.tool).startsWith('hindsight') : false }
    if (s.tok_in != null) {           // +Δ = NEW context this model step added (cumulative - previous)
      o.tok_in_delta = Math.max(0, s.tok_in - prevIn)
      prevIn = s.tok_in
    }
    return o
  })
})

// Group steps into TURNS: opencode issues several tool calls from ONE prompt (one model
// turn). Steps sharing a tok_in belong to the same turn; tokens are per-turn, shown once.
// Feedback markers (🔁) and submitted patches stay as standalone blocks between turns.
const turns = computed(() => {
  const out = []
  let cur = null, prevIn
  for (const s of steps.value) {
    const isSep = s.k === 'patch' || (s.k === 'say' && String(s.text || '').startsWith('🔁'))
    if (isSep) {
      out.push({ kind: s.k === 'patch' ? 'patch' : 'feedback', step: s, key: 'sep' + s.i })
      cur = null; prevIn = undefined; continue
    }
    if (!cur || s.tok_in !== prevIn) {
      cur = { kind: 'turn', key: 'turn' + s.i, n: out.filter(t => t.kind === 'turn').length + 1,
              tok_in: s.tok_in, tok_in_delta: s.tok_in_delta, tok_out: s.tok_out,
              tok_cache: s.tok_cache, tok_reason: s.tok_reason, notes: [], tools: [] }
      out.push(cur)
    }
    prevIn = s.tok_in
    if (s.k === 'say') cur.notes.push(s)
    else cur.tools.push(s)
  }
  return out
})

const expandedSteps = ref(new Set())
function toggleStep(i) {
  const s = new Set(expandedSteps.value)
  s.has(i) ? s.delete(i) : s.add(i)
  expandedSteps.value = s
}
function outPreview(s) {
  const o = (s.out || '').replace(/\s+/g, ' ').trim()
  return o.length > 80 ? o.slice(0, 80) + ' …' : o
}
function tokFmt(n) {
  n = n ?? 0
  return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'k' : String(n)
}
watch(activeIndex, () => { expandedSteps.value = new Set() })

function patchLines(patch) {
  if (!patch) return []
  return patch.split('\n').map((text, i) => {
    let cls = ''
    if (text.startsWith('+') && !text.startsWith('+++')) cls = 'add'
    else if (text.startsWith('-') && !text.startsWith('---')) cls = 'del'
    else if (text.startsWith('@@')) cls = 'hunk'
    else if (/^(diff |index |\+\+\+|---)/.test(text)) cls = 'meta'
    return { text, cls, i }
  })
}
const patch = computed(() => isAgent.value ? patchLines(active.value?.answer) : [])

const memBlocks = computed(() => {
  const c = active.value?.context
  if (!c) return []
  return c.split(/\n(?=## Memory )/).map(b => b.trim()).filter(Boolean)
})

const agentStats = computed(() => {
  if (!isAgent.value) return null
  const rs = results.value
  const sum = k => rs.map(r => r.meta?.[k]).filter(v => v != null).reduce((a, b) => a + b, 0)
  const sumTok = k => rs.map(r => r.meta?.tokens?.[k] ?? 0).reduce((a, b) => a + b, 0)
  return {
    cost: rs.map(r => r.meta?.cost_usd ?? 0).reduce((a, b) => a + b, 0),
    interv: sum('interventions'),
    inTok: sumTok('input') + sumTok('cache_read') + sumTok('cache_write'),
    cachedTok: sumTok('cache_read'),
    outTok: sumTok('output') + sumTok('reasoning'),
    turns: sum('turns'),
    sde: rs.some(r => r.meta?.cost_usd != null || r.meta?.interventions != null),
    tok: sum('out_tokens'),
    memTasks: rs.filter(r => (r.meta?.memories_injected ?? 0) > 0).length,
  }
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
          <span class="text-muted-foreground text-sm">{{ summary.hasScores ? 'avg score' : summary.pass + ' / ' + summary.total + ' correct' }}</span>
          <Badge v-if="data?.oracle" variant="default" class="ml-auto">oracle</Badge>
        </div>

        <div v-if="data?.answer_llm || data?.judge_llm" class="text-sm text-muted-foreground mb-2 space-y-0.5">
          <div v-if="data.answer_llm"><span class="text-foreground/70">Answer LLM</span> {{ data.answer_llm }}</div>
          <div v-if="data.judge_llm && data.mode !== 'coding'"><span class="text-foreground/70">Judge LLM</span> {{ data.judge_llm }}</div>
        </div>

        <div v-if="isAgent && agentStats?.sde" class="grid grid-cols-2 gap-1.5">
          <div v-for="[label, val] in [
            ['Total cost', '$' + agentStats.cost.toFixed(3)],
            ['Interventions', agentStats.interv.toLocaleString()],
            ['Tokens ↑in / ↓out', (agentStats.inTok/1000).toFixed(0) + 'k / ' + (agentStats.outTok/1000).toFixed(1) + 'k'],
            ['⚡ Cached input', (agentStats.cachedTok/1000).toFixed(0) + 'k (' + (agentStats.inTok ? Math.round(100*agentStats.cachedTok/agentStats.inTok) : 0) + '% of ↑)'],
          ]" :key="label" class="stat-box">
            <p class="text-muted-foreground/85 text-sm mb-0.5">{{ label }}</p>
            <p class="font-semibold text-foreground text-sm">{{ val }}</p>
          </div>
        </div>
        <div v-else-if="isAgent && agentStats" class="grid grid-cols-3 gap-1.5">
          <div v-for="[label, val] in [
            ['Output tok', agentStats.tok.toLocaleString()],
            ['Tool turns', agentStats.turns.toLocaleString()],
            ['Mem tasks', agentStats.memTasks + ' / ' + results.length],
          ]" :key="label" class="stat-box">
            <p class="text-muted-foreground/85 text-sm mb-0.5">{{ label }}</p>
            <p class="font-semibold text-foreground text-sm">{{ val }}</p>
          </div>
        </div>
        <div v-else-if="perf && data?.mode !== 'coding'" class="grid grid-cols-3 gap-1.5">
          <div v-for="[label, val] in [
            ['Ingest/doc', perf.ingestAvg.toFixed(1) + 'ms'],
            ['Recall p50', pct50(perf.recTimes).toFixed(0) + 'ms'],
            ['Ctx tokens', Math.round(avg(perf.ctxTokens)).toLocaleString()],
          ]" :key="label" class="stat-box">
            <p class="text-muted-foreground/85 text-sm mb-0.5">{{ label }}</p>
            <p class="font-semibold text-foreground text-sm">{{ val }}</p>
          </div>
        </div>
      </div>

      <!-- Category filters -->
      <div v-if="allAxes.length" class="sidebar-section overflow-y-auto max-h-[25vh]">
        <div v-for="axis in allAxes" :key="axis" class="px-4 py-2.5 border-b border-border last:border-0">
          <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-1.5">{{ axis }}</p>
          <table class="w-full text-sm">
            <tbody>
              <tr v-for="(stats, cat) in axisStats[axis]" :key="cat"
                  @click="toggleCat(axis, cat)"
                  :class="catFilter[axis]?.has(cat) ? 'cat-row-active' : 'cat-row'">
                <td class="py-0.5 pr-2 text-muted-foreground truncate max-w-[110px]">{{ cat }}</td>
                <td class="py-0.5 pr-2 text-right text-muted-foreground/80 w-10">{{ stats.total }}</td>
                <td class="py-0.5 text-right font-semibold w-10" :style="{ color: accuracyColor(stats.scoreCount ? stats.scoreSum / stats.scoreCount : stats.correct / stats.total) }">
                  {{ ((stats.scoreCount ? stats.scoreSum / stats.scoreCount : stats.correct / stats.total) * 100).toFixed(1) }}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="Object.values(catFilter).some(v => v?.size)" class="px-4 py-2">
          <button @click="catFilter = {}" class="text-sm text-muted-foreground hover:text-foreground transition-colors">✕ clear filters</button>
        </div>
      </div>

      <!-- Search + filter -->
      <div class="sidebar-section px-3 py-2.5 shrink-0">
        <Input v-model="search" placeholder="Search queries…" class="mb-2 h-8 text-sm" />
        <div class="flex items-center justify-between">
          <div class="flex gap-1">
            <button v-for="[val, label, cls] in [['all','All','hover:bg-secondary'], ['correct','✓ Pass','hover:bg-cg/10 hover:text-cg'], ['wrong','✗ Fail','hover:bg-cr/10 hover:text-cr']]"
                    :key="val" @click="filter = val"
                    :class="[
                      'px-2.5 py-1 rounded text-sm font-medium transition-colors',
                      filter === val && val === 'all'     && 'bg-secondary text-foreground',
                      filter === val && val === 'correct' && 'bg-cg/15 text-cg',
                      filter === val && val === 'wrong'   && 'bg-cr/15 text-cr',
                      filter !== val && 'text-muted-foreground ' + cls,
                    ]">{{ label }}</button>
          </div>
          <span class="text-sm text-muted-foreground/80">{{ filtered.length }} / {{ results.length }}</span>
        </div>
      </div>

      <!-- Result list -->
      <div class="flex-1 overflow-y-auto">
        <p v-if="!filtered.length" class="p-4 text-sm text-muted-foreground">No results match this filter.</p>
        <button v-for="{ r, i } in filtered" :key="i"
                @click="activeIndex = i"
                :class="i === activeIndex ? 'item-active' : 'hover:bg-secondary/30'"
                class="w-full text-left px-4 py-2.5 border-b border-border/50 last:border-0 transition-colors">
          <div class="flex items-start gap-2">
            <span :class="r.correct ? 'text-cg' : 'text-cr'" class="font-bold text-sm shrink-0 mt-0.5">{{ r.score != null ? r.score.toFixed(2) : (r.correct ? '✓' : '✗') }}</span>
            <div class="min-w-0">
              <p class="text-sm text-foreground leading-snug line-clamp-2">{{ (r.query || '').split('\n')[0].trim().slice(0, 80) }}</p>
              <div class="flex items-center gap-1 mt-0.5 flex-wrap">
                <span class="text-sm font-mono text-muted-foreground/80">{{ r.query_id.slice(0, 8) }}</span>
                <Badge v-for="c in Object.values(r.category_axes ?? {}).flat()" :key="c" variant="default" class="text-sm">{{ c }}</Badge>
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
          <div class="flex items-center justify-between text-sm text-muted-foreground">
            <button @click="navigate(-1)" :disabled="activeIndex === 0"
                    class="hover:text-foreground transition-colors disabled:opacity-25 disabled:pointer-events-none">← prev</button>
            <span class="font-mono flex items-center gap-2.5">
              {{ activeIndex + 1 }} / {{ results.length }}
              <template v-if="isAgent && active.meta?.cost_usd != null">
                <span class="pill" title="Human-like feedback rounds needed">🔁 {{ active.meta?.interventions ?? '–' }} interv</span>
                <span class="pill" title="Cost (USD)">💲 {{ active.meta?.cost_usd?.toFixed(3) ?? '–' }}</span>
                <span class="pill" title="Tokens ↑in (fresh + cached, re-sent every turn) / ↓out · ⚡ = cached portion of ↑, billed ~10× cheaper">🔤 {{ ((active.meta?.tokens?.input ?? 0) + (active.meta?.tokens?.cache_read ?? 0)).toLocaleString() }} in <span v-if="active.meta?.tokens?.cache_read" class="pill-cache">⚡{{ (active.meta.tokens.cache_read).toLocaleString() }}</span> / {{ ((active.meta?.tokens?.output ?? 0) + (active.meta?.tokens?.reasoning ?? 0)).toLocaleString() }} out</span>
                <span class="pill" title="Tool calls (agent turns)">🔧 {{ active.meta?.turns ?? '–' }} turns</span>
                <span class="pill" title="Wall time">⏱ {{ active.meta?.wall_s ?? '–' }}s</span>
              </template>
              <template v-else-if="isAgent">
                <span class="pill" title="Tool calls (agent turns)">🔧 {{ active.meta?.turns ?? '–' }} turns</span>
                <span class="pill" title="Output tokens">🔤 {{ active.meta?.out_tokens?.toLocaleString() ?? '–' }} tok</span>
                <span class="pill" title="Wall time">⏱ {{ active.meta?.elapsed_s ?? '–' }}s</span>
              </template>
              <template v-else>
                <span class="pill" title="Memory recall latency">⏱ recall {{ active.retrieve_time_ms?.toFixed(0) }}ms</span>
                <span class="pill" title="Context tokens fed to the LLM">🔤 {{ active.context_tokens?.toLocaleString() }} tok</span>
              </template>
            </span>
            <button @click="navigate(1)" :disabled="activeIndex === results.length - 1"
                    class="hover:text-foreground transition-colors disabled:opacity-25 disabled:pointer-events-none">next →</button>
          </div>

          <!-- ─────────────── Multi-turn agent view ─────────────── -->
          <template v-if="isAgent">
            <div :class="active.correct ? 'verdict-pass' : 'verdict-fail'" class="text-sm flex items-center gap-2">
              <span class="font-semibold">{{ active.correct ? '✓ Resolved' : '✗ Unresolved' }}</span>
              <span v-if="active.judge_reason" class="opacity-75 font-normal">— {{ active.judge_reason }}</span>
              <span class="ml-auto font-mono text-muted-foreground/70 text-xs">{{ active.query_id }}</span>
            </div>

            <section>
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Issue</p>
              <Card class="p-4 text-sm text-foreground leading-relaxed whitespace-pre-wrap">{{ active.query }}</Card>
            </section>

            <section v-if="active.git_history && active.git_history.length">
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">
                Repository history <span class="text-muted-foreground/60 font-normal normal-case tracking-normal">· {{ active.git_history.length }} commits (newest first) — click to expand the diff</span>
              </p>
              <div class="trajectory">
                <details v-for="(c, i) in active.git_history" :key="i">
                  <summary class="traj-step traj-clickable git-commit">
                    <span class="git-sha">{{ c.sha }}</span>
                    <span class="git-subject">{{ c.subject }}</span>
                  </summary>
                  <div class="traj-detail">
                    <pre v-if="c.body" class="git-body">{{ c.body }}</pre>
                    <div class="diff-block"><div v-for="l in patchLines(c.diff)" :key="l.i" :class="'diff-line diff-' + (l.cls || 'ctx')">{{ l.text || ' ' }}</div></div>
                  </div>
                </details>
              </div>
            </section>
            <section v-else-if="memBlocks.length">
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">
                Memory injected <span class="text-muted-foreground/60 font-normal normal-case tracking-normal">· {{ memBlocks.length }} recalled</span>
              </p>
              <div class="space-y-2">
                <Card v-for="(m, i) in memBlocks" :key="i" class="p-3 text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap border-l-2 border-primary/60">{{ m }}</Card>
              </div>
            </section>

            <section>
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">
                Agent trajectory <span class="text-muted-foreground/60 font-normal normal-case tracking-normal">· {{ turns.filter(t => t.kind === 'turn').length }} turns · {{ steps.length }} steps</span>
              </p>
              <div class="trajectory">
                <template v-for="t in turns" :key="t.key">
                  <!-- one model turn: reasoning/notes + the batch of tool calls it issued + ONE per-turn token badge -->
                  <div v-if="t.kind === 'turn'" class="turn">
                    <div class="turn-head">
                      <span class="turn-n">turn {{ t.n }}</span>
                      <span v-if="t.tok_reason" class="turn-reason" title="Hidden reasoning tokens this turn — Gemini bills these (as output) but does NOT expose the thinking text, so there's nothing to display but the count.">🧠 {{ tokFmt(t.tok_reason) }}</span>
                      <span v-if="t.tok_out != null" class="traj-tok turn-tok" title="Per-TURN tokens — the model issues ALL of this turn's tool calls from ONE prompt. ↑ cumulative context (system prompt + tool defs + every prior turn), mostly cached · +Δ NEW context since the previous turn (the prior turn's tool results) · ↓ tokens generated this turn (tool calls + reasoning)"><span class="traj-tok-arrow">↑</span>{{ tokFmt(t.tok_in) }}<span v-if="t.tok_cache" class="turn-cache" title="cached portion of this turn's prompt — billed at the discount rate (~$0.15/1M vs $1.50)">⚡{{ tokFmt(t.tok_cache) }}</span><span v-if="t.tok_in_delta != null" class="traj-tok-delta">+{{ tokFmt(t.tok_in_delta) }}</span> <span class="traj-tok-arrow">↓</span>{{ tokFmt(t.tok_out) }}</span>
                    </div>
                    <p v-for="note in t.notes" :key="note.i" class="turn-note">{{ note.text }}</p>
                    <template v-for="s in t.tools" :key="s.i">
                      <div class="traj-step" :class="{ 'traj-clickable': s.out || s.input }"
                           @click="(s.out || s.input) && toggleStep(s.i)">
                        <span :class="['traj-tool', s.mem && 'traj-tool-mem']">{{ s.tool }}</span>
                        <span class="traj-mid">
                          <span class="traj-arg">{{ s.arg }}</span>
                          <span v-if="s.out && !expandedSteps.has(s.i)" class="traj-out-prev"> ↳ {{ outPreview(s) }}</span>
                        </span>
                        <span v-if="s.out || s.input" class="traj-exp">{{ expandedSteps.has(s.i) ? '▾' : '▸' }}</span>
                      </div>
                      <div v-if="expandedSteps.has(s.i)" class="traj-detail">
                        <div v-if="s.input"><span class="traj-detail-label">input</span><pre>{{ s.input }}</pre></div>
                        <div v-if="s.out"><span class="traj-detail-label">output</span><pre>{{ s.out }}</pre></div>
                      </div>
                    </template>
                  </div>
                  <!-- feedback round separator -->
                  <div v-else-if="t.kind === 'feedback'" class="traj-step"><span class="traj-say">{{ t.step.text }}</span></div>
                  <!-- submitted patch (✓/✗ + diff) -->
                  <template v-else-if="t.kind === 'patch'">
                    <div class="traj-step traj-clickable" @click="toggleStep(t.step.i)">
                      <span :class="['traj-patch', t.step.passed ? 'traj-patch-ok' : 'traj-patch-fail']">{{ t.step.passed ? '✓ submitted' : '✗ submitted' }}</span>
                      <span class="traj-mid"><span class="traj-arg">{{ t.step.round }}</span><span class="traj-out-prev"> · {{ t.step.pytest }}</span></span>
                      <span class="traj-exp">{{ expandedSteps.has(t.step.i) ? '▾' : '▸' }}</span>
                    </div>
                    <div v-if="expandedSteps.has(t.step.i)" class="traj-detail">
                      <span class="traj-detail-label">patch submitted this round ({{ t.step.pytest }})</span>
                      <div class="diff-block"><div v-for="l in patchLines(t.step.patch)" :key="l.i" :class="'diff-line diff-' + (l.cls || 'ctx')">{{ l.text || ' ' }}</div></div>
                    </div>
                  </template>
                </template>
              </div>
            </section>

            <section>
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Final patch</p>
              <div class="diff-block"><div v-for="l in patch" :key="l.i" :class="'diff-line diff-' + (l.cls || 'ctx')">{{ l.text || ' ' }}</div></div>
            </section>
          </template>

          <!-- ─────────────── RAG view ─────────────── -->
          <template v-else>
          <section>
            <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2 flex items-center gap-2">
              Query
              <span class="font-mono text-muted-foreground/70 font-normal normal-case tracking-normal">{{ active.query_id }}</span>
              <button @click="copyId(active.query_id)" class="text-muted-foreground/70 hover:text-muted-foreground transition-colors" title="Copy ID">
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
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">System answer</p>
              <Card class="p-4 flex-1 text-sm text-foreground leading-relaxed">
                <span v-if="active.answer">{{ active.answer }}</span>
                <em v-else class="text-muted-foreground">empty</em>
              </Card>
            </section>
            <section class="flex flex-col">
              <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Gold answer</p>
              <Card class="p-4 flex-1 text-sm text-foreground leading-relaxed">
                {{ active.gold_answers?.[0] ?? '' }}
                <Badge v-for="a in (active.gold_answers?.slice(1) ?? [])" :key="a" variant="secondary" class="ml-2 font-mono">{{ a }}</Badge>
              </Card>
            </section>
          </div>

          <section>
            <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Judge verdict</p>
            <div :class="active.correct ? 'verdict-pass' : 'verdict-fail'" class="text-sm">
              <span class="font-semibold">{{ active.score != null ? 'Score: ' + active.score.toFixed(3) : (active.correct ? '✓ Correct' : '✗ Incorrect') }}</span>
              <span v-if="active.judge_reason" class="ml-2 opacity-75 font-normal">— {{ active.judge_reason }}</span>
            </div>
            <details v-if="catalog.datasets?.[data?.dataset]?.scoring_note" class="mt-2 text-sm text-muted-foreground">
              <summary class="cursor-pointer hover:text-foreground/80 transition-colors">How is this scored?</summary>
              <p class="mt-1.5 leading-relaxed pl-3 border-l-2 border-muted">{{ catalog.datasets[data.dataset].scoring_note }}</p>
            </details>
          </section>

          <hr class="border-border/40" />

          <section v-if="active.reasoning">
            <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Reasoning</p>
            <Card class="p-4 text-sm text-muted-foreground leading-relaxed italic">{{ active.reasoning }}</Card>
          </section>

          <section>
            <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 mb-2">Injected context</p>
            <div class="code-block"><pre>{{ active.context || '(no context)' }}</pre></div>
          </section>

          <section v-if="active.raw_response">
            <details>
              <summary class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/80 cursor-pointer hover:text-muted-foreground transition-colors">Raw provider response</summary>
              <div class="mt-2 code-block"><pre>{{ JSON.stringify(active.raw_response, null, 2) }}</pre></div>
            </details>
          </section>
          </template>
        </div>
      </div>
    </main>
    </div>
  </div>
</template>
