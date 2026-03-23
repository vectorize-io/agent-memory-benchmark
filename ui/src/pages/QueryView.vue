<script setup>
import { ref, watch, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchQueries, fetchDocument } from '../api.js'
import Input from '@/components/ui/input.vue'
import Badge from '@/components/ui/badge.vue'
import Card from '@/components/ui/card.vue'
import Button from '@/components/ui/button.vue'
import TopNav from '@/components/ui/top-nav.vue'

const route   = useRoute()
const router  = useRouter()
const dataset = computed(() => decodeURIComponent(route.params.name))
const split   = computed(() => decodeURIComponent(route.params.split))

const allItems  = ref([])
const catFilter = ref({})
const search    = ref('')
const loading   = ref(true)
const error     = ref(null)
const activeId  = ref(null)

async function load() {
  loading.value = true; error.value = null; catFilter.value = {}
  try {
    const res = await fetchQueries(dataset.value, split.value, { limit: 99999 })
    allItems.value = res.items
  } catch (e) { error.value = e.message }
  finally { loading.value = false }
}

onMounted(() => {
  const id = route.query.id
  load().then(() => {
    activeId.value = id ? (allItems.value.find(q => q.id === id)?.id ?? allItems.value[0]?.id ?? null) : (allItems.value[0]?.id ?? null)
  })
})

watch(activeId, id => { router.replace({ query: { ...route.query, id: id || undefined } }) })

let searchTimer = null
watch(search, () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { activeId.value = filtered.value[0]?.id ?? null }, 150) })

const _EXCLUDE_KEYS = /timestamp|date|time|_id$|^id$/i
const allAxes = computed(() => {
  const map = {}
  allItems.value.forEach(q => {
    Object.entries(q.meta ?? {}).forEach(([k, v]) => {
      if (v == null || _EXCLUDE_KEYS.test(k)) return
      const sv = String(v)
      if (!map[k]) map[k] = {}
      map[k][sv] = (map[k][sv] ?? 0) + 1
    })
  })
  return Object.fromEntries(Object.entries(map).filter(([, vals]) => { const n = Object.keys(vals).length; return n > 1 && n <= 50 }))
})

const filtered = computed(() =>
  allItems.value.filter(q => {
    for (const [axis, vals] of Object.entries(catFilter.value)) {
      if (!vals?.size) continue
      const qval = q.meta?.[axis]
      if (qval == null || !vals.has(String(qval))) return false
    }
    if (search.value) {
      const s = search.value.toLowerCase()
      if (!q.id.toLowerCase().includes(s) && !q.query.toLowerCase().includes(s) &&
          !q.gold_answers?.some(a => a.toLowerCase().includes(s))) return false
    }
    return true
  })
)

function toggleCat(axis, val) {
  const s = new Set(catFilter.value[axis])
  s.has(val) ? s.delete(val) : s.add(val)
  catFilter.value = { ...catFilter.value, [axis]: s }
}

const hasFilters = computed(() => Object.values(catFilter.value).some(v => v?.size))
const active     = computed(() => allItems.value.find(q => q.id === activeId.value) ?? null)

const goldDocsMap     = ref({})
const loadingGoldDocs = ref(false)

async function loadGoldDocs() {
  const q = active.value
  if (!q || goldDocsMap.value[q.id]) return
  loadingGoldDocs.value = true
  try {
    const docs = await Promise.all(q.gold_ids.map(id => fetchDocument(dataset.value, split.value, id).catch(() => null)))
    goldDocsMap.value = { ...goldDocsMap.value, [q.id]: docs.filter(Boolean) }
  } finally { loadingGoldDocs.value = false }
}

const activeGoldDocs = computed(() => active.value ? (goldDocsMap.value[active.value.id] ?? null) : null)
</script>

<template>
  <div class="flex flex-col h-screen overflow-hidden bg-background">
    <TopNav :crumbs="[
      { label: dataset, to: `/dataset/${encodeURIComponent(dataset)}` },
      { label: split, to: `/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}` },
      { label: 'Queries' },
    ]" />
    <div class="flex flex-1 overflow-hidden">
    <aside class="sidebar w-[320px] min-w-[240px] flex flex-col overflow-hidden">
      <div class="sidebar-section px-4 pt-4 pb-3 shrink-0">
        <div class="flex items-center gap-2 mb-3">
          <div>
            <p class="text-xs text-muted-foreground/60">Queries</p>
          </div>
        </div>
        <Input v-model="search" placeholder="Search queries, answers, or IDs…" class="mb-2 h-8 text-xs" />
        <p class="text-xs text-muted-foreground/60">{{ filtered.length.toLocaleString() }} / {{ allItems.length.toLocaleString() }} queries</p>
      </div>

      <div v-if="Object.keys(allAxes).length" class="sidebar-section overflow-y-auto max-h-56 shrink-0">
        <div v-for="(vals, axis) in allAxes" :key="axis" class="px-4 py-2.5 border-b border-border last:border-0">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-1.5">{{ axis.replace(/_/g,' ') }}</p>
          <table class="w-full text-xs">
            <tbody>
              <tr v-for="(count, val) in vals" :key="val" @click="toggleCat(axis, val)"
                  :class="catFilter[axis]?.has(val) ? 'cat-row-active' : 'cat-row'">
                <td class="py-0.5 pr-2 text-muted-foreground truncate max-w-[160px]">{{ val }}</td>
                <td class="py-0.5 text-right text-muted-foreground/50 w-10">{{ count }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="hasFilters" class="px-4 py-1.5">
          <button @click="catFilter = {}" class="text-xs text-muted-foreground hover:text-foreground transition-colors">✕ clear filters</button>
        </div>
      </div>

      <div class="flex-1 overflow-y-auto">
        <div v-if="loading" class="p-6 text-center text-muted-foreground text-sm animate-pulse">Loading…</div>
        <div v-else-if="error" class="p-4 text-destructive text-sm">{{ error }}</div>
        <template v-else>
          <p v-if="!filtered.length" class="p-4 text-xs text-muted-foreground">No queries match this filter.</p>
          <button v-for="q in filtered" :key="q.id" @click="activeId = q.id"
                  :class="q.id === activeId ? 'item-active' : 'hover:bg-secondary/30'"
                  class="w-full text-left px-4 py-3 border-b border-border/50 last:border-0 transition-colors">
            <p class="text-xs font-mono text-muted-foreground/50 truncate mb-0.5">{{ q.id }}</p>
            <p class="text-sm text-foreground line-clamp-2 leading-snug">{{ q.query.split('\n')[0].trim() }}</p>
            <div class="flex items-center gap-1 mt-0.5 flex-wrap">
              <p v-if="q.gold_answers?.length" class="text-xs text-cg truncate">✓ {{ q.gold_answers[0] }}</p>
              <Badge v-for="(v, k) in q.meta" :key="k" v-if="k in allAxes" variant="default" class="text-xs">{{ v }}</Badge>
            </div>
          </button>
        </template>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto bg-background p-6">
      <div v-if="!active" class="text-muted-foreground text-sm mt-20 text-center">Select a query</div>

      <div v-else class="max-w-3xl mx-auto space-y-5 pb-10">
        <div class="flex flex-wrap gap-2 pt-1">
          <span class="text-xs font-mono text-muted-foreground/50">{{ active.id }}</span>
          <Badge v-for="(v, k) in active.meta" :key="k" variant="default">
            <span class="opacity-60">{{ k.replace(/_/g,' ') }}:</span> {{ v }}
          </Badge>
        </div>

        <section>
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Query</p>
          <Card class="p-4 text-sm text-foreground leading-relaxed whitespace-pre-wrap">{{ active.query }}</Card>
        </section>

        <section>
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Gold answers</p>
          <div class="flex flex-wrap gap-2">
            <Badge v-for="a in active.gold_answers" :key="a" variant="success" class="text-sm px-3 py-1">{{ a }}</Badge>
          </div>
        </section>

        <section v-if="active.gold_ids?.length">
          <div class="flex items-center justify-between mb-2">
            <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50">
              Gold documents
              <span v-if="activeGoldDocs" class="normal-case tracking-normal font-normal text-muted-foreground ml-1">
                {{ activeGoldDocs.length }} doc{{ activeGoldDocs.length !== 1 ? 's' : '' }}
              </span>
            </p>
            <Button v-if="!activeGoldDocs" @click="loadGoldDocs" :disabled="loadingGoldDocs" variant="ghost" size="xs">
              {{ loadingGoldDocs ? 'Loading…' : 'Load documents' }}
            </Button>
          </div>

          <div v-if="activeGoldDocs" class="space-y-2">
            <details v-for="(doc, i) in activeGoldDocs" :key="doc.id" :open="i === 0" class="rounded-lg border border-border overflow-hidden">
              <summary class="doc-summary px-4 py-3 cursor-pointer flex items-center justify-between gap-3 select-none bg-card">
                <div class="flex items-center gap-2 min-w-0">
                  <span class="text-primary text-xs shrink-0">▶</span>
                  <span class="text-xs font-mono text-foreground truncate">{{ doc.id }}</span>
                </div>
                <div class="flex items-center gap-3 shrink-0 text-xs text-muted-foreground">
                  <span v-if="doc.timestamp">{{ doc.timestamp }}</span>
                  <span>{{ doc.content.length.toLocaleString() }} chars</span>
                </div>
              </summary>
              <div class="px-4 pb-4 pt-2 text-sm text-foreground leading-relaxed whitespace-pre-wrap border-t border-border max-h-96 overflow-y-auto bg-card">{{ doc.content }}</div>
            </details>
          </div>

          <div v-else class="flex flex-wrap gap-1">
            <Badge v-for="id in active.gold_ids" :key="id" variant="secondary" class="font-mono">{{ id }}</Badge>
          </div>
        </section>
      </div>
    </main>
    </div>
  </div>
</template>
