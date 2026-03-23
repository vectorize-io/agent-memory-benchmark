<script setup>
import { ref, watch, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchDocuments, fetchDocument, fetchSplitStats } from '../api.js'
import Input from '@/components/ui/input.vue'
import Card from '@/components/ui/card.vue'
import TopNav from '@/components/ui/top-nav.vue'

const PAGE = 20
const route   = useRoute()
const router  = useRouter()
const dataset = computed(() => decodeURIComponent(route.params.name))
const split   = computed(() => decodeURIComponent(route.params.split))

const items       = ref([])
const total       = ref(0)
const offset      = ref(0)
const search      = ref('')
const category    = ref(route.query.category ?? '')
const categories  = ref([])
const loading     = ref(true)
const loadingMore = ref(false)
const error       = ref(null)
const activeId    = ref(null)
const activeDoc   = ref(null)
const loadingDoc  = ref(false)

fetchSplitStats(dataset.value, split.value).then(s => {
  if (s?.categories) categories.value = Object.keys(s.categories)
})

async function load(reset = true) {
  if (reset) { offset.value = 0; items.value = []; activeId.value = null }
  reset ? (loading.value = true) : (loadingMore.value = true)
  error.value = null
  try {
    const res = await fetchDocuments(dataset.value, split.value, {
      search: search.value, category: category.value, limit: PAGE, offset: offset.value,
    })
    total.value  = res.total
    items.value  = reset ? res.items : [...items.value, ...res.items]
    offset.value = items.value.length
  } catch (e) { error.value = e.message }
  finally { loading.value = false; loadingMore.value = false }
}

onMounted(() => load())
watch(category, val => { router.replace({ query: { ...route.query, category: val || undefined } }); load() })
let searchTimer = null
watch(search, () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => load(), 300) })

const active  = computed(() => activeDoc.value ?? items.value.find(d => d.id === activeId.value) ?? null)
const hasMore = computed(() => items.value.length < total.value)

watch(activeId, async id => {
  activeDoc.value = null
  if (!id) return
  loadingDoc.value = true
  try { activeDoc.value = await fetchDocument(dataset.value, split.value, id) }
  finally { loadingDoc.value = false }
})
</script>

<template>
  <div class="flex flex-col h-screen overflow-hidden bg-background">
    <TopNav :crumbs="[
      { label: dataset, to: `/dataset/${encodeURIComponent(dataset)}` },
      { label: split, to: `/dataset/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}` },
      { label: 'Documents' },
    ]" />
    <div class="flex flex-1 overflow-hidden">
    <aside class="sidebar w-[320px] min-w-[240px] flex flex-col overflow-hidden">
      <div class="sidebar-section px-4 pt-4 pb-3 shrink-0 space-y-2.5">
        <div class="flex items-center gap-2">
          <div>
            <p class="text-xs text-muted-foreground/60">Documents</p>
          </div>
        </div>

        <div v-if="categories.length" class="flex flex-wrap gap-1">
          <button v-for="cat in ['', ...categories]" :key="cat" @click="category = cat"
                  :class="category === cat
                    ? 'bg-primary/15 text-primary border-primary/30'
                    : 'bg-secondary/40 text-muted-foreground border-border hover:bg-secondary hover:text-foreground'"
                  class="px-2.5 py-0.5 rounded-full text-xs font-medium border transition-colors">
            {{ cat || 'ALL' }}
          </button>
        </div>

        <Input v-model="search" placeholder="Search document content…" class="h-8 text-xs" />
        <p class="text-xs text-muted-foreground/60">{{ total.toLocaleString() }} documents</p>
      </div>

      <div class="flex-1 overflow-y-auto">
        <div v-if="loading" class="p-6 text-center text-muted-foreground text-sm animate-pulse">Loading…</div>
        <div v-else-if="error" class="p-4 text-destructive text-sm">{{ error }}</div>
        <template v-else>
          <button v-for="doc in items" :key="doc.id" @click="activeId = doc.id"
                  :class="doc.id === activeId ? 'item-active' : 'hover:bg-secondary/30'"
                  class="w-full text-left px-4 py-3 border-b border-border/50 last:border-0 transition-colors">
            <p class="text-xs font-mono text-muted-foreground/50 truncate mb-0.5">{{ doc.id }}</p>
            <p v-if="doc.timestamp" class="text-xs text-muted-foreground/50 mb-0.5">{{ doc.timestamp }}</p>
            <p class="text-sm text-foreground line-clamp-2 leading-snug">{{ doc.content.slice(0, 120) }}</p>
          </button>
          <div v-if="hasMore" class="p-3 text-center">
            <button @click="load(false)" :disabled="loadingMore"
                    class="text-xs text-primary hover:text-primary/80 disabled:opacity-40 transition-colors">
              {{ loadingMore ? 'Loading…' : 'Load more' }}
            </button>
          </div>
        </template>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto bg-background p-6">
      <div v-if="!active" class="text-muted-foreground text-sm mt-20 text-center">Select a document</div>

      <div v-else class="max-w-3xl mx-auto space-y-4 pb-10">
        <div class="flex items-baseline justify-between pt-1">
          <span class="text-xs font-mono text-muted-foreground/50">{{ active.id }}</span>
          <div class="flex gap-3 text-xs text-muted-foreground/50">
            <span v-if="active.user_id">user: {{ active.user_id }}</span>
            <span v-if="active.timestamp">{{ active.timestamp }}</span>
          </div>
        </div>

        <section>
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/50 mb-2">Content</p>
          <div v-if="loadingDoc" class="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground animate-pulse">Loading…</div>
          <div v-else class="code-block font-mono text-sm leading-relaxed whitespace-pre-wrap">{{ active.content }}</div>
        </section>
      </div>
    </main>
    </div>
  </div>
</template>
