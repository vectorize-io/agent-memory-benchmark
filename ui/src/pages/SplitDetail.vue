<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchSplitStats } from '../api.js'
import { fmtTokens } from '../utils.js'
import Card from '@/components/ui/card.vue'
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
const split   = computed(() => decodeURIComponent(route.params.split))

const stats   = ref(null)
const loading = ref(true)
const error   = ref(null)

onMounted(async () => {
  const s = await fetchSplitStats(dataset.value, split.value)
  if (s?.error) error.value = s.error
  else stats.value = s
  loading.value = false
})

const docCategories   = computed(() => Object.entries(stats.value?.categories ?? {}).filter(([, s]) => s.type === 'doc'))
const queryCategories = computed(() => Object.entries(stats.value?.categories ?? {}).filter(([, s]) => s.type === 'query'))

function corpusLink(type, category = '') {
  const base = `/dataset/${encodeURIComponent(dataset.value)}/${encodeURIComponent(split.value)}/${type}`
  return category ? `${base}?category=${encodeURIComponent(category)}` : base
}
</script>

<template>
  <div class="min-h-screen bg-background">
    <TopNav :crumbs="[
      { label: dataset, to: `/dataset/${encodeURIComponent(dataset)}` },
      { label: split },
    ]" />

    <div class="px-6 py-8">
      <div v-if="loading" class="text-center mt-24"><p class="text-muted-foreground text-sm animate-pulse">Loading…</p></div>
      <div v-else-if="error" class="text-destructive text-sm mt-10">{{ error }}</div>

      <div v-else-if="stats" class="space-y-8">

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card v-for="[val, label] in [
            [stats.queries?.toLocaleString(), 'Queries'],
            [stats.docs?.toLocaleString(), 'Documents'],
            [stats.total_tokens != null ? fmtTokens(stats.total_tokens) : '—', 'Total tokens'],
            [stats.avg_tokens_per_doc?.toLocaleString() ?? '—', 'Avg tok/doc'],
          ]" :key="label" class="p-5 text-center">
            <p class="font-display text-2xl font-bold text-foreground tracking-tight">{{ val }}</p>
            <p class="text-xs text-muted-foreground mt-1.5">{{ label }}</p>
          </Card>
        </div>

        <Card class="p-5 flex items-center justify-between">
          <div>
            <p class="font-display font-semibold text-foreground text-sm">All data</p>
            <p class="text-xs text-muted-foreground mt-1">
              {{ stats.queries?.toLocaleString() }} queries · {{ stats.docs?.toLocaleString() }} docs
            </p>
          </div>
          <div class="flex gap-5 text-sm">
            <router-link :to="corpusLink('queries')"   class="text-primary hover:text-primary/80 font-medium text-sm transition-colors">Queries →</router-link>
            <router-link :to="corpusLink('documents')" class="text-primary hover:text-primary/80 font-medium text-sm transition-colors">Docs →</router-link>
          </div>
        </Card>

        <div v-if="docCategories.length">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-3">Document categories</p>
          <Card class="overflow-hidden">
            <UiTable>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead :right="true">Queries</TableHead>
                  <TableHead :right="true">Docs</TableHead>
                  <TableHead :right="true">Tokens</TableHead>
                  <TableHead :right="true">Tok/doc</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow v-for="[cat, s] in docCategories" :key="cat">
                  <TableCell :primary="true">{{ cat }}</TableCell>
                  <TableCell :right="true">{{ s.queries?.toLocaleString() }}</TableCell>
                  <TableCell :right="true">{{ s.docs?.toLocaleString() }}</TableCell>
                  <TableCell :right="true">{{ fmtTokens(s.total_tokens) }}</TableCell>
                  <TableCell :right="true" class="text-muted-foreground/60 text-xs">{{ s.avg_tokens_per_doc?.toLocaleString() }}</TableCell>
                  <TableCell :right="true">
                    <div class="flex gap-4 justify-end text-xs">
                      <router-link :to="corpusLink('queries', cat)"   class="text-primary hover:text-primary/80 transition-colors">Queries →</router-link>
                      <router-link :to="corpusLink('documents', cat)" class="text-primary hover:text-primary/80 transition-colors">Docs →</router-link>
                    </div>
                  </TableCell>
                </TableRow>
              </TableBody>
            </UiTable>
          </Card>
        </div>

        <div v-if="queryCategories.length">
          <p class="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-3">Query categories</p>
          <Card class="overflow-hidden">
            <UiTable>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead :right="true">Queries</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow v-for="[cat, s] in queryCategories" :key="cat">
                  <TableCell :primary="true">{{ cat }}</TableCell>
                  <TableCell :right="true">{{ s.queries?.toLocaleString() }}</TableCell>
                  <TableCell :right="true">
                    <router-link :to="corpusLink('queries', cat)" class="text-primary hover:text-primary/80 text-xs transition-colors">Queries →</router-link>
                  </TableCell>
                </TableRow>
              </TableBody>
            </UiTable>
          </Card>
        </div>

      </div>
    </div>
  </div>
</template>
