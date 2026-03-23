<script setup>
import { useRouter } from 'vue-router'

// crumbs: [{ label, to? }]  — each renders as "/ label", clickable if `to` is set
defineProps({ crumbs: { type: Array, default: () => [] } })

const router = useRouter()
</script>

<template>
  <nav class="topnav shrink-0">
    <div class="px-6 h-14 flex items-center gap-3 min-w-0">
      <button @click="router.push('/')"
              class="font-display font-semibold text-foreground tracking-tight hover:text-primary transition-colors text-sm shrink-0">
        Agent Memory Benchmark
      </button>
      <template v-for="(crumb, i) in crumbs" :key="i">
        <span class="text-muted-foreground/40 shrink-0">/</span>
        <button v-if="crumb.to" @click="router.push(crumb.to)"
                class="text-muted-foreground hover:text-foreground text-sm transition-colors truncate">
          {{ crumb.label }}
        </button>
        <span v-else class="font-semibold text-foreground text-sm truncate">{{ crumb.label }}</span>
      </template>
      <div class="ml-auto flex items-center gap-4 shrink-0">
        <slot name="right" />
        <a href="https://github.com/vectorize-io/agent-memory-benchmark" target="_blank" rel="noopener"
           class="text-xs text-muted-foreground hover:text-foreground transition-colors">GitHub ↗</a>
      </div>
    </div>
  </nav>
</template>
