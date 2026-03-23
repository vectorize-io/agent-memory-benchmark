<script setup>
import { computed } from 'vue'

const props = defineProps({
  rows:       { type: Array,    required: true },   // [{ label, logo, value }]
  max:        { type: Number,   default: null },    // null = relative to row max; 1 = absolute (accuracy)
  format:     { type: Function, required: true },   // value => display string
  variant:    { type: String,   default: 'accuracy' }, // 'accuracy' | 'recall' | 'tokens'
  labelWidth: { type: String,   default: 'w-24' },
})

const barClass = computed(() => ({
  accuracy: 'chart-bar-a',
  recall:   'chart-bar-r',
  tokens:   'chart-bar-t',
}[props.variant] ?? 'chart-bar-a'))

const effectiveMax = computed(() =>
  props.max ?? (props.rows.reduce((m, r) => Math.max(m, r.value), 0) || 1)
)
</script>

<template>
  <div class="space-y-2">
    <div v-for="row in rows" :key="row.label" class="flex items-center gap-2 min-w-0">
      <div :class="[labelWidth, 'shrink-0 flex items-center gap-1.5 justify-end min-w-0']">
        <img v-if="row.logo" :src="row.logo"
             class="w-3 h-3 rounded object-contain shrink-0"
             @error="$event.target.style.display='none'" />
        <span class="text-muted-foreground text-xs truncate" :title="row.label">{{ row.label }}</span>
      </div>
      <div class="chart-track flex-1">
        <div :class="barClass" :style="{ width: (row.value / effectiveMax * 100) + '%' }"></div>
        <div class="chart-val">{{ format(row.value) }}</div>
      </div>
    </div>
  </div>
</template>
