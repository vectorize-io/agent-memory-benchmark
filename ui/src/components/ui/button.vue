<script setup>
import { computed } from 'vue'
import { cn } from '@/lib/utils.js'

const props = defineProps({
  variant: { type: String, default: 'default' },
  size:    { type: String, default: 'default' },
  class:   { type: String, default: '' },
  disabled: Boolean,
})

const base = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50'

const variants = {
  default:  'bg-primary text-primary-foreground hover:bg-primary/90',
  ghost:    'text-muted-foreground hover:bg-secondary hover:text-foreground',
  link:     'text-primary underline-offset-4 hover:underline',
  outline:  'border border-border bg-transparent hover:bg-secondary text-foreground',
}

const sizes = {
  default: 'h-9 px-4 py-2 text-sm',
  sm:      'h-7 px-3 text-xs',
  xs:      'h-6 px-2 text-xs',
  icon:    'h-9 w-9',
}

const cls = computed(() =>
  cn(base, variants[props.variant] ?? variants.default, sizes[props.size] ?? sizes.default, props.class)
)
</script>

<template>
  <button :class="cls" :disabled="disabled"><slot /></button>
</template>
