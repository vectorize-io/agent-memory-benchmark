<script setup>
import { computed } from 'vue'
import { cn } from '@/lib/utils.js'

const props = defineProps({
  variant: { type: String, default: 'default' },
  class: { type: String, default: '' },
})

const base = 'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium tracking-wide transition-colors'

const variants = {
  default:     'bg-primary/15 text-primary border border-primary/30',
  secondary:   'bg-secondary text-secondary-foreground border border-border',
  destructive: 'bg-destructive/15 text-destructive border border-destructive/30',
  outline:     'border border-border text-muted-foreground',
  success:     'bg-cg/10 text-cg border border-cg/25',
  warning:     'bg-ca/10 text-ca border border-ca/25',
  mcq:         'bg-ca/10 text-ca border border-ca/25',
  open:        'bg-violet-500/10 text-violet-400 border border-violet-500/25',
  local:       'bg-cg/10 text-cg border border-cg/25',
  cloud:       'bg-primary/10 text-primary border border-primary/25',
}

const cls = computed(() => cn(base, variants[props.variant] ?? variants.default, props.class))
</script>

<template>
  <span :class="cls"><slot /></span>
</template>
