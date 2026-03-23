<script setup>
import { cn } from '@/lib/utils.js'
defineProps({ class: { type: String, default: '' }, sortable: Boolean, right: Boolean })
</script>

<template>
  <th :class="cn(
    'px-4 py-3 text-left font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground/70 whitespace-nowrap select-none',
    sortable && 'cursor-pointer hover:text-muted-foreground',
    right && 'text-right',
    $props.class
  )">
    <slot />
  </th>
</template>
