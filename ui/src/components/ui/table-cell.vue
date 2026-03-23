<script setup>
import { cn } from '@/lib/utils.js'
defineProps({ class: { type: String, default: '' }, right: Boolean, primary: Boolean })
</script>

<template>
  <td :class="cn(
    'px-4 py-3 text-muted-foreground',
    right && 'text-right',
    primary && 'text-foreground font-medium',
    $props.class
  )">
    <slot />
  </td>
</template>
