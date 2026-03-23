<script setup>
import { cn } from '@/lib/utils.js'
defineProps({ class: { type: String, default: '' }, clickable: Boolean })
</script>

<template>
  <tr :class="cn('transition-colors', clickable && 'cursor-pointer hover:bg-secondary/50', $props.class)">
    <slot />
  </tr>
</template>
