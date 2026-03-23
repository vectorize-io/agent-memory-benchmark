<script setup>
import { cn } from '@/lib/utils.js'
defineProps({ class: { type: String, default: '' } })
</script>

<template>
  <div :class="cn('rounded-lg border border-border bg-card text-card-foreground', $props.class)">
    <slot />
  </div>
</template>
