<script setup>
import { cn } from '@/lib/utils.js'
defineProps({ class: { type: String, default: '' }, modelValue: String, placeholder: String, type: { type: String, default: 'text' } })
defineEmits(['update:modelValue'])
</script>

<template>
  <input
    v-bind="$attrs"
    :type="type"
    :value="modelValue"
    :placeholder="placeholder"
    @input="$emit('update:modelValue', $event.target.value)"
    :class="cn(
      'flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
      $props.class
    )"
  />
</template>
