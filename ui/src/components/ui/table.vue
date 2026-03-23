<script setup>
import { cn } from '@/lib/utils.js'
</script>

<template>
  <div class="w-full overflow-auto">
    <table class="w-full caption-bottom text-sm border-collapse">
      <slot />
    </table>
  </div>
</template>
