<script setup>
import { computed, ref } from 'vue'

// groups: [{ label, agent, model, runs: {vanilla, hindsight}, metrics: {key: {vanilla, hindsight}} }]
const props = defineProps({ groups: { type: Array, required: true } })

const X_AXES = {
  cost_usd: { title: 'Cost / task (USD, log scale)', fmt: v => '$' + (v < 0.1 ? v.toFixed(3) : v.toFixed(2)), ticks: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5] },
  wall_s:   { title: 'Wall time / task (seconds, log scale)', fmt: v => Math.round(v) + 's', ticks: [5, 10, 20, 50, 100, 200, 500, 1000] },
}
const xKey = ref('cost_usd')
const Y_KEY = 'interventions'

const W = 1200, H = 440, L = 64, R = 24, T = 16, B = 48
const PW = W - L - R, PH = H - T - B

// Harness logos, copied from the Hindsight control plane (public/img/harness). Codex's mark is black,
// so it is inverted on this dark theme — the control plane flags it the same way (invertOnDark).
const AGENTS = {
  'claude-code': { name: 'Claude Code', logo: '/img/harness/claude-code.png' },
  codex:         { name: 'Codex CLI',   logo: '/img/harness/codex.svg', invert: true },
  opencode:      { name: 'opencode',    logo: '/img/harness/opencode.png' },
}
// Hindsight's mark (hindsight-docs/static/img/favicon.png, 186×139) badges every point run WITH memory.
const HS_MARK = '/img/hindsight.png'
const agentOf = agent => AGENTS[agent] ?? { name: agent, logo: null }
const letter = agent => (agent || '?')[0].toUpperCase()
const shortModel = m => (m ?? '?').replace(/^google\//, '').replace(/^claude-/, '')

const values = computed(() => props.groups.flatMap(g =>
  ['vanilla', 'hindsight'].map(arm => ({ x: g.metrics[xKey.value][arm], y: g.metrics[Y_KEY][arm] }))))

// x: log scale padded around the data; y: corrections with 0 at the TOP — fewer is better, so the
// attractive corner is top-left, as in an accuracy-vs-cost plot.
const xDomain = computed(() => {
  const xs = values.value.map(v => v.x)
  return [Math.min(...xs) / 1.4, Math.max(...xs) * 1.4]
})
const yMax = computed(() => Math.ceil(Math.max(...values.value.map(v => v.y)) * 1.08 * 4) / 4 || 1)
const sx = v => {
  const [a, b] = xDomain.value
  return L + (Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a)) * PW
}
const sy = v => T + v / yMax.value * PH
const xTicks = computed(() => X_AXES[xKey.value].ticks.filter(t => t >= xDomain.value[0] && t <= xDomain.value[1]))
const yTicks = computed(() => Array.from({ length: Math.round(yMax.value / 0.25) + 1 }, (_, i) => i * 0.25))

const points = computed(() => props.groups.map(g => {
  const at = arm => ({ x: sx(g.metrics[xKey.value][arm]), y: sy(g.metrics[Y_KEY][arm]), v: g.metrics[Y_KEY][arm] })
  return { ...g, van: at('vanilla'), hs: at('hindsight') }
}))
const agents = computed(() => [...new Set(props.groups.map(g => g.agent))])

// Label placement: each point tries right, left, above, below and takes the first spot that hits no
// circle and no label already placed — at this density a fixed "always right" label ran into its
// neighbours (Sonnet 5's vanilla point sits right under opencode's hindsight one).
const LOGO = 32, R_PT = 18, CHAR_W = 7.4, LINE_H = 16
const labels = computed(() => {
  const dots = points.value.flatMap(p => [p.van, p.hs])
  const boxes = []
  const hits = b =>
    b.x0 < L || b.x1 > W - R || b.y0 < T || b.y1 > H - B ||
    dots.some(d => d.x + R_PT > b.x0 && d.x - R_PT < b.x1 && d.y + R_PT > b.y0 && d.y - R_PT < b.y1) ||
    boxes.some(o => o.x0 < b.x1 && o.x1 > b.x0 && o.y0 < b.y1 && o.y1 > b.y0)
  const out = []
  for (const p of points.value) {
    for (const [arm, pt] of [['van', p.van], ['hs', p.hs]]) {
      const text = shortModel(p.model), val = pt.v.toFixed(2)
      const w = (text.length + val.length + 1) * CHAR_W
      const tries = [
        { x: pt.x + R_PT + 6, y: pt.y + 4, anchor: 'start', x0: pt.x + R_PT + 4, y0: pt.y - LINE_H / 2 },
        { x: pt.x - R_PT - 6, y: pt.y + 4, anchor: 'end',   x0: pt.x - R_PT - 8 - w, y0: pt.y - LINE_H / 2 },
        { x: pt.x, y: pt.y - R_PT - 7, anchor: 'middle', x0: pt.x - w / 2, y0: pt.y - R_PT - 5 - LINE_H },
        { x: pt.x, y: pt.y + R_PT + 16, anchor: 'middle', x0: pt.x - w / 2, y0: pt.y + R_PT + 3 },
      ].map(c => ({ ...c, x1: c.x0 + w, y1: c.y0 + LINE_H }))
      const pick = tries.find(c => !hits(c)) ?? tries[0]
      boxes.push(pick)
      out.push({ key: p.label + arm, arm, text, val, ...pick })
    }
  }
  return out
})
</script>

<template>
  <div>
    <p class="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground/85 mb-1">
      Corrections vs. {{ xKey === 'cost_usd' ? 'cost' : 'time' }}
    </p>
    <p class="text-muted-foreground/70 text-sm mb-4">per task, mean over runs · arrow: no memory → Hindsight</p>
    <div class="flex flex-wrap items-center justify-between gap-3 mb-3 text-xs">
      <div class="flex items-center gap-4">
        <span class="flex items-center gap-1.5"><span class="swatch swatch-van"></span>No memory (grey)</span>
        <span class="flex items-center gap-1.5"><img :src="HS_MARK" class="h-3.5 w-auto" alt="" />Hindsight</span>
      </div>
      <div class="flex items-center gap-3 text-muted-foreground">
        <span v-for="a in agents" :key="a" class="flex items-center gap-1.5">
          <img v-if="agentOf(a).logo" :src="agentOf(a).logo" class="w-5 h-5 object-contain" :class="{ invert: agentOf(a).invert }" alt="" />
          {{ agentOf(a).name }}
        </span>
      </div>
      <div class="flex rounded-md border border-border overflow-hidden">
        <button v-for="(ax, key) in X_AXES" :key="key" @click="xKey = key"
                class="px-2.5 py-1 transition-colors"
                :class="xKey === key ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'">
          {{ key === 'cost_usd' ? 'Cost' : 'Time' }}
        </button>
      </div>
    </div>

    <svg :viewBox="`0 0 ${W} ${H}`" class="w-full h-auto" role="img"
         :aria-label="`Corrections per task versus ${X_AXES[xKey].title}`">
      <defs>
        <marker id="scatter-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="arrow" />
        </marker>
      </defs>

      <!-- attractive quadrant: cheaper/faster AND fewer corrections -->
      <rect :x="L" :y="T" :width="PW / 2" :height="PH / 2" class="quadrant" />
      <text :x="L + PW / 2 - 8" :y="T + 16" text-anchor="end" class="quadrant-label">Most attractive quadrant</text>

      <g v-for="t in yTicks" :key="'y' + t">
        <line :x1="L" :x2="W - R" :y1="sy(t)" :y2="sy(t)" class="grid" />
        <text :x="L - 10" :y="sy(t) + 4" text-anchor="end" class="tick">{{ t.toFixed(2) }}</text>
      </g>
      <g v-for="t in xTicks" :key="'x' + t">
        <line :x1="sx(t)" :x2="sx(t)" :y1="T" :y2="H - B" class="grid" />
        <text :x="sx(t)" :y="H - B + 18" text-anchor="middle" class="tick">{{ X_AXES[xKey].fmt(t) }}</text>
      </g>
      <text :x="W - R" :y="H - 6" text-anchor="end" class="axis">{{ X_AXES[xKey].title }} →</text>
      <text :x="14" :y="T + PH / 2" text-anchor="middle" class="axis"
            :transform="`rotate(-90 14 ${T + PH / 2})`">Corrections / task — fewer ↑</text>

      <!-- vanilla → hindsight, per agent+model -->
      <line v-for="p in points" :key="'l' + p.label"
            :x1="p.van.x" :y1="p.van.y" :x2="p.hs.x" :y2="p.hs.y"
            class="link" marker-end="url(#scatter-arrow)" />

      <g v-for="p in points" :key="'p' + p.label">
        <g v-for="[arm, pt] in [['van', p.van], ['hs', p.hs]]" :key="arm">
          <title>{{ p.label }} · {{ arm === 'hs' ? 'Hindsight' : 'no memory' }} (×{{ arm === 'hs' ? p.runs.hindsight : p.runs.vanilla }} runs)
{{ pt.v.toFixed(2) }} corrections · {{ X_AXES[xKey].fmt(p.metrics[xKey][arm === 'hs' ? 'hindsight' : 'vanilla']) }}</title>
          <!-- the logo alone marks the point (a ring around it read as clutter); no memory is the same
               logo in grey, Hindsight the logo in colour with the Hindsight badge -->
          <circle :cx="pt.x" :cy="pt.y" :r="R_PT" class="pt-hit" />
          <image v-if="agentOf(p.agent).logo" :href="agentOf(p.agent).logo"
                 :x="pt.x - LOGO / 2" :y="pt.y - LOGO / 2" :width="LOGO" :height="LOGO"
                 preserveAspectRatio="xMidYMid meet" class="pt-logo"
                 :class="{ 'logo-invert': agentOf(p.agent).invert, 'pt-faded': arm === 'van' }" />
          <text v-else :x="pt.x" :y="pt.y + 5" text-anchor="middle" class="pt-letter"
                :class="{ 'pt-faded': arm === 'van' }">{{ letter(p.agent) }}</text>
          <image v-if="arm === 'hs'" :href="HS_MARK" :x="pt.x + LOGO / 2 - 9" :y="pt.y - LOGO / 2 - 9"
                 width="20" height="15" preserveAspectRatio="xMidYMid meet" class="pt-badge" />
        </g>
      </g>
      <text v-for="lb in labels" :key="lb.key" :x="lb.x" :y="lb.y" :text-anchor="lb.anchor" class="pt-label">
        {{ lb.text }} <tspan :class="lb.arm === 'hs' ? 'val-hs' : 'val-van'">{{ lb.val }}</tspan>
      </text>
    </svg>
  </div>
</template>

<style scoped>
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; background: var(--foreground); }
.swatch-van { background: var(--muted-foreground); }
.quadrant { fill: oklch(0.72 0.15 150 / 0.08); }
.quadrant-label { fill: oklch(0.72 0.15 150); font-size: 12px; font-family: ui-monospace, monospace; }
.grid { stroke: var(--border); stroke-opacity: 0.5; stroke-width: 1; }
.tick { fill: var(--muted-foreground); font-size: 12px; font-family: ui-monospace, monospace; }
.axis { fill: var(--muted-foreground); font-size: 12px; }
.link { stroke: var(--muted-foreground); stroke-width: 1.5; stroke-dasharray: 5 4; opacity: 0.8; }
.arrow { fill: var(--muted-foreground); }
.pt-hit { fill: transparent; }
.pt-logo { pointer-events: none; }
.pt-logo:not(.pt-faded) { filter: drop-shadow(0 0 6px color-mix(in oklch, var(--primary) 70%, transparent)); }
.pt-logo.logo-invert:not(.pt-faded) { filter: invert(1) drop-shadow(0 0 6px color-mix(in oklch, var(--primary) 70%, transparent)); }
.pt-badge { pointer-events: none; filter: drop-shadow(0 0 2px var(--card)); }
.pt-faded { filter: grayscale(1); }
.pt-faded.logo-invert { filter: invert(1) grayscale(1); }
.pt-letter.pt-faded { fill: var(--muted-foreground); }
.logo-invert { filter: invert(1); }
.pt-letter { fill: var(--foreground); font-size: 16px; font-weight: 700; font-family: ui-monospace, monospace; pointer-events: none; }
.pt-label { fill: var(--foreground); font-size: 12.5px; font-weight: 500;
  /* halo in the card colour: the dashed arrows pass behind labels, not through them */
  paint-order: stroke; stroke: var(--card); stroke-width: 5px; stroke-linejoin: round; }
.val-hs  { fill: var(--primary); font-family: ui-monospace, monospace; }
.val-van { fill: var(--muted-foreground); font-family: ui-monospace, monospace; }
</style>
