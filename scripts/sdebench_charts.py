#!/usr/bin/env python3
"""Generate comparison charts for the sdebench coding results (vanilla vs memory, both agents).

Reusable across reruns: each "arm" is matched by a run-name GLOB, so an n=3 rerun that produces
nz-oc-none-1/-2/-3 is averaged automatically (with error bars). Point --outputs at the results dir
and --out at where the PNGs go (defaults to the doc's charts/ folder).

Usage:
  uv run --with matplotlib python scripts/sdebench_charts.py \
      [--outputs outputs/sdebench] [--out ~/Documents/charts]
"""
import argparse, glob, gzip, json, os, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 4 arms: (agent, arm, run-name glob). Globs so n>1 reruns (nz-oc-none-1/-2/-3) auto-average.
ARMS = [
    ("OpenCode", "vanilla", "nz-oc-none*"),
    ("OpenCode", "memory",  "nz-oc-hs*"),
    ("Claude",   "vanilla", "nz-cc-none*"),
    ("Claude",   "memory",  "nz-cc-hs*"),
]
AGENTS = ["OpenCode", "Claude"]
AGENT_LABELS = {"OpenCode": "OpenCode\nGemini 3.5 Flash", "Claude": "Claude Code\nClaude Sonnet"}
ARM_LABELS = {"vanilla": "Vanilla", "memory": "Hindsight"}
VANILLA_C, MEMORY_C = "#9aa0a6", "#0080b0"   # muted gray vs Hindsight brand blue
TOK_COLORS = {"Input": "#3b82f6", "Cached": "#a5b4fc", "Output": "#f59e0b"}


def load_run(path):
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    a = dict(interventions=0, cost=0, turns=0, wall=0, tok_input=0, tok_cached=0, tok_output=0,
             solved=0, tasks=len(d["results"]))
    for r in d["results"]:
        m = r["meta"]; t = m.get("tokens") or {}
        a["interventions"] += m.get("interventions") or 0
        a["cost"]          += m.get("cost_usd") or 0
        a["turns"]         += m.get("turns") or 0
        a["wall"]          += m.get("wall_s") or 0
        a["tok_input"]     += (t.get("input") or 0) + (t.get("cache_write") or 0)  # fresh prompt tokens
        a["tok_cached"]    += t.get("cache_read") or 0                              # re-read from cache (cheap)
        a["tok_output"]    += (t.get("output") or 0) + (t.get("reasoning") or 0)
        a["solved"]        += 1 if m.get("solved") else 0
    return a


def arm_stats(outputs, run_glob):
    paths = sorted(glob.glob(f"{outputs}/{run_glob}/coding/boltons.json.gz")
                   + glob.glob(f"{outputs}/{run_glob}/coding/boltons.json"))
    # de-dup (prefer .gz) and skip a raw .json if its .gz twin exists
    seen, runs = set(), []
    for p in paths:
        key = p.replace(".json.gz", "").replace(".json", "")
        if key in seen:
            continue
        seen.add(key); runs.append(load_run(p))
    if not runs:
        return None
    # Report per-task averages (divide summed metrics by task count) — more interpretable than run totals.
    nt = runs[0]["tasks"]
    keys = [k for k in runs[0] if k != "tasks"]
    mean = {k: st.mean(r[k] for r in runs) / nt for k in keys}
    std  = {k: (st.stdev([r[k] for r in runs]) / nt if len(runs) > 1 else 0.0) for k in keys}
    mean["tasks"] = nt; mean["_n"] = len(runs)
    return mean, std


def _pct(v, m):
    return f"{(m - v) / v * 100:+.0f}%" if v else ""


def grouped_bar(data, metric, title, ylabel, fmt, out, note=""):
    """One grouped bar per agent: [vanilla, memory]. Value labels + % change on the memory bar."""
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x = range(len(AGENTS)); w = 0.36
    for i, (arm, color, off) in enumerate((("vanilla", VANILLA_C, -w/2), ("memory", MEMORY_C, w/2))):
        vals = [data[(ag, arm)][0][metric] for ag in AGENTS]
        errs = [data[(ag, arm)][1][metric] for ag in AGENTS]
        bars = ax.bar([xi + off for xi in x], vals, w, label=ARM_LABELS[arm], color=color,
                      yerr=errs if any(errs) else None, capsize=4, error_kw=dict(ecolor="#555", lw=1))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), fmt(v), ha="center", va="bottom", fontsize=9)
    # % change annotation above each agent group
    for xi, ag in zip(x, AGENTS):
        v = data[(ag, "vanilla")][0][metric]; m = data[(ag, "memory")][0][metric]
        ax.text(xi, max(v, m) * 1.14, _pct(v, m), ha="center", fontsize=10, fontweight="bold", color=MEMORY_C)
    ax.set_xticks(list(x)); ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.margins(y=0.20); ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        ax.text(0.0, -0.16, note, transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  wrote", out)


def tokens_chart(data, out):
    """Distinguish Input vs Cached vs Output — grouped bars per run, log y (spans M to k)."""
    short = {"OpenCode": "OpenCode·Gemini", "Claude": "Claude·Sonnet"}
    runs = [(f"{short[ag]}\n{ARM_LABELS[arm]}", (ag, arm)) for ag in AGENTS for arm in ("vanilla", "memory")]
    cats = ["Input", "Cached", "Output"]; keys = {"Input": "tok_input", "Cached": "tok_cached", "Output": "tok_output"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(runs)); w = 0.26
    for j, cat in enumerate(cats):
        vals = [data[k][0][keys[cat]] for _, k in runs]
        bars = ax.bar([xi + (j - 1) * w for xi in x], vals, w, label=cat, color=TOK_COLORS[cat])
        for b, v in zip(bars, vals):
            lab = f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else str(int(v)))
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), lab, ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylabel("tokens / task (log scale)")
    ax.set_title("Tokens per task: input vs cached vs output", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([lbl for lbl, _ in runs], fontsize=9)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs/sdebench")
    ap.add_argument("--out", default=os.path.expanduser("~/Documents/charts"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = {}
    for ag, arm, glb in ARMS:
        s = arm_stats(args.outputs, glb)
        if s is None:
            raise SystemExit(f"no runs matched glob {glb!r} under {args.outputs}")
        data[(ag, arm)] = s
    n = data[("OpenCode", "vanilla")][0]["_n"]
    nt = data[("OpenCode", "vanilla")][0]["tasks"]
    note = f"average per task (over {nt} tasks)" + (f", mean of n={n} runs (error bars = std)" if n > 1 else ", n=1")
    print(f"charts -> {out}  ({note})")

    grouped_bar(data, "interventions", "Interventions per task", "interventions / task",
                lambda v: f"{v:.2f}", out / "interventions.png", note)
    grouped_bar(data, "cost", "Cost per task", "USD / task", lambda v: f"${v:.2f}", out / "cost.png", note)
    grouped_bar(data, "turns", "Tool-turns per task", "turns / task", lambda v: f"{v:.0f}",
                out / "turns.png", note)
    grouped_bar(data, "wall", "Wall-clock per task", "seconds / task", lambda v: f"{v:.0f}s",
                out / "wall.png", note)
    tokens_chart(data, out / "tokens.png")


if __name__ == "__main__":
    main()
