"""Export sdebench runs into the AMB UI's agent-trace view.

Reads the harness's trace.json files, groups them by (model, history) into runs, and writes
outputs/sdebench/<run_name>/agent/all.json in the EvalSummary format the UI expects
(view="agent"). Each task-run becomes a QueryResult whose trajectory is the FULL multi-round
conversation (bug report -> agent -> feedback -> agent ...), so the whole task is browsable.

Usage:
    uv run python sdebench/harness/ui_export.py                 # all runs under /tmp/sdebench/run
    uv run python sdebench/harness/ui_export.py --glob '*ttl*'  # filter
    uv run omb view    # then open dataset 'sdebench'
"""
import argparse, json, re, glob, sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
SDEBENCH = HARNESS.parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "sdebench"
RUN_DIR = Path("/tmp/sdebench/run")
sys.path.insert(0, str(HARNESS))
from run import capture_git_history   # reuse to backfill git history for any run


def repo_of(t):
    return t["task_id"].rsplit("-regression", 1)[0]


PRICES = {  # $ per 1M tokens (gemini-3.5-flash, Jun 2026)
    "google/gemini-3.5-flash": {"input": 1.50, "cache_read": 0.15, "cache_write": 1.50, "output": 9.00},
}


def compute_cost(model, tok):
    p = PRICES.get(model)
    if not p:
        return 0.0
    return round((tok.get("input", 0) * p["input"] + tok.get("cache_read", 0) * p["cache_read"]
                  + tok.get("cache_write", 0) * p["cache_write"]
                  + (tok.get("output", 0) + tok.get("reasoning", 0)) * p["output"]) / 1_000_000, 4)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9.+_-]", "_", s)


def flatten_trace(trace):
    """Multi-round trace -> one trajectory with feedback markers + the submitted patch per round."""
    steps = []
    for rnd in trace:
        if rnd["role"] != "initial":
            steps.append({"k": "say",
                          "text": f"🔁 {rnd['role']} — feedback sent to the agent:\n{rnd['prompt'][:1400]}"})
        steps.extend(rnd.get("trajectory", []))
        if rnd.get("patch") is not None:   # the patch submitted this round + its grade outcome
            steps.append({"k": "patch", "round": rnd["role"], "passed": rnd.get("grade_passed"),
                          "pytest": rnd.get("grade_pytest"), "patch": rnd["patch"]})
    return steps


def parse_task(task_id):
    """(host, trap, source) — host is the SPLIT (boltons=real repo, synthetic=mock module);
    the query label is trap+source, e.g. budget-F, omdset-H."""
    if task_id.startswith("boltons-"):
        trap = task_id[len("boltons-"):].rsplit("-", 1)[0]
        try:
            src = json.loads((SDEBENCH / "datasets" / f"boltons-{trap}" / "tasks" / "main" / "task.json").read_text()).get("source", "F")
        except Exception:
            src = "F"
        return "boltons", trap, src
    if task_id.startswith("gen-"):
        parts = task_id.split("-")   # gen-budget-F-001
        return "synthetic", parts[1], (parts[2] if len(parts) > 2 else "?")
    return "other", task_id, "?"


def to_query_result(t, key, git_history, query_id=None):
    tok = t.get("tokens", {})
    cost = compute_cost(t.get("model"), tok)
    return {
        "query_id": query_id or key,
        "query": t.get("bug_report", ""),
        "answer": t.get("final_patch", "") or "(empty patch)",
        "trajectory": flatten_trace(t.get("trace", [])),
        "reasoning": "\n".join(s.get("text", "") for r in t.get("trace", []) for s in r.get("trajectory", []) if s.get("k") == "say")[:4000] or "(no assistant text)",
        "context": "", "context_tokens": 0, "retrieve_time_ms": 0.0,
        "gold_answers": [], "correct": t.get("solved", False),
        "judge_reason": (f"solved after {t.get('interventions')} intervention(s)" if t.get("solved")
                         else f"unsolved (capped) — {t.get('final_pytest','')}"),
        "score": None,
        "meta": {"interventions": t.get("interventions"), "history": t.get("history"),
                 "tokens": tok, "wall_s": t.get("wall_s"), "turns": t.get("turns"),
                 "cost_usd": cost},
        "git_history": git_history,   # the repo's engineered commits (source docs)
        "raw_response": None, "category_axes": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="filter trace dirs under /tmp/sdebench/run")
    args = ap.parse_args()

    traces = sorted(glob.glob(str(RUN_DIR / args.glob / "trace.json")))
    # group into SPLITS by host (boltons / synthetic); each task = one query labeled trap-source.
    runs = {}  # (model, history, variant, host) -> {task_id: (key, trace)}
    for f in traces:
        t = json.loads(Path(f).read_text())
        host, trap, src = parse_task(t["task_id"])
        key4 = (t["model"], t["history"], t.get("variant", "base"), host)
        runs.setdefault(key4, {})[t["task_id"]] = (Path(f).parent.name, t, f"{trap}-{src}")

    hist_cache = {}  # repo -> git history (fallback capture for old runs w/o stored history)
    for (model, history, variant, host), tasks in runs.items():
        results = []
        for task_id, (key, t, label) in sorted(tasks.items(), key=lambda kv: kv[1][2]):
            gh = t.get("git_history")
            if not gh:   # backfill old runs by rebuilding the codebase
                cb = repo_of(t)
                if cb not in hist_cache:
                    task = json.loads((SDEBENCH / "datasets" / cb / "task.json").read_text())
                    hist_cache[cb] = capture_git_history(task)
                gh = hist_cache[cb]
            results.append(to_query_result(t, key, gh, query_id=label))
        repo = host   # the SPLIT is the host (boltons | synthetic)
        run_name = f"opencode+{model}+{history}+{variant}"
        correct = sum(1 for r in results if r["correct"])
        avg_interv = round(sum((r["meta"]["interventions"] or 0) for r in results) / len(results), 2)
        tot_cost = round(sum(r["meta"]["cost_usd"] for r in results), 4)
        avg_cost = round(tot_cost / len(results), 4)
        sum_tok = {k: sum(r["meta"]["tokens"].get(k, 0) for r in results)
                   for k in ("input", "output", "reasoning", "cache_read", "cache_write")}
        summary = {
            "view": "agent",
            "dataset": "sdebench", "split": repo, "category": None,
            "memory_provider": run_name, "run_name": _safe(run_name), "mode": "agent", "oracle": False,
            "total_queries": len(results), "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "ingestion_time_ms": 0.0, "ingested_docs": 0,
            "description": f"sdebench — {history} history — mean interv {avg_interv}, avg ${avg_cost}/task",
            "answer_llm": model, "judge_llm": "execution (FAIL_TO_PASS+PASS_TO_PASS+HIDDEN)",
            "avg_retrieve_time_ms": None, "avg_context_tokens": None,
            # sdebench cost/speed roll-up (shown in the UI)
            "sde_mean_interventions": avg_interv, "sde_total_cost_usd": tot_cost,
            "sde_avg_cost_usd": avg_cost, "sde_tokens": sum_tok,
            "results": results,
        }
        dest = OUT / _safe(run_name) / "agent" / f"{repo}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2))
        print(f"[ok] {run_name}: {correct}/{len(results)} solved, mean interv {avg_interv} -> {dest.relative_to(REPO_ROOT)}")
    print("\nOpen `uv run omb view` -> dataset 'sdebench'")


if __name__ == "__main__":
    main()
