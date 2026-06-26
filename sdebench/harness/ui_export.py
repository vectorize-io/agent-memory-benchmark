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
import argparse, json, re, glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "sdebench"
RUN_DIR = Path("/tmp/sdebench/run")


def _safe(s):
    return re.sub(r"[^A-Za-z0-9.+_-]", "_", s)


def flatten_trace(trace):
    """Multi-round trace -> one trajectory with feedback markers between rounds."""
    steps = []
    for rnd in trace:
        if rnd["role"] != "initial":
            steps.append({"k": "say",
                          "text": f"🔁 {rnd['role']} — feedback sent to the agent:\n{rnd['prompt'][:1400]}"})
        steps.extend(rnd.get("trajectory", []))
    return steps


def to_query_result(t, key):
    tok = t.get("tokens", {})
    return {
        "query_id": key,
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
                 "input_tok": tok.get("input"), "output_tok": tok.get("output"),
                 "cache_read": tok.get("cache_read")},
        "raw_response": None, "category_axes": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="filter trace dirs under /tmp/sdebench/run")
    args = ap.parse_args()

    traces = sorted(glob.glob(str(RUN_DIR / args.glob / "trace.json")))
    runs = {}  # (model, history) -> [trace dicts]
    for f in traces:
        t = json.loads(Path(f).read_text())
        runs.setdefault((t["model"], t["history"]), []).append((Path(f).parent.name, t))

    for (model, history), items in runs.items():
        run_name = f"opencode+{model}+{history}"
        results = [to_query_result(t, key) for key, t in items]
        correct = sum(1 for r in results if r["correct"])
        avg_interv = round(sum((r["meta"]["interventions"] or 0) for r in results) / len(results), 2)
        summary = {
            "view": "agent",
            "dataset": "sdebench", "split": "all", "category": None,
            "memory_provider": run_name, "run_name": _safe(run_name), "mode": "agent", "oracle": False,
            "total_queries": len(results), "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "ingestion_time_ms": 0.0, "ingested_docs": 0,
            "description": f"sdebench — {history} history — mean interventions {avg_interv}",
            "answer_llm": model, "judge_llm": "execution (FAIL_TO_PASS+PASS_TO_PASS+HIDDEN)",
            "avg_retrieve_time_ms": None, "avg_context_tokens": None,
            "results": results,
        }
        dest = OUT / _safe(run_name) / "agent" / "all.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2))
        print(f"[ok] {run_name}: {correct}/{len(results)} solved, mean interv {avg_interv} -> {dest.relative_to(REPO_ROOT)}")
    print("\nOpen `uv run omb view` -> dataset 'sdebench'")


if __name__ == "__main__":
    main()
