"""Aggregate a boltons-suite run (vanilla vs memsys) into a per-task + total table with n and means.
Usage: uv run python sdebench/harness/aggregate.py [run-id-glob]   (default n*)"""
import json, glob, sys, statistics as st
from collections import defaultdict

RID = sys.argv[1] if len(sys.argv) > 1 else "n*"
rows = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"/tmp/sdebench/run/boltons-*_{RID}/result.json"):
    r = json.load(open(f))
    rows[r["task_id"]][r["history"]].append(r)


def agg(rs):
    if not rs:
        return None
    return dict(n=len(rs), interv=sum(x["interventions"] for x in rs),
                solved=sum(x["solved"] for x in rs), turns=st.mean(x["turns"] for x in rs),
                cost=st.mean(x["cost_usd"] for x in rs))


print(f"{'task':<22}{'n':>3}  {'vanilla interv/turns':>22}   {'memsys interv/turns':>22}")
tb = tm = tbt = tmt = nrun = 0
for tid in sorted(rows):
    v, m = agg(rows[tid].get("full")), agg(rows[tid].get("memsys"))
    if not (v and m):
        continue
    tb += v["interv"]; tm += m["interv"]; tbt += v["turns"] * v["n"]; tmt += m["turns"] * m["n"]; nrun += v["n"]
    print(f"{tid.replace('boltons-','').replace('-001',''):<22}{v['n']:>3}  "
          f"{v['interv']:>3} ({v['interv']/v['n']:.1f}/run) {v['turns']:>5.0f}t   "
          f"{m['interv']:>3} ({m['interv']/m['n']:.1f}/run) {m['turns']:>5.0f}t")
print("-" * 74)
print(f"{'TOTAL':<22}{'':>3}  vanilla {tb} interv / {tbt/max(nrun,1):.0f} avg turns   "
      f"memsys {tm} interv / {tmt/max(nrun,1):.0f} avg turns")
print(f"  vanilla {tb} -> memsys {tm} interventions | turns {tbt/max(nrun,1):.0f} -> {tmt/max(nrun,1):.0f} "
      f"({100*(tmt-tbt)/max(tbt,1):.0f}%)")
