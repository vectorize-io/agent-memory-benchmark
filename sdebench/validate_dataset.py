"""Dataset integrity check (structural). Verifies every boltons-* task is well-formed:
required task.json fields, tests parse, manifest consistency. (Discrimination — HEAD fails repro+hidden,
correct passes, naive fails hidden — is validated by gen/emit_realfn.py and gen/validate.py.)
Usage: uv run python sdebench/validate_dataset.py"""
import ast, json, sys
from pathlib import Path

DS = Path(__file__).resolve().parent / "datasets"
REQ = ["task_id", "codebase", "build", "source", "tier", "module", "policy", "bug_report",
       "fail_to_pass", "pass_to_pass", "hidden_to_pass", "regression_test_file", "hidden_test_file"]
problems = []
tasks = sorted(DS.glob("boltons-*"))
for d in tasks:
    tp = d / "tasks" / "main" / "task.json"
    if not tp.exists():
        problems.append(f"{d.name}: missing task.json"); continue
    t = json.loads(tp.read_text())
    for k in REQ:
        if k not in t:
            problems.append(f"{d.name}: missing field '{k}'")
    if not (d / "build.py").exists():
        problems.append(f"{d.name}: missing build.py")
    for tf in ["regression_test.py", "hidden_test.py"]:
        p = d / "tasks" / "main" / tf
        if not p.exists():
            problems.append(f"{d.name}: missing {tf}")
        else:
            try:
                ast.parse(p.read_text())
            except SyntaxError as e:
                problems.append(f"{d.name}: {tf} syntax error: {e}")
    if t.get("source") not in ("H", "F"):
        problems.append(f"{d.name}: source not in H/F")
    if t.get("source") == "F" and not t.get("conversations"):
        problems.append(f"{d.name}: F source needs a chat (conversations)")

man = json.loads((DS / "MANIFEST.json").read_text())
man_ids = {m["task_id"] for m in man["tasks"]}
disk_ids = {json.loads((d / "tasks/main/task.json").read_text())["task_id"] for d in tasks if (d / "tasks/main/task.json").exists()}
if man_ids != disk_ids:
    problems.append(f"manifest/disk mismatch: only-manifest={man_ids - disk_ids} only-disk={disk_ids - man_ids}")

print(f"checked {len(tasks)} boltons tasks against {len(REQ)} required fields + tests + manifest")
rf = sum(1 for d in tasks if json.loads((d/'tasks/main/task.json').read_text()).get('tier')=='real-function')
print(f"  tiers: {rf} real-function, {len(tasks)-rf} planted")
if problems:
    print(f"\n{len(problems)} PROBLEMS:")
    for p in problems:
        print("  x", p)
    sys.exit(1)
print("\nALL TASKS WELL-FORMED")
