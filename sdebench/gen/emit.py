"""Emit generated (trap, source) tasks as harness-runnable dataset dirs.

For each trap x source it writes sdebench/datasets/gen-<trap>-<source>/{build.py stub, tasks/main/*}
so the existing harness can run them with --task .../tasks/main/task.json.

Usage: uv run python sdebench/gen/emit.py            # emit the whole matrix (H/K/F)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traps import TRAPS
from core import task_spec

DATASETS = Path(__file__).resolve().parents[1] / "datasets"

STUB = ('import sys\nfrom pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen"))\n'
        'from traps import TRAPS\nfrom core import build\n\n'
        'build(TRAPS[{name!r}], {source!r}, sys.argv[1])\n')


def emit(trap_name, source):
    trap = TRAPS[trap_name]
    codebase = f"gen-{trap_name}-{source}"
    ds = DATASETS / codebase
    (ds / "tasks" / "main").mkdir(parents=True, exist_ok=True)
    (ds / "build.py").write_text(STUB.format(name=trap_name, source=source))
    task, repro, hidden = task_spec(trap, source, codebase)
    (ds / "tasks" / "main" / "task.json").write_text(json.dumps(task, indent=2))
    (ds / "tasks" / "main" / "regression_test.py").write_text(repro)
    (ds / "tasks" / "main" / "hidden_test.py").write_text(hidden)
    return f"{codebase}/tasks/main/task.json"


def main():
    sources = ["H", "K", "F"]
    for name in TRAPS:
        for src in sources:
            print("emitted:", emit(name, src))


if __name__ == "__main__":
    main()
