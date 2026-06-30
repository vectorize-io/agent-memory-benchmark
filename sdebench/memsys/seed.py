"""Seed the shared memory store from EVERY generated project (so retrieval has distractors)."""
import json, sys, tempfile, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "gen"))
from mem import ingest_project, write_store          # memsys/mem.py
import importlib.util
spec = importlib.util.spec_from_file_location("gencore", HERE.parents[0] / "gen" / "core.py")
gencore = importlib.util.module_from_spec(spec); spec.loader.exec_module(gencore)
import importlib.util as _i
_t = _i.spec_from_file_location("traps", HERE.parents[0] / "gen" / "traps.py")
traps = _i.module_from_spec(_t); _t.loader.exec_module(traps)

DATASETS = HERE.parents[0] / "datasets"


def main():
    all_entries = []
    for name in traps.TRAPS:
        for src in ("H", "F"):
            tp = DATASETS / f"gen-{name}-{src}" / "tasks" / "main" / "task.json"
            if not tp.exists():
                continue
            task = json.loads(tp.read_text())
            repo = Path(tempfile.mkdtemp(prefix=f"seed_{name}_{src}_"))
            try:
                gencore.build(traps.TRAPS[name], src, repo)
                all_entries += ingest_project(repo, task.get("conversations"), f"gen-{name}-{src}")
            finally:
                shutil.rmtree(repo, ignore_errors=True)
    # host (real-codebase) tasks: ingest their chat + the host's real git history as noise
    import subprocess
    import mem as _mem
    for ds in sorted(DATASETS.glob("boltons-*")):
        tp = ds / "tasks" / "main" / "task.json"
        if not tp.exists():
            continue
        task = json.loads(tp.read_text())
        repo = Path(tempfile.mkdtemp(prefix="seed_host_"))
        try:
            subprocess.run(["python", str(ds / "build.py"), str(repo)], check=True, capture_output=True)
            all_entries += ingest_project(repo, task.get("conversations"), task["task_id"])
            out = subprocess.run(["git", "-C", str(repo), "log", "--format=%s%x1e"],
                                 capture_output=True, text=True).stdout
            for s in out.split("\x1e"):
                s = s.strip()
                if s and not _mem._NOISE_SUBJ.match(s):     # real commit subjects = retrieval noise
                    all_entries.append({"project": "boltons-history", "kind": "decision",
                                        "title": s, "text": s, "symbols": _mem._text_symbols(s)})
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    from distractors import ENTRIES as DISTRACTORS   # realistic noise from other domains
    all_entries += DISTRACTORS
    uniq = write_store(all_entries)
    from collections import Counter
    print(f"seeded {len(uniq)} unique entries (from {len(all_entries)}); by kind:", dict(Counter(e["kind"] for e in uniq)))


if __name__ == "__main__":
    main()
