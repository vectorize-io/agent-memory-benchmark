"""Seed the shared memory store from EVERY generated project (so retrieval has distractors)."""
import json, sys, tempfile, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "gen"))
from mem import ingest_project, ingest_history_noise, write_store          # memsys/mem.py

REF = "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe"   # pinned boltons fork point; all tasks branch here
import importlib.util
spec = importlib.util.spec_from_file_location("gencore", HERE.parents[0] / "gen" / "core.py")
gencore = importlib.util.module_from_spec(spec); spec.loader.exec_module(gencore)
import importlib.util as _i
_t = _i.spec_from_file_location("traps", HERE.parents[0] / "gen" / "traps.py")
traps = _i.module_from_spec(_t); _t.loader.exec_module(traps)

DATASETS = HERE.parents[0] / "datasets"


def main():
    all_entries = []
    # Launch store: ONLY the boltons-* host tasks + decoys + distractors.
    # gen-* generator tasks are dev scaffolding and are NOT seeded (near-duplicate chats).
    import subprocess
    seeded_history = False
    for ds in sorted(DATASETS.glob("boltons-*")):
        tp = ds / "tasks" / "main" / "task.json"
        if not tp.exists():
            continue
        task = json.loads(tp.read_text())
        repo = Path(tempfile.mkdtemp(prefix="seed_host_"))
        try:
            subprocess.run(["python", str(ds / "build.py"), str(repo)], check=True, capture_output=True)
            # the task's OWN decision commits (REF..HEAD, code-enriched) + its conversation
            all_entries += ingest_project(repo, task.get("conversations"), task["task_id"], base_ref=REF)
            # inherited upstream boltons history = realistic retrieval noise, seeded ONCE
            if not seeded_history:
                all_entries += ingest_history_noise(repo, head=REF)
                seeded_history = True
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    decoyf = HERE.parents[0] / "gen" / "decoy_sessions.json"   # irrelevant chat sessions = chat-noise
    if decoyf.exists():
        for i, d in enumerate(json.loads(decoyf.read_text())):
            all_entries += ingest_project(Path(tempfile.mkdtemp(prefix="decoy_")), d["turns"], f"decoy-{i}")
    from distractors import ENTRIES as DISTRACTORS   # realistic noise from other domains
    all_entries += DISTRACTORS
    uniq = write_store(all_entries)
    from collections import Counter
    print(f"seeded {len(uniq)} unique entries (from {len(all_entries)}); by kind:", dict(Counter(e["kind"] for e in uniq)))


if __name__ == "__main__":
    main()
