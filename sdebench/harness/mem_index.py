"""Build a local 'intent index' over a codebase's git history for the recall_intent tool.

This is the LOCAL memory system (an alternative to Hindsight, which summarizes commits into
facts and loses the precise diff/sha the agent needs to fix a regression). It keeps the RAW
commits — subject, body, changed files, and the diff — so the recall_intent tool can return
the exact change + rationale, ranked by the agent's query. General-purpose: nothing here is
task-specific; it just indexes whatever history the codebase has.

Output: /tmp/sdebench/memindex/<codebase>.json = [{sha, subject, body, files, diff}], newest first.

Usage: python mem_index.py <codebase_build.py> <out.json>
"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path


def build_index(build_py: str, out_path: str):
    src = Path(tempfile.mkdtemp(prefix="memindex_"))
    try:
        subprocess.run(["python", build_py, str(src)], check=True, capture_output=True)
        shas = subprocess.run(["git", "-C", str(src), "rev-list", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.split()
        out = []
        for sha in shas:
            def g(*a):
                return subprocess.run(["git", "-C", str(src), *a], capture_output=True, text=True).stdout
            subject = g("show", "-s", "--format=%s", sha).strip()
            body = g("show", "-s", "--format=%b", sha).strip()
            files = g("show", "--name-only", "--format=", sha).split()
            diff = g("show", "--format=", sha)
            out.append({"sha": sha[:8], "subject": subject, "body": body,
                        "files": files, "diff": diff[:6000]})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out))
        print(f"[mem_index] {len(out)} commits -> {out_path}")
    finally:
        shutil.rmtree(src, ignore_errors=True)


if __name__ == "__main__":
    build_index(sys.argv[1], sys.argv[2])
