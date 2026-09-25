"""Faithfulness self-check for the PrecisionMemBench port.

Feeds the scorer a perfect provider — one that returns exactly the beliefs each
case expects — and asserts it reproduces upstream's reference row for `tenure`:
43/43 active passes, 77/77 total, mean precision and recall 1.00, and the
43 active / 25 structural / 9 trivially-empty case split.

    uv run python scripts/precisionmembench_selfcheck.py
"""
import sys
from collections import Counter

from memory_bench.dataset.precisionmembench import PrecisionMemBenchDataset
from memory_bench.models import Document

EXPECTED_SPLIT = {"active": 43, "structural": 25, "trivially-empty": 9}


def main() -> int:
    ds = PrecisionMemBenchDataset()
    queries = ds.load_queries("single-turn")

    passed: Counter = Counter()
    total: Counter = Counter()
    failures: list[tuple[str, str]] = []
    for q in queries:
        rb = q.meta["expect"].get("relevantBeliefs") or {}
        ids = rb.get("shouldOnlyInclude")
        if ids is None:
            ids = rb.get("mustInclude") or []
        docs = [Document(id=i, content="", source_ids=[i]) for i in ids]
        ok, reason = ds.score_retrieval(q, docs)
        total[q.meta["pass_type"]] += 1
        passed[q.meta["pass_type"]] += int(ok)
        if not ok:
            failures.append((q.id, reason))

    precisions = [q.meta["retrieval_precision"] for q in queries if q.meta["retrieval_precision"] is not None]
    recalls = [q.meta["retrieval_recall"] for q in queries if q.meta["retrieval_recall"] is not None]
    mean_p = sum(precisions) / len(precisions)
    mean_r = sum(recalls) / len(recalls)

    print(f"case split      : {dict(total)}")
    print(f"passes by type  : {dict(passed)}")
    print(f"total passes    : {sum(passed.values())}/{len(queries)}")
    print(f"precision/recall: {mean_p:.2f} / {mean_r:.2f}")

    problems = []
    if dict(total) != EXPECTED_SPLIT:
        problems.append(f"case split {dict(total)} != upstream {EXPECTED_SPLIT}")
    if sum(passed.values()) != len(queries):
        problems.append(f"perfect provider scored {sum(passed.values())}/{len(queries)}, expected all")
    if round(mean_p, 2) != 1.00 or round(mean_r, 2) != 1.00:
        problems.append(f"precision/recall {mean_p:.2f}/{mean_r:.2f}, expected 1.00/1.00")
    for case_id, reason in failures:
        print(f"  FAIL {case_id}: {reason}")
    if problems:
        for p in problems:
            print(f"MISMATCH: {p}")
        return 1
    print("OK — port matches upstream's reference row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
