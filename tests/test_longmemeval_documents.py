from memory_bench.dataset.longmemeval import LongMemEvalDataset


def test_a_repeated_session_gets_its_own_document_id(monkeypatch):
    item = {
        "question_id": "q1",
        "question_type": "multi-session",
        "haystack_session_ids": ["s1", "s2", "s1"],
        "haystack_dates": ["2023/05/20 (Sat) 10:00", "2023/05/21 (Sun) 10:00", "2023/05/22 (Mon) 10:00"],
        "haystack_sessions": [[{"role": "user", "content": "hi"}]] * 3,
    }
    ds = LongMemEvalDataset()
    monkeypatch.setattr(ds, "_load_raw", lambda: [item])
    ids = [d.id for d in ds.load_documents("s")]
    assert ids == ["q1_s1", "q1_s2", "q1_s1~2"]
