from memory_bench.dataset import locomo_fixed
from memory_bench.dataset.locomo import LoComoDataset
from memory_bench.dataset.locomo_fixed import LoComoFixedDataset
from memory_bench.memory.hindsight import _HindsightBase


def _item() -> dict:
    sessions = {f"session_{n}": [{"speaker": "A", "dia_id": f"D{n}:1", "text": f"turn {n}"}] for n in (1, 2, 9, 10)}
    dates = {f"session_{n}_date_time": f"1:00 pm on {n} May, 2023" for n in (1, 2, 9, 10)}
    return {
        "sample_id": "conv-x",
        "conversation": {"speaker_a": "A", "speaker_b": "B", **sessions, **dates},
        "qa": [
            {"question": "first?", "answer": "a", "category": 1, "evidence": ["D1:1"]},
            {"question": "second?", "answer": "b", "category": 2, "evidence": ["D10:1"]},
        ],
    }


def test_sessions_are_in_numeric_order_so_questions_are_dated_by_the_last_one(monkeypatch):
    ds = LoComoDataset()
    monkeypatch.setattr(ds, "_load_raw", lambda: [_item()])
    assert [d.id for d in ds.load_documents("locomo10")] == [
        "conv-x_session_1", "conv-x_session_2", "conv-x_session_9", "conv-x_session_10",
    ]
    # As strings, session_9 sorted last and dated every question 9 May instead of 10 May.
    assert ds.load_queries("locomo10")[0].meta["query_timestamp"].startswith("2023-05-10")


def test_locomo_fixed_drops_only_the_listed_questions(monkeypatch):
    ds = LoComoFixedDataset()
    monkeypatch.setattr(ds, "_load_raw", lambda: [_item()])
    monkeypatch.setattr(locomo_fixed, "load_exclusions", lambda: {"conv-x_q1": "gold answer absent"})
    assert [q.id for q in ds.load_queries("locomo10")] == ["conv-x_q0"]


def test_every_exclusion_names_a_real_question_and_a_reason():
    ids = {q.id for q in LoComoDataset().load_queries("locomo10")}
    for query_id, reason in locomo_fixed.load_exclusions().items():
        assert query_id in ids, query_id
        assert reason.strip(), query_id


def test_reflect_gets_the_question_date_and_the_answer_rules():
    provider = _HindsightBase()
    provider._dataset = "locomo-fixed"
    kwargs = provider._reflect_kwargs("When?", None, "2023-10-22T09:55:00+00:00")
    assert "query_timestamp" not in kwargs, "reflect has no such field; the server drops it"
    assert kwargs["context"].startswith("The question is being asked on: 2023-10-22T09:55:00+00:00 UTC")
    assert "Always convert relative time references" in kwargs["context"]
