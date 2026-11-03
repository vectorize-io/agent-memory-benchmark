from memory_bench import llm


class FakeLLM:
    def __init__(self, model=None):
        self.model = model


def test_answer_llm_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("OMB_ANSWER_LLM", raising=False)
    monkeypatch.setitem(llm.REGISTRY, "gemini", FakeLLM)

    assert isinstance(llm.get_answer_llm(), FakeLLM)


def test_answer_llm_uses_environment_provider(monkeypatch):
    monkeypatch.setenv("OMB_ANSWER_LLM", "groq")
    monkeypatch.setitem(llm.REGISTRY, "groq", FakeLLM)

    assert isinstance(llm.get_answer_llm(), FakeLLM)


def test_explicit_provider_overrides_environment(monkeypatch):
    monkeypatch.setenv("OMB_ANSWER_LLM", "groq")
    monkeypatch.setitem(llm.REGISTRY, "openai", FakeLLM)

    assert isinstance(llm.get_answer_llm("openai"), FakeLLM)
