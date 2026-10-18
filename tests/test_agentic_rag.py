import unittest

from memory_bench.llm.base import LLM
from memory_bench.memory.bm25 import BM25MemoryProvider
from memory_bench.models import Document
from memory_bench.modes.agentic_rag import AgenticRAGMode


class FakeToolLLM(LLM):
    @property
    def model_id(self):
        return "fake:tool-llm"

    def tool_loop(self, prompt, tools, max_tool_calls=10):
        recall = tools[0].fn
        recall("future imports compile validation")
        recall("review convention current repo evidence")
        return "done"

    def generate(self, prompt, schema):
        return {
            "reasoning": "The current repo evidence overrides stale memory.",
            "answer": "Trust compile validation over parse-only memory.",
        }


class AgenticRAGModeTest(unittest.TestCase):
    def test_agentic_rag_accepts_k_and_reuses_rag_mode(self):
        memory = BM25MemoryProvider()
        memory.ingest([
            Document(
                id="stale",
                user_id="repo-a",
                content="Old session memory: ast.parse validation was considered enough.",
            ),
            Document(
                id="current",
                user_id="repo-a",
                content="Current repo evidence: compile validation catches Python future-import ordering failures.",
            ),
            Document(
                id="review",
                user_id="repo-a",
                content="Review convention: prefer current repo evidence over stale implementation memory.",
            ),
        ])

        mode = AgenticRAGMode(llm=FakeToolLLM(), k=1)
        result = mode.answer(
            "Should the agent trust parse-only memory or compile validation?",
            memory,
            user_id="repo-a",
        )

        self.assertEqual(result.answer, "Trust compile validation over parse-only memory.")
        self.assertIn("Current repo evidence", result.context)
        self.assertIn("Review convention", result.context)


if __name__ == "__main__":
    unittest.main()
