from abc import ABC, abstractmethod
from typing import Literal

from ..models import Document, Query
from ..utils import count_tokens

_DEFAULT_OPEN_PROMPT = """\
You are a helpful assistant. Answer the following question based ONLY on the provided context.

Context:
{context}

Question: {query}

Think step by step about what the context says, then provide your answer.\
"""

_DEFAULT_MCQ_PROMPT = """\
You are a helpful assistant. Based ONLY on the provided context, choose the best answer to the question below.

Important: wherever the context refers to "user", it refers to the same person named in the question.

Context:
{context}

Question and options:
{query}

Think step by step using the context. Focus on what is UNIQUELY true about this specific user — \
pay special attention to documented pain points, past frustrations, and explicit preferences, \
not just general background traits. The best option is the one most precisely grounded in the \
user's specific documented experiences, not the one that merely fits their general profile.\
"""


class Dataset(ABC):
    name: str
    description: str
    splits: list[str]
    task_type: Literal["open", "mcq"] = "open"
    isolation_unit: str | None = None
    links: list[dict] = []
    published: bool = False
    """Named links for the dataset, e.g. [{"label": "Paper", "url": "https://arxiv.org/..."}]."""
    """Semantic name of the storage isolation unit, e.g. 'episode' or 'session'.
    When set, memory providers may create separate storage per unit (e.g. one bank
    per episode). None means no meaningful isolation boundary exists."""

    def get_isolation_id(self, doc: Document) -> str | None:
        """Return the isolation unit ID for a document.

        The runner uses this to collect all unit IDs before calling prepare(), so
        providers can set up per-unit storage upfront. Also used to determine the
        user_id passed to ingest/retrieve.

        Default: doc.user_id. Override when the isolation key comes from elsewhere
        (e.g. doc.id, or a field in doc.messages)."""
        return doc.user_id

    def categories(self, split: str) -> list[str] | None:
        """Return the list of category names for a split, or None if not applicable."""
        return None

    def category_type(self, split: str, category: str) -> Literal["doc", "query"]:
        """Return whether a category partitions documents ('doc') or is query-only ('query').

        'doc'   — category partitions the document space; per-category doc/token counts are meaningful.
        'query' — category is a query-level filter only; doc/token counts would equal the full split.
        """
        return "doc"

    @abstractmethod
    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        """Load queries for a split. If category given, filter to that subset."""
        ...

    @abstractmethod
    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        """Load documents for a split. If ids is provided, return only those documents.
        If user_ids is provided, return only documents belonging to those users/units."""
        ...

    def split_stats(self, split: str) -> dict:
        """Return query/doc counts, token stats, and per-category breakdown."""
        queries = self.load_queries(split)
        docs = self.load_documents(split)
        token_counts = [count_tokens(d.content) for d in docs]
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens // len(token_counts) if token_counts else 0
        stats: dict = {
            "queries": len(queries),
            "docs": len(docs),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
        }
        cats = self.categories(split)
        if cats:
            breakdown: dict[str, dict] = {}
            for cat in cats:
                cat_queries = self.load_queries(split, category=cat)
                cat_type = self.category_type(split, cat)
                if cat_type == "query":
                    breakdown[cat] = {
                        "type": "query",
                        "queries": len(cat_queries),
                        "docs": None,
                        "total_tokens": None,
                        "avg_tokens_per_doc": None,
                    }
                else:
                    cat_docs = self.load_documents(split, ids={gid for q in cat_queries for gid in q.gold_ids})
                    cat_tokens = [count_tokens(d.content) for d in cat_docs]
                    total = sum(cat_tokens)
                    breakdown[cat] = {
                        "type": "doc",
                        "queries": len(cat_queries),
                        "docs": len(cat_docs),
                        "total_tokens": total,
                        "avg_tokens_per_doc": total // len(cat_tokens) if cat_tokens else 0,
                    }
            stats["categories"] = breakdown
        return stats

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        task_type: str,
        split: str,
        category: str | None = None,
        meta: dict | None = None,
    ) -> str:
        """Build the RAG answer prompt. Override per-dataset for custom prompting."""
        if task_type == "mcq":
            return _DEFAULT_MCQ_PROMPT.format(context=context, query=query)
        return _DEFAULT_OPEN_PROMPT.format(context=context, query=query)

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        """Build the judge prompt for evaluating an answer. Override per-dataset for custom judging."""
        return None  # None means use the default judge prompt in GeminiJudge

    def default_judge_llm(self):
        """Return the preferred LLM for judging this dataset, or None to use the global default."""
        return None

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        """Return category axes for a result based on its meta dict.

        Returns a dict mapping axis name → list of category values.
        Used to annotate QueryResult.category_axes for breakdown and filtering.
        Default: uses meta["category"] if present. Override per dataset.
        """
        c = meta.get("category")
        return {"Category": [c]} if c else {}

    def supports_oracle(self) -> bool:
        """Return True if this dataset provides gold_ids for oracle mode."""
        return True

    def dataset_stats(self, console) -> None:
        """Print dataset-specific statistics. Implement per dataset."""
        raise NotImplementedError(f"Dataset '{self.name}' does not implement dataset_stats()")
