from rank_bm25 import BM25Okapi

from ..models import Document
from ..utils import chunk_text
from .base import MemoryProvider


class BM25MemoryProvider(MemoryProvider):
    name = "bm25"
    description = "Keyword search baseline. No embeddings — splits docs into chunks and uses BM25 ranking."
    kind = "local"

    def __init__(self):
        self._chunks: list[Document] = []
        self._index: BM25Okapi | None = None

    def ingest(self, documents: list[Document]) -> None:
        self._chunks = [
            Document(id=doc.id, content=chunk, user_id=doc.user_id)
            for doc in documents
            for chunk in chunk_text(doc.content)
        ]
        tokenized = [c.content.lower().split() for c in self._chunks]
        self._index = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        if self._index is None:
            raise RuntimeError("No documents ingested yet")

        if user_id is not None:
            subset = [c for c in self._chunks if c.user_id == user_id]
            if subset:
                index = BM25Okapi([c.content.lower().split() for c in subset])
                scores = index.get_scores(query.lower().split())
                top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
                return [subset[i] for i in top_k], None

        scores = self._index.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_k], None
