import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ..models import Document
from ..utils import chunk_text
from .base import MemoryProvider

_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_DENSE_DIMS = 1024
_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_COLLECTION = "bench"


class HybridSearchMemoryProvider(MemoryProvider):
    name = "hybrid-search"
    description = (
        "Hybrid dense + sparse vector search via Qdrant with RRF fusion. "
        "Dense: Qwen3-Embedding-0.6B (1024d, asymmetric query/doc encoding). "
        "Sparse: BM42 (bm42-all-minilm-l6-v2-attentions). "
        "Documents chunked into 512-token windows before indexing. "
        "Retrieves top-k=50 chunks (prefetch 100 per branch)."
    )
    kind = "local"
    link = "https://qdrant.tech"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=qdrant.tech"

    def __init__(self):
        self._client: QdrantClient | None = None
        self._dense_model: SentenceTransformer | None = None
        self._sparse_model = None

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None) -> None:
        qdrant_path = store_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(qdrant_path))
        self._init_models()
        self._setup_collection()

    def _ensure_ready(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=tempfile.mkdtemp(prefix="qdrant_bench_"))
            self._init_models()
            self._setup_collection()
        return self._client

    def _init_models(self) -> None:
        from fastembed import SparseTextEmbedding
        self._dense_model = SentenceTransformer(_DENSE_MODEL, trust_remote_code=True, device="cpu")
        self._sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL)

    def _setup_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLLECTION in existing:
            self._client.delete_collection(_COLLECTION)
        self._client.create_collection(
            collection_name=_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=_DENSE_DIMS,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    def _dense(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        prompt_name = "query" if is_query else None
        return self._dense_model.encode(texts, prompt_name=prompt_name, normalize_embeddings=True).tolist()

    def _sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._sparse_model.embed(texts)
        ]

    def ingest(self, documents: list[Document]) -> None:
        client = self._ensure_ready()

        # Expand each document into chunks
        chunks: list[tuple[str, str, str | None]] = []  # (chunk, doc_id, user_id)
        for doc in documents:
            for chunk in chunk_text(doc.content):
                chunks.append((chunk, doc.id, doc.user_id))

        texts = [c[0] for c in chunks]
        dense_vecs = self._dense(texts)
        sparse_vecs = self._sparse(texts)

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "content": chunk,
                },
            )
            for (chunk, doc_id, user_id), dense, sparse in zip(chunks, dense_vecs, sparse_vecs)
        ]
        client.upsert(collection_name=_COLLECTION, points=points)

    async def async_retrieve(self, query: str, k: int = 50, user_id: str | None = None, query_timestamp: str | None = None):
        import asyncio
        return await asyncio.to_thread(self.retrieve, query, k, user_id, query_timestamp)

    def retrieve(
        self,
        query: str,
        k: int = 50,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        client = self._ensure_ready()
        dense_query = self._dense([query], is_query=True)[0]
        sparse_query = self._sparse([query])[0]

        filter_ = None
        if user_id is not None:
            filter_ = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )

        prefetch_limit = k * 2
        results = client.query_points(
            collection_name=_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=prefetch_limit,
                    filter=filter_,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=filter_,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )

        docs = []
        raw = []
        for point in results.points:
            payload = point.payload or {}
            raw.append({"id": str(point.id), "score": point.score, "payload": payload})
            docs.append(
                Document(
                    id=payload.get("doc_id", str(point.id)),
                    content=payload.get("content", ""),
                    user_id=payload.get("user_id"),
                )
            )
        return docs, {"results": raw}
