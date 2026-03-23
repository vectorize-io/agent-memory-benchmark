from .base import MemoryProvider
from .bm25 import BM25MemoryProvider
from .cognee import CogneeMemoryProvider
from .hindsight import HindsightCloudMemoryProvider, HindsightHTTPMemoryProvider, HindsightMemoryProvider
from .mastra import MastraMemoryProvider
from .mastra_om import MastraOMMemoryProvider
from .mem0 import Mem0MemoryProvider
from .mem0_cloud import Mem0CloudMemoryProvider
from .hybrid_search import HybridSearchMemoryProvider
from .supermemory import SupermemoryMemoryProvider

REGISTRY: dict[str, type[MemoryProvider]] = {
    "bm25": BM25MemoryProvider,
    "cognee": CogneeMemoryProvider,
    "hindsight": HindsightMemoryProvider,
    "hindsight-cloud": HindsightCloudMemoryProvider,
    "hindsight-http": HindsightHTTPMemoryProvider,

    "mastra": MastraMemoryProvider,
    "mastra-om": MastraOMMemoryProvider,
    "mem0": Mem0MemoryProvider,
    "mem0-cloud": Mem0CloudMemoryProvider,
    "qdrant": HybridSearchMemoryProvider,
    "supermemory": SupermemoryMemoryProvider,
}


def get_memory_provider(name: str) -> MemoryProvider:
    if name not in REGISTRY:
        raise ValueError(f"Unknown memory provider: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
