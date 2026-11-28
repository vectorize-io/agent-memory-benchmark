from .base import MemoryProvider
from .bm25 import BM25MemoryProvider
from .cognee import CogneeMemoryProvider
from .evolving_profile import EvolvingProfileMemoryProvider
from .hindsight import HindsightCloudMemoryProvider, HindsightHTTPMemoryProvider, HindsightMemoryProvider
from .hindsight_knowledge import HindsightKnowledgeMemoryProvider
from .hindsight_support import HindsightSupportKnowledgeProvider, HindsightSupportLinkedProvider, HindsightSupportMemoryProvider
from .mastra import MastraMemoryProvider
from .mastra_om import MastraOMMemoryProvider
from .mem0 import Mem0MemoryProvider
from .mem0_cloud import Mem0CloudMemoryProvider
from .hybrid_search import HybridSearchMemoryProvider
from .ogham import OghamMemoryProvider
from .supermemory import SupermemoryMemoryProvider
from .none import NoMemoryProvider
from .hscoding import HsCodingProvider

REGISTRY: dict[str, type[MemoryProvider]] = {
    "vanilla": NoMemoryProvider,
    "hindsight-coding": HsCodingProvider,
    "bm25": BM25MemoryProvider,
    "cognee": CogneeMemoryProvider,
    "evolving-profile": EvolvingProfileMemoryProvider,
    "hindsight": HindsightMemoryProvider,
    "hindsight-cloud": HindsightCloudMemoryProvider,
    "hindsight-http": HindsightHTTPMemoryProvider,
    "hindsight-knowledge": HindsightKnowledgeMemoryProvider,
    "hindsight-support-memory": HindsightSupportMemoryProvider,
    "hindsight-support-knowledge": HindsightSupportKnowledgeProvider,
    "hindsight-support-linked": HindsightSupportLinkedProvider,

    "mastra": MastraMemoryProvider,
    "mastra-om": MastraOMMemoryProvider,
    "mem0": Mem0MemoryProvider,
    "mem0-cloud": Mem0CloudMemoryProvider,
    "ogham": OghamMemoryProvider,
    "qdrant": HybridSearchMemoryProvider,
    "supermemory": SupermemoryMemoryProvider,
}
# legacy aliases (docs/scripts used these); canonical names above
REGISTRY["none"] = NoMemoryProvider
REGISTRY["hscoding"] = HsCodingProvider


def get_memory_provider(name: str) -> MemoryProvider:
    if name not in REGISTRY:
        raise ValueError(f"Unknown memory provider: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
