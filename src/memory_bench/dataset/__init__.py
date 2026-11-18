from .base import Dataset
from .beam import BEAMDataset
from .knowledge import KNOWLEDGE_DATASETS
from .lifebench import LifeBenchDataset
from .locomo import LoComoDataset
from .locomo_fixed import LoComoFixedDataset
from .longmemeval import LongMemEvalDataset
from .personamem import PersonaMemDataset
from .precisionmembench import PrecisionMemBenchDataset
from .sdebench import SdebenchDataset
from .supportbench import SupportBenchDataset

REGISTRY: dict[str, type[Dataset]] = {
    "beam":         BEAMDataset,
    "lifebench":    LifeBenchDataset,
    "locomo":       LoComoDataset,
    "locomo-fixed": LoComoFixedDataset,
    "longmemeval":  LongMemEvalDataset,    "personamem":   PersonaMemDataset,
    "precisionmembench": PrecisionMemBenchDataset,
    "sdebench":     SdebenchDataset,
    "supportbench": SupportBenchDataset,
    # Knowledge-bank retrieval (search only, ranked scores): see dataset/knowledge.py
    **{cls.name: cls for cls in KNOWLEDGE_DATASETS},
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
