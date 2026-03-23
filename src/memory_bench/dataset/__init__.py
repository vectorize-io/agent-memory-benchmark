from .base import Dataset
from .lifebench import LifeBenchDataset
from .locomo import LoComoDataset
from .longmemeval import LongMemEvalDataset
from .membench import MemBenchDataset
from .memsim import MemSimDataset
from .personamem import PersonaMemDataset

REGISTRY: dict[str, type[Dataset]] = {
    "lifebench":    LifeBenchDataset,
    "locomo":       LoComoDataset,
    "longmemeval":  LongMemEvalDataset,
    "membench":     MemBenchDataset,
    "memsim":       MemSimDataset,
    "personamem":   PersonaMemDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
