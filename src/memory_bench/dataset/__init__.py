from .base import Dataset
from .beam import BEAMDataset
from .codingagent import CodingAgentDataset
from .lifebench import LifeBenchDataset
from .locomo import LoComoDataset
from .longmemeval import LongMemEvalDataset
from .personamem import PersonaMemDataset
from .precisionmembench import PrecisionMemBenchDataset
from .sdebench import SdebenchDataset

REGISTRY: dict[str, type[Dataset]] = {
    "beam":         BEAMDataset,
    "codingagent":  CodingAgentDataset,
    "lifebench":    LifeBenchDataset,
    "locomo":       LoComoDataset,
    "longmemeval":  LongMemEvalDataset,    "personamem":   PersonaMemDataset,
    "precisionmembench": PrecisionMemBenchDataset,
    "sdebench":     SdebenchDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
