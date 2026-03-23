from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

SPLITS = [
    "bitcoin", "cardano", "iota", "monero",
    "economics", "law", "politics", "history",
    "quant", "travel", "workplace", "genealogy", "hsm",
]


class TempoDataset(Dataset):
    name = "tempo"
    description = "Time-sensitive QA testing temporal awareness of stored information."
    splits = SPLITS
    links = [
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/tempo26/Tempo"},
    ]

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        from datasets import load_dataset
        ds = load_dataset("tempo26/Tempo", "examples", split=split,
                          cache_dir=str(dataset_cache_dir("tempo")))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [
            Query(
                id=row["id"],
                query=row["query"],
                gold_ids=row["gold_ids"],
                gold_answers=row["gold_answers"],
            )
            for row in ds
        ]

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
    ) -> list[Document]:
        from datasets import load_dataset
        ds = load_dataset("tempo26/Tempo", "documents", split=split,
                          cache_dir=str(dataset_cache_dir("tempo")))
        if ids is not None:
            ds = ds.filter(lambda row: row["id"] in ids)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [Document(id=row["id"], content=row["content"]) for row in ds]

    def dataset_stats(self, console: Console, sample_size: int = 200) -> None:
        from datasets import load_dataset, load_dataset_builder
        cache = str(dataset_cache_dir("tempo"))

        console.print("[dim]Fetching counts...[/dim]")
        doc_counts   = {s: i.num_examples for s, i in load_dataset_builder("tempo26/Tempo", "documents", cache_dir=cache).info.splits.items()}
        query_counts = {s: i.num_examples for s, i in load_dataset_builder("tempo26/Tempo", "examples",   cache_dir=cache).info.splits.items()}

        table = Table(title="Tempo dataset stats")
        table.add_column("Split",               style="bold")
        table.add_column("Queries",             justify="right")
        table.add_column("Docs",                justify="right")
        table.add_column("Est. tokens (total)", justify="right")
        table.add_column("Avg tokens / doc",    justify="right")

        total_docs = total_queries = total_tokens = 0

        for split in SPLITS:
            n_docs    = doc_counts.get(split, 0)
            n_queries = query_counts.get(split, 0)
            sample    = min(sample_size, n_docs)
            ds = load_dataset("tempo26/Tempo", "documents", split=f"{split}[:{sample}]", cache_dir=cache)
            avg_tokens = int(sum(len(row["content"]) / 4 for row in ds) / max(sample, 1))
            est_total  = avg_tokens * n_docs

            total_docs    += n_docs
            total_queries += n_queries
            total_tokens  += est_total

            table.add_row(split, f"{n_queries:,}", f"{n_docs:,}", f"~{est_total / 1_000_000:.1f}M", f"~{avg_tokens:,}")

        table.add_section()
        table.add_row("TOTAL", f"{total_queries:,}", f"{total_docs:,}", f"~{total_tokens / 1_000_000:.1f}M", "", style="bold")
        console.print(table)
