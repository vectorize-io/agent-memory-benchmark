import json
import os
import threading
import webbrowser
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

from .dataset import REGISTRY as DATASET_REGISTRY, get_dataset
from .llm import REGISTRY as LLM_REGISTRY, get_llm, get_answer_llm
from .memory import REGISTRY as MEMORY_REGISTRY, get_memory_provider
from .modes import REGISTRY as MODE_REGISTRY, get_mode
from .runner import EvalRunner


app = typer.Typer(help="Open Memory Benchmark (OMB).")
console = Console()


def _resolve_gemini_key() -> None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        typer.echo("Error: GEMINI_API_KEY environment variable is not set.", err=True)
        raise typer.Exit(1)
    os.environ["GOOGLE_API_KEY"] = key


@app.command()
def run(
    split: str = typer.Option(..., "--split", "-s"),
    dataset: str = typer.Option("tempo", "--dataset", help=f"Dataset. Available: {', '.join(DATASET_REGISTRY)}"),
    memory: str = typer.Option("bm25", "--memory", "-m", help=f"Memory provider. Available: {', '.join(MEMORY_REGISTRY)}"),
    mode: str = typer.Option("rag", "--mode", help=f"Response mode. Available: {', '.join(MODE_REGISTRY)}"),
    llm: str = typer.Option("gemini", "--llm", help=f"LLM for answer generation. Available: {', '.join(LLM_REGISTRY)}"),
    category: str = typer.Option(None, "--category", "-c", help="Category filter(s), comma-separated (e.g. 'a,b,c'). With --query-limit, runs N queries per category."),
    query_limit: int = typer.Option(None, "--query-limit", "-q", help="Max queries to evaluate. When combined with multiple --category values, applies per category."),
    query_id: str = typer.Option(None, "--query-id", help="Run a single specific query by ID"),
    doc_limit: int = typer.Option(None, "--doc-limit", help="Max number of documents to ingest"),
    oracle: bool = typer.Option(False, "--oracle", help="Ingest only gold documents (bypasses retrieval noise)"),
    skip_ingestion: bool = typer.Option(False, "--skip-ingestion", help="Skip ingestion and query the existing memory state"),
    skip_ingested: bool = typer.Option(False, "--skip-ingested", help="Skip units already present in a previous run's output (resume mode for unit-sequential datasets)"),
    skip_retrieval: bool = typer.Option(False, "--skip-retrieval", help="Skip retrieval and re-run answer generation using cached contexts from the previous run"),
    skip_answer: bool = typer.Option(False, "--skip-answer", help="Skip retrieval and answer generation entirely — re-judge cached answers from the previous run"),
    only_failed: bool = typer.Option(False, "--only-failed", help="Restrict queries to those that failed in the previous run"),
    show_raw: bool = typer.Option(False, "--show-raw", help="Print raw provider response after each query"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
    name: str = typer.Option(None, "--name", "-n", help="Run name used as output directory (defaults to memory provider name)"),
    description: str = typer.Option(None, "--description", "-d", help="Optional description for this run (stored in the result JSON)"),
) -> None:
    """Run an evaluation on a single split (optionally filtered to a category)."""
    _resolve_gemini_key()

    ds = get_dataset(dataset)

    if split not in ds.splits:
        typer.echo(f"Error: unknown split '{split}'. Available: {', '.join(ds.splits)}", err=True)
        raise typer.Exit(1)

    summary = EvalRunner(output_dir=output_dir).run(
        dataset=ds,
        split=split,
        memory=get_memory_provider(memory),
        mode=get_mode(mode, llm=get_answer_llm()),
        category=category,
        query_limit=query_limit,
        query_id=query_id,
        doc_limit=doc_limit,
        oracle=oracle,
        skip_ingestion=skip_ingestion,
        skip_ingested=skip_ingested,
        skip_retrieval=skip_retrieval,
        skip_answer=skip_answer,
        only_failed=only_failed,
        show_raw=show_raw,
        run_name=name,
        description=description,
    )

    cat_label = f"/{category}" if category else ""
    table = Table(title=f"Results — {dataset}/{split}{cat_label} | {memory} | {mode}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total queries", str(summary.total_queries))
    table.add_row("Correct", str(summary.correct))
    table.add_row("Accuracy", f"{summary.accuracy:.1%}")
    console.print(table)


@app.command("dataset-stats")
def dataset_stats(
    dataset: str = typer.Option("tempo", "--dataset", help=f"Dataset. Available: {', '.join(DATASET_REGISTRY)}"),
    sample_size: int = typer.Option(200, "--sample", help="Docs sampled per split to estimate token density"),
) -> None:
    """Print dataset-specific statistics."""
    ds = get_dataset(dataset)
    ds.dataset_stats(console, sample_size=sample_size)


@app.command()
def splits(
    dataset: str = typer.Option("tempo", "--dataset", help=f"Dataset. Available: {', '.join(DATASET_REGISTRY)}"),
) -> None:
    """List available splits for a dataset."""
    ds = get_dataset(dataset)
    for s in ds.splits:
        cats = ds.categories(s)
        if cats:
            typer.echo(f"{s}  [{', '.join(cats)}]")
        else:
            typer.echo(s)


@app.command()
def providers() -> None:
    """List all available memory providers and response modes."""
    console.print("[bold]Datasets:[/bold]",         ", ".join(DATASET_REGISTRY))
    console.print("[bold]Memory providers:[/bold]", ", ".join(MEMORY_REGISTRY))
    console.print("[bold]Response modes:[/bold]",   ", ".join(MODE_REGISTRY))


@app.command("publish-results")
def publish_results(
    result: Path = typer.Argument(..., help="Path to a result file (.json or .json.gz), e.g. outputs/personamem/mem0/rag/32k.json"),
    push: bool = typer.Option(False, "--push", help="Upload compressed result to Vercel Blob"),
    token: str = typer.Option(None, "--token", envvar="BLOB_READ_WRITE_TOKEN", help="Vercel Blob token (or set BLOB_READ_WRITE_TOKEN)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-upload even if unchanged"),
) -> None:
    """Compress a single benchmark result and optionally upload to Vercel Blob.

    Strips raw_response and gzips to .json.gz. Blob upload is checksum-based
    so re-running is instant if nothing changed. Always regenerates
    results-manifest.json from all local .json.gz files.

    Workflow:
        uv run omb publish-results outputs/personamem/mem0/rag/32k.json --push
        git add outputs/ results-manifest.json && git commit -m 'results: ...' && git push
    """
    import gzip as _gzip
    import hashlib
    import urllib.request

    root = Path(__file__).parents[2]
    abs_result = (root / result).resolve() if not result.is_absolute() else result.resolve()

    if not abs_result.exists():
        console.print(f"[red]File not found: {abs_result}[/red]")
        raise typer.Exit(1)

    # ── Compress if needed ────────────────────────────────────────────────
    if abs_result.suffix == ".json" and not abs_result.name.endswith(".json.gz"):
        try:
            data = json.loads(abs_result.read_text())
            results = data.pop("results", [])
            for r in results:
                r.pop("raw_response", None)
            rec_times  = [r["retrieve_time_ms"] for r in results if r.get("retrieve_time_ms") is not None]
            ctx_tokens = [r["context_tokens"]   for r in results if r.get("context_tokens")   is not None]
            data["avg_retrieve_time_ms"] = round(sum(rec_times)  / len(rec_times),  1) if rec_times  else None
            data["avg_context_tokens"]   = round(sum(ctx_tokens) / len(ctx_tokens), 1) if ctx_tokens else None
            data["results"] = results
            gz_path = abs_result.with_suffix(".json.gz")
            with _gzip.open(gz_path, "wt", compresslevel=9) as fh:
                json.dump(data, fh)
            abs_result.unlink()
            abs_result = gz_path
            console.print(f"  [green]✓[/green] compressed → {gz_path.name}")
        except Exception as e:
            console.print(f"  [red]✗[/red] compression failed: {e}")
            raise typer.Exit(1)
    elif not str(abs_result).endswith(".json.gz"):
        console.print(f"[red]Expected a .json or .json.gz file, got: {abs_result.name}[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"  [dim]already compressed: {abs_result.name}[/dim]")

    # ── Generate results-manifest.json ───────────────────────────────────
    # Determine outputs dir as the 4-levels-up parent of the result file
    # (outputs/<dataset>/<memory>/<mode>/<split>.json.gz → parts[-4:])
    abs_output = abs_result
    for _ in range(4):
        abs_output = abs_output.parent

    from .server import _list_results
    import os as _os
    _os.environ["OMB_OUTPUT_DIR"] = str(abs_output)
    _os.environ["OMB_ROOT"] = str(root)
    manifest_entries = _list_results(published_only=True)
    manifest_out = root / "results-manifest.json"
    manifest_out.write_text(json.dumps(manifest_entries, indent=2))
    console.print(f"  [green]✓[/green] results-manifest.json ({len(manifest_entries)} entries)")

    if not push:
        console.print("[dim]Skipping Blob upload (use --push to upload).[/dim]")
        return

    # ── Upload to Vercel Blob ─────────────────────────────────────────────
    if not token:
        console.print("[red]BLOB_READ_WRITE_TOKEN not set. Pass --token or add it to .env.[/red]")
        raise typer.Exit(1)

    console.rule("Uploading result to Vercel Blob")
    blob_manifest_path = root / ".blob_manifest.json"
    blob_manifest: dict[str, str] = json.loads(blob_manifest_path.read_text()) if blob_manifest_path.exists() else {}

    rel = str(abs_result.relative_to(root))
    sha = hashlib.sha256(abs_result.read_bytes()).hexdigest()
    if not force and blob_manifest.get(rel) == sha:
        console.print(f"  [dim]unchanged, skipping upload: {rel}[/dim]")
    else:
        url = f"https://blob.vercel-storage.com/{rel}?access=public"
        req = urllib.request.Request(url, data=abs_result.read_bytes(), method="PUT", headers={
            "Authorization": f"Bearer {token}",
            "x-api-version": "7",
            "x-content-type": "application/octet-stream",
            "Content-Type": "application/octet-stream",
        })
        try:
            with urllib.request.urlopen(req):
                pass
            blob_manifest[rel] = sha
            blob_manifest_path.write_text(json.dumps(blob_manifest, indent=2))
            console.print(f"  [green]✓[/green] uploaded {rel}")
        except Exception as e:
            console.print(f"  [red]✗[/red] upload failed: {e}")
            raise typer.Exit(1)


@app.command("unpublish-results")
def unpublish_results(
    result: Path = typer.Argument(..., help="Path to a compressed result file (.json.gz), e.g. outputs/personamem/mem0/rag/32k.json.gz"),
    push: bool = typer.Option(False, "--push", help="Delete from Vercel Blob as well"),
    token: str = typer.Option(None, "--token", envvar="BLOB_READ_WRITE_TOKEN", help="Vercel Blob token (or set BLOB_READ_WRITE_TOKEN)"),
) -> None:
    """Decompress a result back to .json and optionally remove it from Vercel Blob.

    Workflow:
        uv run omb unpublish-results outputs/personamem/mem0/rag/32k.json.gz --push
        git add outputs/ results-manifest.json && git commit -m 'results: remove ...' && git push
    """
    import gzip as _gzip
    import urllib.request

    root = Path(__file__).parents[2]
    abs_result = (root / result).resolve() if not result.is_absolute() else result.resolve()

    if not abs_result.exists():
        console.print(f"[red]File not found: {abs_result}[/red]")
        raise typer.Exit(1)

    if not str(abs_result).endswith(".json.gz"):
        console.print(f"[red]Expected a .json.gz file, got: {abs_result.name}[/red]")
        raise typer.Exit(1)

    # ── Decompress ────────────────────────────────────────────────────────
    try:
        with _gzip.open(abs_result, "rt") as fh:
            data = json.load(fh)
        json_path = abs_result.with_suffix("").with_suffix(".json") if abs_result.name.endswith(".json.gz") else abs_result.with_suffix(".json")
        json_path.write_text(json.dumps(data, indent=2))
        abs_result.unlink()
        console.print(f"  [green]✓[/green] decompressed → {json_path.name}")
    except Exception as e:
        console.print(f"  [red]✗[/red] decompression failed: {e}")
        raise typer.Exit(1)

    # ── Regenerate results-manifest.json ─────────────────────────────────
    abs_output = abs_result
    for _ in range(4):
        abs_output = abs_output.parent

    from .server import _list_results
    import os as _os
    _os.environ["OMB_OUTPUT_DIR"] = str(abs_output)
    _os.environ["OMB_ROOT"] = str(root)
    manifest_entries = _list_results(published_only=True)
    manifest_out = root / "results-manifest.json"
    manifest_out.write_text(json.dumps(manifest_entries, indent=2))
    console.print(f"  [green]✓[/green] results-manifest.json ({len(manifest_entries)} entries)")

    if not push:
        console.print("[dim]Skipping Blob deletion (use --push to delete from Blob).[/dim]")
        return

    # ── Delete from Vercel Blob ───────────────────────────────────────────
    if not token:
        console.print("[red]BLOB_READ_WRITE_TOKEN not set. Pass --token or add it to .env.[/red]")
        raise typer.Exit(1)

    rel = str(abs_result.relative_to(root))
    url = f"https://blob.vercel-storage.com/{rel}"
    req = urllib.request.Request(url, method="DELETE", headers={
        "Authorization": f"Bearer {token}",
        "x-api-version": "7",
    })
    try:
        with urllib.request.urlopen(req):
            pass
        console.print(f"  [green]✓[/green] deleted from Blob: {rel}")
    except Exception as e:
        console.print(f"  [red]✗[/red] Blob deletion failed: {e}")
        raise typer.Exit(1)

    # Remove from local blob manifest
    blob_manifest_path = root / ".blob_manifest.json"
    if blob_manifest_path.exists():
        blob_manifest = json.loads(blob_manifest_path.read_text())
        blob_manifest.pop(rel, None)
        blob_manifest_path.write_text(json.dumps(blob_manifest, indent=2))


@app.command("publish-dataset")
def publish_dataset(
    dataset: str = typer.Option(..., "--dataset", "-d", help=f"Dataset to publish. Available: {', '.join(DATASET_REGISTRY)}"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    push: bool = typer.Option(False, "--push", help="Upload to Vercel Blob after export"),
    token: str = typer.Option(None, "--token", envvar="BLOB_READ_WRITE_TOKEN", help="Vercel Blob token (or set BLOB_READ_WRITE_TOKEN)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-export and re-upload even if unchanged"),
) -> None:
    """Export a dataset to data/ and optionally upload to Vercel Blob.

    Idempotent: skips splits already exported unless --force is set.
    Blob upload skips files whose content hasn't changed (checksum-based).

    Workflow (first time or after dataset update):
        uv run omb publish-dataset --dataset personamem --push
        git add data/ && git commit -m 'data: export personamem' && git push
    """
    import gzip as _gzip
    import hashlib
    import urllib.request

    root = Path(__file__).parents[2]
    abs_data = (root / data_dir).resolve()

    try:
        ds = get_dataset(dataset)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # ── Export ────────────────────────────────────────────────────────────
    info_path = abs_data / dataset / "info.json.gz"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with _gzip.open(info_path, "wt", compresslevel=9) as fh:
        json.dump({"links": ds.links}, fh)

    for split in ds.splits:
        split_dir = abs_data / dataset / split
        split_dir.mkdir(parents=True, exist_ok=True)
        required = ["stats.json.gz", "queries.json.gz", "documents.json.gz"]
        if not force and all((split_dir / f).exists() for f in required):
            console.print(f"[dim]{dataset}/{split} — already exported, skipping[/dim]")
            continue

        console.print(f"[bold]{dataset}/{split}[/bold]")
        try:
            with console.status("  computing stats…"):
                stats = ds.split_stats(split)
            with _gzip.open(split_dir / "stats.json.gz", "wt", compresslevel=9) as fh:
                json.dump(stats, fh)
            console.print("  [green]✓[/green] stats")
        except Exception as e:
            console.print(f"  [red]✗ stats: {e}[/red]")

        try:
            with console.status("  loading queries…"):
                queries = ds.load_queries(split)
            qs = [{"id": q.id, "query": q.query, "gold_answers": q.gold_answers,
                   "gold_ids": q.gold_ids, "user_id": q.user_id, "meta": q.meta}
                  for q in queries]
            with _gzip.open(split_dir / "queries.json.gz", "wt", compresslevel=9) as fh:
                json.dump(qs, fh)
            cats = ds.categories(split)
            if cats:
                with console.status("  building category index…"):
                    cat_index = {cat: [q.id for q in ds.load_queries(split, category=cat)] for cat in cats}
                with _gzip.open(split_dir / "categories.json.gz", "wt", compresslevel=9) as fh:
                    json.dump(cat_index, fh)
            console.print(f"  [green]✓[/green] {len(qs)} queries" + (f" · {len(cats)} categories" if cats else ""))
        except Exception as e:
            console.print(f"  [red]✗ queries: {e}[/red]")

        try:
            with console.status("  loading documents…"):
                docs = ds.load_documents(split)
            docs_list = [{"id": d.id, "content": d.content, "user_id": d.user_id, "timestamp": d.timestamp}
                          for d in docs]
            with _gzip.open(split_dir / "documents.json.gz", "wt", compresslevel=9) as fh:
                json.dump(docs_list, fh)
            console.print(f"  [green]✓[/green] {len(docs_list)} documents")
        except Exception as e:
            console.print(f"  [red]✗ documents: {e}[/red]")

    # ── Regenerate catalog.json ───────────────────────────────────────────
    import os as _os
    _os.environ.setdefault("OMB_DATA_DIR", str(abs_data))
    _os.environ.setdefault("OMB_ROOT", str(root))
    from .server import _generate_catalog
    catalog_out = root / "catalog.json"
    catalog_out.write_text(json.dumps(_generate_catalog(), indent=2))
    console.print(f"  [green]✓[/green] catalog.json updated")

    if not push:
        console.print("\n[dim]Skipping Blob upload (use --push to upload).[/dim]")
        return

    # ── Upload to Vercel Blob ─────────────────────────────────────────────
    console.rule("Uploading to Vercel Blob")
    if not token:
        console.print("[red]BLOB_READ_WRITE_TOKEN not set. Pass --token or add it to .env.[/red]")
        raise typer.Exit(1)

    manifest_path = root / ".blob_manifest.json"
    manifest: dict[str, str] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    files = sorted((abs_data / dataset).rglob("*.json.gz"))
    uploaded = skipped = errors = 0
    for f in files:
        rel = str(f.relative_to(root))
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if not force and manifest.get(rel) == sha:
            skipped += 1
            continue
        url = f"https://blob.vercel-storage.com/{rel}?access=public"
        req = urllib.request.Request(url, data=f.read_bytes(), method="PUT", headers={
            "Authorization": f"Bearer {token}",
            "x-api-version": "7",
            "x-content-type": "application/octet-stream",
            "Content-Type": "application/octet-stream",
        })
        try:
            with urllib.request.urlopen(req):
                pass
            manifest[rel] = sha
            console.print(f"  [green]✓[/green] {rel}")
            uploaded += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {rel}: {e}")
            errors += 1

    manifest_path.write_text(json.dumps(manifest, indent=2))
    console.print(f"\n[bold]{uploaded} uploaded, {skipped} unchanged, {errors} errors.[/bold]")


@app.command("unpublish-dataset")
def unpublish_dataset(
    dataset: str = typer.Argument(..., help=f"Dataset to unpublish. Available: {', '.join(DATASET_REGISTRY)}"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    push: bool = typer.Option(False, "--push", help="Delete from Vercel Blob as well"),
    token: str = typer.Option(None, "--token", envvar="BLOB_READ_WRITE_TOKEN", help="Vercel Blob token (or set BLOB_READ_WRITE_TOKEN)"),
) -> None:
    """Remove a dataset's exported files from data/ and optionally from Vercel Blob.

    Workflow:
        uv run omb unpublish-dataset ama-bench --push
        git add data/ && git commit -m 'data: remove ama-bench' && git push
    """
    import shutil
    import urllib.request

    root = Path(__file__).parents[2]
    abs_data = (root / data_dir).resolve()
    dataset_dir = abs_data / dataset

    if not dataset_dir.exists():
        console.print(f"[yellow]No local data found for {dataset!r} at {dataset_dir}[/yellow]")
    else:
        # Collect files before deleting so we can remove them from Blob
        local_files = sorted(dataset_dir.rglob("*.json.gz"))
        shutil.rmtree(dataset_dir)
        console.print(f"  [green]✓[/green] removed {dataset_dir.relative_to(root)} ({len(local_files)} files)")

        if push:
            if not token:
                console.print("[red]BLOB_READ_WRITE_TOKEN not set. Pass --token or add it to .env.[/red]")
                raise typer.Exit(1)

            console.rule("Deleting from Vercel Blob")
            blob_manifest_path = root / ".blob_manifest.json"
            blob_manifest: dict[str, str] = json.loads(blob_manifest_path.read_text()) if blob_manifest_path.exists() else {}

            deleted = errors = 0
            for f in local_files:
                rel = str(f.relative_to(root))
                url = f"https://blob.vercel-storage.com/{rel}"
                req = urllib.request.Request(url, method="DELETE", headers={
                    "Authorization": f"Bearer {token}",
                    "x-api-version": "7",
                })
                try:
                    with urllib.request.urlopen(req):
                        pass
                    blob_manifest.pop(rel, None)
                    console.print(f"  [green]✓[/green] deleted {rel}")
                    deleted += 1
                except Exception as e:
                    console.print(f"  [red]✗[/red] {rel}: {e}")
                    errors += 1

            blob_manifest_path.write_text(json.dumps(blob_manifest, indent=2))
            console.print(f"\n[bold]{deleted} deleted, {errors} errors.[/bold]")
        else:
            console.print("[dim]Skipping Blob deletion (use --push to delete from Blob).[/dim]")

    # ── Regenerate catalog.json ───────────────────────────────────────────
    import os as _os
    _os.environ["OMB_DATA_DIR"] = str(abs_data)
    _os.environ["OMB_ROOT"] = str(root)
    from .server import _generate_catalog
    catalog_out = root / "catalog.json"
    catalog_out.write_text(json.dumps(_generate_catalog(), indent=2))
    console.print(f"  [green]✓[/green] catalog.json updated")


@app.command()
def compress(
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
    keep: bool = typer.Option(False, "--keep", help="Keep original .json files after compression"),
) -> None:
    """Compress result files for git: strips raw_response, gzips to .json.gz.

    Run this before pushing results. The viewer transparently serves .json.gz
    when the original .json is absent.
    """
    import gzip as _gzip

    root = Path(__file__).parents[2]
    abs_output = (root / output_dir).resolve()

    if not abs_output.exists():
        typer.echo(f"Output directory not found: {abs_output}", err=True)
        raise typer.Exit(1)

    files = sorted(f for f in abs_output.rglob("*.json")
                   if len(f.relative_to(abs_output).parts) == 4
                   and not f.name.endswith(".bak"))

    if not files:
        console.print("[yellow]No result files found.[/yellow]")
        return

    table = Table(title=f"Compressed results in {output_dir}")
    table.add_column("File", style="dim")
    table.add_column("Original", justify="right")
    table.add_column("Compressed", justify="right")
    table.add_column("Ratio", justify="right")

    for f in files:
        try:
            data = json.loads(f.read_text())
            results = data.pop("results", [])
            for r in results:
                r.pop("raw_response", None)
            # Inject/refresh top-level aggregate stats so the manifest scanner picks them up
            # (written before 'results' so they appear in the first 512 bytes of the gz file)
            rec_times  = [r["retrieve_time_ms"] for r in results if r.get("retrieve_time_ms") is not None]
            ctx_tokens = [r["context_tokens"]   for r in results if r.get("context_tokens")   is not None]
            data["avg_retrieve_time_ms"] = round(sum(rec_times)  / len(rec_times),  1) if rec_times  else None
            data["avg_context_tokens"]   = round(sum(ctx_tokens) / len(ctx_tokens), 1) if ctx_tokens else None
            data["results"] = results
            gz_path = f.with_suffix(".json.gz")
            with _gzip.open(gz_path, "wt", compresslevel=9) as fh:
                json.dump(data, fh)
            orig_mb = f.stat().st_size / 1024 / 1024
            gz_mb = gz_path.stat().st_size / 1024 / 1024
            table.add_row(
                str(f.relative_to(abs_output)),
                f"{orig_mb:.1f} MB",
                f"{gz_mb:.1f} MB",
                f"{gz_mb / orig_mb:.1%}",
            )
            if not keep:
                f.unlink()
        except Exception as e:
            console.print(f"[red]Error compressing {f}: {e}[/red]")

    console.print(table)
    if not keep:
        console.print("[dim]Original .json files removed. Run with --keep to retain them.[/dim]")


@app.command("export-data")
def export_data(
    dataset: str = typer.Option(None, "--dataset", "-d", help=f"Dataset to export. Available: {', '.join(DATASET_REGISTRY)}. Default: all published."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Output directory for exported data files"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-export even if files already exist"),
) -> None:
    """Download and export published dataset files to gzipped JSON for deployment.

    By default exports all datasets marked published=True and skips splits
    that already have all files. Run with --force to re-export everything.
    The generated files are committed to git alongside the result outputs.
    """
    import gzip as _gzip

    root = Path(__file__).parents[2]
    abs_data = (root / data_dir).resolve()

    if dataset:
        names = [dataset]
    else:
        names = [n for n, cls in DATASET_REGISTRY.items() if cls.published]

    for ds_name in names:
        try:
            ds = get_dataset(ds_name)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            continue

        # Dataset-level info (links)
        info_path = abs_data / ds_name / "info.json.gz"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with _gzip.open(info_path, "wt", compresslevel=9) as fh:
            json.dump({"links": ds.links}, fh)

        for split in ds.splits:
            split_dir = abs_data / ds_name / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # Skip if all files already exist and --force not set
            required = ["stats.json.gz", "queries.json.gz", "documents.json.gz"]
            if not force and all((split_dir / f).exists() for f in required):
                console.print(f"[dim]{ds_name}/{split} — already exported, skipping[/dim]")
                continue

            console.print(f"[bold]{ds_name}/{split}[/bold]")

            # Stats
            try:
                with console.status("  computing stats…"):
                    stats = ds.split_stats(split)
                with _gzip.open(split_dir / "stats.json.gz", "wt", compresslevel=9) as fh:
                    json.dump(stats, fh)
                console.print(f"  [green]✓[/green] stats")
            except Exception as e:
                console.print(f"  [red]✗ stats: {e}[/red]")

            # Queries (all + per-category index)
            try:
                with console.status("  loading queries…"):
                    queries = ds.load_queries(split)
                qs = [{"id": q.id, "query": q.query, "gold_answers": q.gold_answers,
                       "gold_ids": q.gold_ids, "user_id": q.user_id, "meta": q.meta}
                      for q in queries]
                with _gzip.open(split_dir / "queries.json.gz", "wt", compresslevel=9) as fh:
                    json.dump(qs, fh)
                cats = ds.categories(split)
                if cats:
                    with console.status("  building category index…"):
                        cat_index = {cat: [q.id for q in ds.load_queries(split, category=cat)]
                                     for cat in cats}
                    with _gzip.open(split_dir / "categories.json.gz", "wt", compresslevel=9) as fh:
                        json.dump(cat_index, fh)
                console.print(f"  [green]✓[/green] {len(qs)} queries" +
                               (f" · {len(cats)} categories" if cats else ""))
            except Exception as e:
                console.print(f"  [red]✗ queries: {e}[/red]")

            # Documents
            try:
                with console.status("  loading documents…"):
                    docs = ds.load_documents(split)
                docs_list = [{"id": d.id, "content": d.content,
                               "user_id": d.user_id, "timestamp": d.timestamp}
                              for d in docs]
                with _gzip.open(split_dir / "documents.json.gz", "wt", compresslevel=9) as fh:
                    json.dump(docs_list, fh)
                console.print(f"  [green]✓[/green] {len(docs_list)} documents")
            except Exception as e:
                console.print(f"  [red]✗ documents: {e}[/red]")


@app.command("upload-blob")
def upload_blob(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Local data directory to upload"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Only upload a specific dataset"),
    token: str = typer.Option(None, "--token", envvar="BLOB_READ_WRITE_TOKEN", help="Vercel Blob read-write token"),
) -> None:
    """Upload data/ files to Vercel Blob for public deployment.

    Run this after 'omb export-data' to make dataset files available on Vercel.
    Requires BLOB_READ_WRITE_TOKEN env var or --token option.
    """
    import urllib.request

    if not token:
        console.print("[red]Error: BLOB_READ_WRITE_TOKEN not set. Pass --token or set the env var.[/red]")
        raise typer.Exit(1)

    root = Path(__file__).parents[2]
    abs_data = (root / data_dir).resolve()

    if not abs_data.exists():
        console.print(f"[red]Data directory not found: {abs_data}[/red]")
        raise typer.Exit(1)

    files = sorted(abs_data.rglob("*.json.gz"))
    if dataset:
        files = [f for f in files if f.relative_to(abs_data).parts[0] == dataset]

    if not files:
        console.print("[yellow]No .json.gz files found to upload.[/yellow]")
        return

    success = 0
    for f in files:
        rel = str(f.relative_to(root))  # e.g. data/personamem/32k/docs.json.gz
        url = f"https://blob.vercel-storage.com/{rel}?access=public"
        req = urllib.request.Request(
            url,
            data=f.read_bytes(),
            method="PUT",
            headers={
                "Authorization": f"Bearer {token}",
                "x-api-version": "7",
                "x-content-type": "application/octet-stream",
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(req):
                pass
            console.print(f"  [green]✓[/green] {rel}")
            success += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {rel}: {e}")

    console.print(f"\n[bold]{success}/{len(files)} files uploaded.[/bold]")


@app.command()
def view(
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
    port: int = typer.Option(7979, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto-reload server on code changes"),
    dev: bool = typer.Option(False, "--dev", help="Start Vite dev server alongside the API (hot-reload UI)"),
) -> None:
    """Launch the result viewer in the browser. Watches for new results automatically."""
    import subprocess
    import uvicorn
    root = Path(__file__).parents[2]
    abs_output = (root / output_dir).resolve()

    os.environ["OMB_OUTPUT_DIR"] = str(abs_output)
    os.environ["OMB_ROOT"] = str(root)

    vite_proc = None
    if dev:
        ui_dir = root / "ui"
        if not ui_dir.exists():
            console.print("[red]ui/ directory not found. Cannot start dev server.[/red]")
            raise typer.Exit(1)
        vite_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(ui_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = "http://localhost:5173"
        console.print(f"[green]Viewer → {url}[/green]  (Vite dev server + FastAPI on :{port})")
        console.print("[dim]  Vite hot-reload enabled.[/dim]")
    else:
        url = f"http://localhost:{port}"
        console.print(f"[green]Viewer → {url}[/green]  (Ctrl+C to stop)")
        if reload:
            console.print("[dim]  Hot-reload enabled.[/dim]")

    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(
            "memory_bench.server:app",
            host="0.0.0.0",
            port=port,
            reload=reload,
            reload_dirs=[str(Path(__file__).parent)] if reload else None,
            log_level="warning",
        )
    finally:
        if vite_proc is not None:
            vite_proc.terminate()
