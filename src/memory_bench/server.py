"""FastAPI server for the OMB result viewer."""
import gzip as _gzip
import json as _json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

_root = Path(os.environ.get("OMB_ROOT", Path(__file__).parents[2]))
_output_dir = Path(os.environ.get("OMB_OUTPUT_DIR", _root / "outputs"))
_data_dir = Path(os.environ.get("OMB_DATA_DIR", _root / "data"))
_ui_dist = _root / "ui" / "dist"

# Optional Vercel Blob base URL for serving data/ files when not on disk
_BLOB_BASE = os.environ.get(
    "OMB_BLOB_BASE_URL",
    "https://l4cy6iaq2c4g2ldt.public.blob.vercel-storage.com",
)


def _fetch_blob(relative_path: str) -> bytes:
    """Fetch a file from Vercel Blob and cache it in /tmp."""
    import urllib.request
    cache_path = Path(tempfile.gettempdir()) / "omb_blob" / relative_path
    if cache_path.exists():
        return cache_path.read_bytes()
    url = f"{_BLOB_BASE}/{relative_path}"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data

# Serve built UI assets
if (_ui_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_ui_dist / "assets")), name="ui-assets")


_results_cache: list[dict] | None = None
_results_cache_mtime: dict[str, float] = {}


def _list_results(published_only: bool = False) -> list[dict]:
    import json as _json

    global _results_cache, _results_cache_mtime

    # Use pre-generated static manifest if outputs/ is missing or empty (Vercel deployment)
    static = _root / "results-manifest.json"
    has_outputs = _output_dir.exists() and any(_output_dir.rglob("*.json*"))
    if not has_outputs:
        return _json.loads(static.read_text(encoding="utf-8")) if static.exists() else []

    # Collect current files and their mtimes (.json and .json.gz)
    if not _output_dir.exists():
        return []
    gz_files = set(f for f in _output_dir.rglob("*.json.gz")
                   if len(f.relative_to(_output_dir).parts) == 4)
    if published_only:
        # Only include committed/compressed results (.json.gz); ignore raw .json
        files = sorted(gz_files)
    else:
        json_files = set(f for f in _output_dir.rglob("*.json")
                         if len(f.relative_to(_output_dir).parts) == 4)
        # Prefer .json over .json.gz if both exist; otherwise include .json.gz
        files = sorted(json_files | {f for f in gz_files if f.with_suffix("") not in json_files})
    current_mtime = {str(f): f.stat().st_mtime for f in files}

    # Return cached result if nothing changed
    if _results_cache is not None and current_mtime == _results_cache_mtime:
        return _results_cache

    import gzip as _gzip
    entries = []
    for f in files:
        is_gz = f.name.endswith(".json.gz")
        parts = f.relative_to(_output_dir).parts
        run_name = parts[1]
        # Read only first 512 bytes to extract memory_provider cheaply
        try:
            raw = f.read_bytes()
            snippet_bytes = _gzip.decompress(raw)[:512] if is_gz else raw[:512]
            snippet = snippet_bytes.decode("utf-8", errors="ignore")
            import re
            def _extract(key):
                m = re.search(r'"' + key + r'"\s*:\s*([^,}\n]+)', snippet)
                return m.group(1).strip().strip('"') if m else None
            memory_provider = _extract("memory_provider") or run_name
            total_queries = _extract("total_queries")
            correct = _extract("correct")
            accuracy = _extract("accuracy")
            ingestion_time_ms = _extract("ingestion_time_ms")
            ingested_docs = _extract("ingested_docs")
            avg_retrieve_time_ms = _extract("avg_retrieve_time_ms")
            avg_context_tokens = _extract("avg_context_tokens")
            category = _extract("category")
        except Exception:
            memory_provider = run_name
            total_queries = correct = accuracy = ingestion_time_ms = ingested_docs = category = None
        split_name = parts[3].removesuffix(".json.gz").removesuffix(".json")
        # Path exposed to the viewer always uses .json (server serves .gz transparently)
        json_path = str(f.relative_to(_root).with_name(split_name + ".json")) if is_gz else str(f.relative_to(_root))
        entries.append({
            "path": json_path,
            "dataset": parts[0],
            "run_name": run_name,
            "memory": memory_provider,
            "mode": parts[2],
            "split": split_name,
            "total_queries": int(total_queries) if total_queries and total_queries != "null" else None,
            "correct": int(correct) if correct and correct != "null" else None,
            "accuracy": float(accuracy) if accuracy and accuracy != "null" else None,
            "ingestion_time_ms": float(ingestion_time_ms) if ingestion_time_ms and ingestion_time_ms != "null" else None,
            "ingested_docs": int(ingested_docs) if ingested_docs and ingested_docs != "null" else None,
            "avg_retrieve_time_ms": float(avg_retrieve_time_ms) if avg_retrieve_time_ms and avg_retrieve_time_ms != "null" else None,
            "avg_context_tokens": float(avg_context_tokens) if avg_context_tokens and avg_context_tokens != "null" else None,
            "category": category if category and category != "null" else None,
        })

    _results_cache = entries
    _results_cache_mtime = current_mtime
    return entries


@lru_cache(maxsize=256)
def _load_data_file(dataset: str, split: str, name: str):
    """Load a gzipped JSON file from the data/ directory, falling back to Blob."""
    path = _data_dir / dataset / split / f"{name}.json.gz"
    if path.exists():
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            return _json.load(f)
    # Fall back to Vercel Blob
    relative = f"data/{dataset}/{split}/{name}.json.gz"
    try:
        raw = _fetch_blob(relative)
        return _json.loads(_gzip.decompress(raw))
    except Exception as exc:
        raise FileNotFoundError(
            f"Data file missing locally and from Blob: {relative}\n"
            f"Run 'omb export-data --dataset {dataset}' to generate it."
        ) from exc


@lru_cache(maxsize=64)
def _load_dataset_info_cached(dataset: str) -> dict:
    path = _data_dir / dataset / "info.json.gz"
    if path.exists():
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            return _json.load(f)
    relative = f"data/{dataset}/info.json.gz"
    try:
        raw = _fetch_blob(relative)
        return _json.loads(_gzip.decompress(raw))
    except Exception as exc:
        raise FileNotFoundError(
            f"Dataset info missing locally and from Blob: {relative}"
        ) from exc


@lru_cache(maxsize=64)
def _split_stats_cached(dataset: str, split: str) -> dict:
    try:
        return _load_data_file(dataset, split, "stats")
    except Exception as e:
        return {"error": str(e)}


@lru_cache(maxsize=64)
def _load_queries_cached(dataset: str, split: str, category: str = "") -> list[dict]:
    queries = _load_data_file(dataset, split, "queries")
    if category:
        try:
            cat_index = _load_data_file(dataset, split, "categories")
            ids = set(cat_index.get(category, []))
            queries = [q for q in queries if q["id"] in ids]
        except FileNotFoundError:
            queries = []
    return queries


@lru_cache(maxsize=64)
def _load_documents_cached(dataset: str, split: str) -> dict:
    """Load all documents for a split and return as {id: doc_dict} map."""
    docs = _load_data_file(dataset, split, "documents")
    return {d["id"]: d for d in docs}


_DOC_PREVIEW_CHARS = 300


def _truncate_doc(doc: dict) -> dict:
    """Return doc with content truncated for list views."""
    content = doc.get("content", "")
    return {**doc, "content": content[:_DOC_PREVIEW_CHARS] + ("…" if len(content) > _DOC_PREVIEW_CHARS else "")}



@app.on_event("startup")
async def _prewarm_caches():
    import threading

    def _warm():
        seen: set[tuple[str, str]] = set()
        for entry in _list_results():
            key = (entry["dataset"], entry["split"])
            if key not in seen:
                seen.add(key)
                try:
                    _split_stats_cached(entry["dataset"], entry["split"])
                    _load_queries_cached(entry["dataset"], entry["split"])
                    _load_documents_cached(entry["dataset"], entry["split"])
                except Exception:
                    pass

    threading.Thread(target=_warm, daemon=True).start()


def _generate_catalog() -> dict:
    from .dataset import REGISTRY as DS_REGISTRY
    from .memory import REGISTRY as MEM_REGISTRY
    from .modes import REGISTRY as MODE_REGISTRY

    def _task_label(task_type: str) -> str:
        return "MCQ" if task_type == "mcq" else "LLM-judged"

    datasets = {
        name: {
            "description": cls.description,
            "task": _task_label(cls.task_type),
            "splits": cls.splits,
        }
        for name, cls in DS_REGISTRY.items()
        if (_data_dir / name).exists()
    }
    providers: dict = {}
    for key, cls in MEM_REGISTRY.items():
        family = cls.provider or cls.name
        if cls.variant is None:
            providers[family] = {
                "key": key,
                "description": cls.description,
                "kind": cls.kind,
                "link": cls.link,
                "logo": cls.logo,
            }
        else:
            if family not in providers:
                providers[family] = {
                    "link": cls.link,
                    "logo": cls.logo,
                    "variants": {},
                }
            providers[family]["variants"][cls.variant] = {
                "key": key,
                "description": cls.description,
                "kind": cls.kind,
            }
    modes = {
        name: {"description": cls.description}
        for name, cls in MODE_REGISTRY.items()
    }
    return {"datasets": datasets, "providers": providers, "modes": modes}


@app.get("/api/catalog")
def catalog():
    # Use pre-generated static catalog if available (needed for deployments without full deps)
    static = _root / "catalog.json"
    if static.exists():
        return JSONResponse(_json.loads(static.read_text(encoding="utf-8")))
    return JSONResponse(_generate_catalog())


@app.get("/api/results")
def results():
    return JSONResponse(_list_results())


@app.get("/api/split-category-breakdown")
def split_category_breakdown(dataset: str, split: str):
    """Compute per-run per-category accuracy breakdown for a dataset/split."""
    import json as _json

    entries = [e for e in _list_results() if e["dataset"] == dataset and e["split"] == split]
    if not entries:
        return JSONResponse([])

    out = []
    for entry in entries:
        path = _root / entry["path"]
        # Try .gz variant if plain .json not found
        if not path.exists():
            path = path.with_suffix(".json.gz")
        if not path.exists():
            continue
        try:
            raw = path.read_bytes()
            data = _json.loads(_gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        except Exception:
            continue

        results_list = data.get("results") or []
        if not results_list:
            continue

        # Determine breakdown strategy
        # Strategy 1: top-level single category per file (already in manifest)
        top_cat = entry.get("category")
        if top_cat and "," not in top_cat:
            # Single-category file — already handled by the manifest view
            continue

        # Strategy 2: per-result category_axes
        # Find axis with cardinality 2-30 (exclude high-cardinality ID axes)
        axes: dict[str, dict] = {}  # axis -> { value -> {correct, total} }
        for r in results_list:
            cat_axes = r.get("category_axes") or {}
            if not isinstance(cat_axes, dict):
                continue
            for axis, vals in cat_axes.items():
                if not isinstance(vals, list):
                    vals = [vals]
                for v in vals:
                    if v is None:
                        continue
                    v = str(v)
                    axes.setdefault(axis, {}).setdefault(v, {"correct": 0, "total": 0})
                    axes[axis][v]["total"] += 1
                    if r.get("correct"):
                        axes[axis][v]["correct"] += 1

        # Pick best axis: cardinality 2-30, prefer lower cardinality
        chosen_axis = None
        for axis, vals in sorted(axes.items(), key=lambda x: len(x[1])):
            if 2 <= len(vals) <= 30:
                chosen_axis = axis
                break

        if not chosen_axis:
            continue

        categories = {
            v: round(s["correct"] / s["total"], 4) if s["total"] else None
            for v, s in axes[chosen_axis].items()
        }
        out.append({
            "run_name": entry["run_name"],
            "memory": entry["memory"],
            "mode": entry["mode"],
            "path": entry["path"],
            "axis": chosen_axis,
            "categories": categories,
        })

    return JSONResponse(out)


@app.get("/api/external-results")
def external_results():
    path = _root / "external_results.json"
    if not path.exists():
        return JSONResponse({})
    try:
        import json as _json
        return JSONResponse(_json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/dataset/{name}/info")
def dataset_info(name: str):
    try:
        return JSONResponse(_load_dataset_info_cached(name))
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/split-stats")
def split_stats(dataset: str = "", split: str = ""):
    return JSONResponse(_split_stats_cached(dataset, split))


@app.get("/api/dataset/{name}/{split}/queries")
def dataset_queries(name: str, split: str, search: str = "", category: str = "", limit: int = 50, offset: int = 0, expand_docs: bool = False):
    try:
        queries = _load_queries_cached(name, split, category)
        if search:
            s = search.lower()
            queries = [q for q in queries
                       if s in q["id"].lower() or s in q["query"].lower()
                       or any(s in a.lower() for a in q["gold_answers"])]
        total = len(queries)
        page = queries[offset: offset + limit]
        if expand_docs:
            docs_map = _load_documents_cached(name, split)
            items = []
            for q in page:
                item = dict(q)
                item["gold_docs"] = [docs_map[gid] for gid in q["gold_ids"] if gid in docs_map]
                items.append(item)
        else:
            items = [dict(q) for q in page]
        return JSONResponse({"total": total, "items": items})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/dataset/{name}/{split}/documents")
def dataset_documents(name: str, split: str, search: str = "", category: str = "", limit: int = 20, offset: int = 0, full: bool = False):
    try:
        docs_map = _load_documents_cached(name, split)
        docs = list(docs_map.values())
        if search:
            s = search.lower()
            docs = [d for d in docs if s in d["content"].lower()]
        total = len(docs)
        page = docs[offset: offset + limit]
        if not full:
            page = [_truncate_doc(d) for d in page]
        return JSONResponse({"total": total, "items": page})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/dataset/{name}/{split}/documents/{doc_id}")
def dataset_document(name: str, split: str, doc_id: str):
    try:
        docs_map = _load_documents_cached(name, split)
        doc = docs_map.get(doc_id)
        if doc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(doc)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/")
def root():
    index = _ui_dist / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "UI not built. Run: cd ui && npm install && npm run build"}, status_code=503)


@app.get("/{file_path:path}")
def serve_file(file_path: str):
    from fastapi.responses import Response
    # Serve output JSON files (transparently handles .json.gz, falls back to Blob)
    if file_path.startswith("outputs/"):
        target = _root / file_path
        if target.exists() and target.is_file():
            return FileResponse(str(target), headers={"Cache-Control": "no-cache"})
        gz_path = file_path if file_path.endswith(".gz") else file_path + ".gz"
        gz_target = _root / gz_path
        if gz_target.exists() and gz_target.is_file():
            return FileResponse(
                str(gz_target),
                media_type="application/json",
                headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
            )
        # Fall back to Vercel Blob
        blob_path = gz_path  # always fetch .gz from blob
        try:
            data = _fetch_blob(blob_path)
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
            )
        except Exception:
            pass
        return JSONResponse({"error": "not found"}, status_code=404)
    # Serve static UI files (e.g. favicon, vite manifests)
    ui_target = _ui_dist / file_path
    if ui_target.exists() and ui_target.is_file():
        return FileResponse(str(ui_target))
    # SPA fallback: return index.html for all other routes
    index = _ui_dist / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "not found"}, status_code=404)
