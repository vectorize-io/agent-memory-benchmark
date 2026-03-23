import asyncio
import os
import time
from pathlib import Path

from ..models import Document
from .base import MemoryProvider


def _deduplicate_results(results):
    """Remove duplicate results by chunk_id, keeping first occurrence."""
    seen = set()
    out = []
    for r in results:
        key = r.chunk_id if r.chunk_id else r.id
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _format_result(r, chunks: dict | None = None, seen_chunk_ids: set | None = None) -> str:
    lines = []
    if r.type:
        lines.append(f"**[{r.type}]** {r.text}")
    else:
        lines.append(r.text)

    meta = []
    date_start = r.occurred_start
    date_end = r.occurred_end
    if date_start and date_end and date_start != date_end:
        meta.append(f"occurred: {date_start} – {date_end}")
    elif date_start:
        meta.append(f"occurred: {date_start}")
    if r.mentioned_at:
        meta.append(f"mentioned: {r.mentioned_at}")
    if r.chunk_id:
        meta.append(f"chunk: {r.chunk_id}")
    if meta:
        lines.append("_" + " · ".join(meta) + "_")

    if chunks and r.chunk_id and r.chunk_id in chunks:
        if seen_chunk_ids is None or r.chunk_id not in seen_chunk_ids:
            lines.append(f"> {chunks[r.chunk_id].text}")
            if seen_chunk_ids is not None:
                seen_chunk_ids.add(r.chunk_id)

    return "\n".join(lines)


def _format_results(results, chunks: dict | None = None) -> list[str]:
    """Format a list of results, inlining each chunk_id's text only on first appearance."""
    seen_chunk_ids: set = set()
    return [_format_result(r, chunks, seen_chunk_ids) for r in results]


def _build_docs(results, chunks: dict | None = None) -> "list[Document]":
    """Build Document list from recall results, inlining chunk text only on first chunk_id."""
    return [Document(id=r.id, content=c) for r, c in zip(results, _format_results(results, chunks))]


def _bank_id_from_store_dir(store_dir: Path) -> tuple[str, str | None, str | None]:
    """Return (bank_id, dataset_name, category) from the store_dir path."""
    parts = store_dir.parts
    try:
        idx = parts.index("_store")
        dataset = parts[idx - 2]
        split = parts[idx + 1]
        category = parts[idx + 2] if idx + 2 < len(parts) else None
        if category == "all":
            category = None
        return f"{dataset}-{split}", dataset, category
    except (ValueError, IndexError):
        return "bench", None, None


class _HindsightBase(MemoryProvider):
    """Shared logic for Hindsight memory providers."""

    def __init__(self):
        self._bank_id = "bench"
        self._dataset: str | None = None
        self._category: str | None = None
        self._default_user_id = "omb-bench-default"
        self._client = None  # set by subclass
        self._async_client = None  # lazily created (cloud only)
        self._per_unit = False

    def _bank_id_for(self, user_id: str | None) -> str:
        if self._per_unit and user_id is not None:
            return f"{self._bank_id}-u{user_id}"
        return self._bank_id

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._bank_id, self._dataset, self._category = _bank_id_from_store_dir(store_dir)
        self._per_unit = unit_ids is not None

    # ── Bank creation (sync) ──────────────────────────────────────────────────

    def _bank_kwargs(self) -> dict:
        return dict(enable_observations=False)

    def _create_bank(self, bank_id: str) -> None:
        kwargs = self._bank_kwargs()
        try:
            self._client.banks.delete(bank_id=bank_id)
        except Exception:
            pass
        self._client.create_bank(bank_id=bank_id, name=f"Benchmark Bank ({bank_id})", **kwargs)

    async def _await_operation(self, client, bank_id: str, operation_id: str, max_wait_s: int = 300) -> None:
        """Poll until an async retain operation completes (5-minute timeout)."""
        from hindsight_client_api.api.operations_api import OperationsApi
        ops_api = OperationsApi(client._api_client)
        waited = 0
        last_status = None
        while waited < max_wait_s:
            try:
                resp = await asyncio.wait_for(
                    ops_api.get_operation_status(bank_id=bank_id, operation_id=operation_id),
                    timeout=30,
                )
                last_status = resp.status
                if last_status in ("completed", "failed"):
                    break
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(1)
            waited += 1
        if waited >= max_wait_s:
            import logging
            logging.getLogger(__name__).warning(
                f"_await_operation timed out after {max_wait_s}s for bank={bank_id} op={operation_id} "
                f"last_status={last_status!r}; continuing anyway."
            )

    # ── Bank creation (async) ─────────────────────────────────────────────────

    async def _acreate_bank(self, client, bank_id: str) -> None:
        kwargs = self._bank_kwargs()
        try:
            await client.adelete_bank(bank_id=bank_id)
        except Exception:
            pass
        await client.acreate_bank(bank_id=bank_id, name=f"Benchmark Bank ({bank_id})", **kwargs)

    # ── Item builders ─────────────────────────────────────────────────────────

    def _doc_to_items(self, doc: Document) -> list[dict]:
        """Convert a Document to a list of retain items."""
        content = doc.content.replace("\x00", "")
        base: dict = {}
        if not self._per_unit:
            base["tags"] = [f"user:{doc.user_id or self._default_user_id}"]
        if doc.timestamp:
            base["timestamp"] = doc.timestamp
        if doc.context:
            base["context"] = doc.context

        return [{**base, "content": content, "document_id": doc.id,
                 "metadata": {"doc_id": doc.id}}]

    # ── Sync ingest (embedded) ────────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        from hindsight_client.hindsight_client import _run_async
        from hindsight_client_api.api.operations_api import OperationsApi

        if not self._per_unit:
            self._create_bank(self._bank_id)

        _BATCH_SIZE = 20
        created: set[str] = set()
        operation_ids: list[tuple[str, str]] = []

        # Collect all items across all documents first, grouped by bank_id,
        # then batch across documents so fewer (larger) operations are created.
        # This makes each operation durable in async_operations and resumable on restart.
        items_by_bank: dict[str, list[dict]] = {}
        for doc in documents:
            bank_id = self._bank_id_for(doc.user_id)
            if self._per_unit and bank_id not in created:
                self._create_bank(bank_id)
                created.add(bank_id)
            items_by_bank.setdefault(bank_id, []).extend(self._doc_to_items(doc))

        for bank_id, all_items in items_by_bank.items():
            # Deduplicate by document_id — the dataset may have sessions with identical IDs.
            seen_doc_ids: set[str] = set()
            unique_items: list[dict] = []
            for item in all_items:
                did = item.get("document_id")
                if did is None or did not in seen_doc_ids:
                    unique_items.append(item)
                    if did is not None:
                        seen_doc_ids.add(did)
            all_items = unique_items

            for i in range(0, len(all_items), _BATCH_SIZE):
                batch = all_items[i:i + _BATCH_SIZE]
                for attempt in range(3):
                    try:
                        self._client.retain_batch(
                            bank_id=bank_id,
                            items=batch,
                            retain_async=True,
                        )
                        break
                    except Exception as e:
                        err = str(e)
                        etype = type(e).__name__
                        if ("duplicate key" in err or "duplicate document_ids" in err
                                or "violates foreign key constraint" in err
                                or "empty response" in err or "Cannot connect" in err
                                or "Timeout" in etype or "Timeout" in err):
                            # Skip: already ingested / duplicate / FK race / LLM/timeout failure / daemon down.
                            break
                        if attempt < 2:
                            time.sleep(10)
                        else:
                            # Last resort: skip any unrecognised transient error rather than killing the run.
                            import logging
                            logging.getLogger(__name__).warning(
                                f"retain_batch unhandled error (skipping batch): {etype}: {err[:200]}"
                            )
                            break

    # ── Recall kwargs ─────────────────────────────────────────────────────────

    def _recall_kwargs(self, query: str, user_id: str | None, query_timestamp: str | None, include_chunks: bool = True, max_chunk_tokens: int | None = None) -> dict:
        is_lifebench = self._dataset == "lifebench"
        is_personamem = self._dataset == "personamem"
        if max_chunk_tokens is None:
            if is_personamem:
                max_chunk_tokens = 10240
            elif is_lifebench:
                max_chunk_tokens = 16384
            else:
                max_chunk_tokens = 32768
        if is_personamem:
            max_tokens = 4096
        elif is_lifebench:
            max_tokens = 16384
        else:
            max_tokens = 32768
        kwargs: dict = {
            "bank_id": self._bank_id_for(user_id),
            "query": query[:1900],
            "budget": "high",
            "max_tokens": max_tokens,
            "include_chunks": include_chunks,
            "include_entities": False,
        }
        if include_chunks:
            kwargs["max_chunk_tokens"] = max_chunk_tokens
        if query_timestamp:
            kwargs["query_timestamp"] = query_timestamp
        if user_id is not None and not self._per_unit:
            kwargs["tags"] = [f"user:{user_id}"]
            kwargs["tags_match"] = "any_strict"
        return kwargs

    def _reflect_kwargs(self, query: str, user_id: str | None, query_timestamp: str | None) -> dict:
        uid = user_id or self._default_user_id
        kwargs: dict = {
            "bank_id": self._bank_id_for(user_id),
            "query": query[:1900],
        }
        if query_timestamp:
            kwargs["query_timestamp"] = query_timestamp
        if user_id is not None and not self._per_unit:
            kwargs["tags"] = [f"user:{uid}"]
            kwargs["tags_match"] = "any_strict"
        return kwargs

    # ── Sync retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        try:
            response = self._client.recall(**self._recall_kwargs(query, user_id, query_timestamp))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"recall failed (returning empty): {e}")
            return [], None
        chunks = response.chunks or {}
        docs = _build_docs(_deduplicate_results(response.results), chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw

    def retrieve_by_steps(self, steps: list[int], query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, compact: bool | None = None) -> tuple[list[Document], dict | None]:
        # Legacy: For small step sets include chunks; for large ranges rely on entity tags.
        include_chunks = len(steps) <= 6
        kwargs = self._recall_kwargs(query, user_id, query_timestamp, include_chunks=include_chunks, max_chunk_tokens=16384)
        if steps:
            kwargs["tags"] = [f"step_number:{s}" for s in steps]
            kwargs["tags_match"] = "any_strict"
        response = self._client.recall(**kwargs)
        chunks = response.chunks or {}
        results = _deduplicate_results(response.results)
        if not self._per_unit and user_id is not None and steps:
            uid_filter = f"user:{user_id}"
            results = [r for r in results if uid_filter in (r.tags or [])]
        docs = _build_docs(results, chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw

    def direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        response = self._client.reflect(**self._reflect_kwargs(query, user_id, query_timestamp))
        answer = response.text or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return answer, answer, raw

    def retrieve_by_tag(self, tag: str, query: str = "", user_id: str | None = None) -> tuple[list[Document], dict | None]:
        kwargs = self._recall_kwargs(query or "relevant information", user_id, None)
        kwargs["tags"] = [tag]
        kwargs["tags_match"] = "any_strict"
        response = self._client.recall(**kwargs)
        chunks = response.chunks or {}
        docs = _build_docs(_deduplicate_results(response.results), chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw


# ── Embedded provider ─────────────────────────────────────────────────────────

class HindsightMemoryProvider(_HindsightBase):
    name = "hindsight"
    description = "Embedded Hindsight fact store using gemini-2.5-flash-lite as the extraction model. Recall uses all memory types (world + experience + observation) with no type filter applied."
    kind = "local"
    provider = "hindsight"
    variant = "local"
    link = "https://hindsight.vectorize.io"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=hindsight.vectorize.io"
    concurrency = 4

    def __init__(self):
        super().__init__()
        self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        super().prepare(store_dir, unit_ids)
        from hindsight import HindsightEmbedded
        self._client = HindsightEmbedded(
            profile=f"omb-{self._bank_id}",
            llm_provider="gemini",
            llm_model="gemini-2.5-flash-lite",
            llm_api_key=self._api_key,
        )
        try:
            self._client.banks.list()
        except Exception:
            pass

    def ingest(self, documents: list[Document]) -> None:
        super().ingest(documents)
        # After sync ingest, _run_async in the hindsight client creates a temporary event loop
        # that may leave an aiohttp session bound to it. Reset so the next async_retrieve
        # (called from asyncio.run()) creates a fresh session on the correct loop.
        try:
            rc = self._client._memory_api.api_client.rest_client
            rc._pool_manager = None
            rc._retry_client = None
        except Exception:
            pass

    async def async_ingest(self, documents: list[Document]) -> None:
        # Close any existing aiohttp session BEFORE running ingest in a thread.
        # ingest → retain_batch → _run_async creates a fresh event loop in the thread;
        # if an open session bound to the main loop exists, its TimerContext.__enter__
        # calls asyncio.current_task(loop=main_loop) from the thread → None → RuntimeError.
        try:
            rc = self._client._memory_api.api_client.rest_client
            if rc._pool_manager is not None:
                await rc._pool_manager.close()
            if rc._retry_client is not None:
                await rc._retry_client.close()
            rc._pool_manager = None
            rc._retry_client = None
        except Exception:
            pass
        await asyncio.to_thread(self.ingest, documents)
        # Reset again after ingest so the next arecall creates a fresh session
        # in the correct (main) event loop rather than reusing the thread's session.
        try:
            rc = self._client._memory_api.api_client.rest_client
            rc._pool_manager = None
            rc._retry_client = None
        except Exception:
            pass

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None):
        kwargs = self._recall_kwargs(query, user_id, query_timestamp)
        response = await self._client.arecall(**kwargs)
        chunks = response.chunks or {}
        docs = _build_docs(_deduplicate_results(response.results), chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw

    async def async_retrieve_by_steps(self, steps: list[int], query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, compact: bool | None = None):
        return await asyncio.to_thread(self.retrieve_by_steps, steps, query, k, user_id, query_timestamp, compact)

    async def async_retrieve_by_tag(self, tag: str, query: str = "", user_id: str | None = None):
        return await asyncio.to_thread(self.retrieve_by_tag, tag, query, user_id)

    async def async_direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None):
        return await asyncio.to_thread(self.direct_answer, query, user_id=user_id, query_timestamp=query_timestamp)


# ── Cloud provider ────────────────────────────────────────────────────────────

class HindsightCloudMemoryProvider(_HindsightBase):
    name = "hindsight-cloud"
    description = "Hindsight hosted cloud API. Recall uses all memory types (world + experience + observation) with no type filter applied."
    kind = "cloud"
    provider = "hindsight"
    variant = "cloud"

    def __init__(self):
        super().__init__()
        from hindsight import HindsightClient
        self._cloud_api_key = os.environ["HINDSIGHT_CLOUD_KEY"]
        self._cloud_base_url = os.environ.get("HINDSIGHT_CLOUD_URL", "https://api.hindsight.vectorize.io")
        self._client = HindsightClient(base_url=self._cloud_base_url, api_key=self._cloud_api_key)

    def _get_async_client(self):
        """Return the shared async client, creating it lazily inside the running event loop."""
        if self._async_client is None:
            from hindsight_client import Hindsight
            self._async_client = Hindsight(base_url=self._cloud_base_url, api_key=self._cloud_api_key)
        return self._async_client

    async def async_ingest(self, documents: list[Document]) -> None:
        client = self._get_async_client()

        if not self._per_unit:
            await self._acreate_bank(client, self._bank_id)

        _BATCH_SIZE = 20
        created: set[str] = set()
        operation_ids: list[tuple[str, str]] = []

        for doc in documents:
            bank_id = self._bank_id_for(doc.user_id)
            if self._per_unit and bank_id not in created:
                await self._acreate_bank(client, bank_id)
                created.add(bank_id)

            items = self._doc_to_items(doc)
            for i in range(0, len(items), _BATCH_SIZE):
                batch = items[i:i + _BATCH_SIZE]
                for attempt in range(3):
                    try:
                        resp = await asyncio.wait_for(
                            client.aretain_batch(
                                bank_id=bank_id,
                                items=batch,
                                retain_async=True,
                            ),
                            timeout=300,
                        )
                        break
                    except Exception:
                        if attempt < 2:
                            await asyncio.sleep(10)
                        else:
                            raise

                if resp.var_async:
                    if not resp.operation_id:
                        raise RuntimeError(
                            f"Server processed retain asynchronously but returned no operation_id "
                            f"for bank={bank_id}. Cannot wait for extraction to complete."
                        )
                    operation_ids.append((bank_id, resp.operation_id))

        # Wait for all operations after all batches are submitted (same as embedded)
        for bank_id, op_id in operation_ids:
            await self._await_operation(client, bank_id, op_id)

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict | None]:
        client = self._get_async_client()
        try:
            response = await asyncio.wait_for(
                client.arecall(**self._recall_kwargs(query, user_id, query_timestamp)),
                timeout=300,
            )
        except asyncio.TimeoutError:
            import logging
            logging.getLogger(__name__).warning(f"async_retrieve timed out for query={query[:60]!r}")
            return [], None
        chunks = response.chunks or {}
        docs = _build_docs(_deduplicate_results(response.results), chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw

    async def async_retrieve_by_steps(self, steps: list[int], query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, compact: bool | None = None) -> tuple[list[Document], dict | None]:
        # Legacy path: include_chunks for small sets, facts-only for large ranges
        include_chunks = len(steps) <= 6
        kwargs = self._recall_kwargs(query, user_id, query_timestamp, include_chunks=include_chunks, max_chunk_tokens=16384)
        if steps:
            kwargs["tags"] = [f"step_number:{s}" for s in steps]
            kwargs["tags_match"] = "any_strict"
        client = self._get_async_client()
        try:
            response = await asyncio.wait_for(client.arecall(**kwargs), timeout=120)
        except asyncio.TimeoutError:
            import logging
            logging.getLogger(__name__).warning(f"async_retrieve_by_steps timed out for query={query[:60]!r}")
            return [], None
        chunks = response.chunks or {}
        results = _deduplicate_results(response.results)
        if not self._per_unit and user_id is not None and steps:
            uid_filter = f"user:{user_id}"
            results = [r for r in results if uid_filter in (r.tags or [])]
        docs = _build_docs(results, chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw

    async def async_direct_answer(self, query: str, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[str, str, dict | None]:
        client = self._get_async_client()
        try:
            response = await asyncio.wait_for(
                client.areflect(**self._reflect_kwargs(query, user_id, query_timestamp)),
                timeout=300,
            )
        except asyncio.TimeoutError:
            import logging
            logging.getLogger(__name__).warning(f"async_direct_answer timed out for query={query[:60]!r}")
            return "", "", None
        answer = response.text or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return answer, answer, raw

    async def async_retrieve_by_tag(self, tag: str, query: str = "", user_id: str | None = None) -> tuple[list[Document], dict | None]:
        client = self._get_async_client()
        kwargs = self._recall_kwargs(query or "relevant information", user_id, None)
        kwargs["tags"] = [tag]
        kwargs["tags_match"] = "any_strict"
        response = await client.arecall(**kwargs)
        chunks = response.chunks or {}
        docs = _build_docs(_deduplicate_results(response.results), chunks)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return docs, raw


# ── HTTP provider (local server) ──────────────────────────────────────────────

class HindsightHTTPMemoryProvider(HindsightCloudMemoryProvider):
    name = "hindsight-http"
    description = "Hindsight via a self-hosted HTTP endpoint. Recall uses all memory types (world + experience + observation) with no type filter applied."
    kind = "cloud"
    provider = "hindsight"
    variant = "http"

    def __init__(self):
        # Bypass HindsightCloudMemoryProvider.__init__ — no API key required.
        _HindsightBase.__init__(self)
        from hindsight import HindsightClient
        self._cloud_api_key = os.environ.get("HINDSIGHT_HTTP_KEY", "")
        self._cloud_base_url = os.environ.get("HINDSIGHT_HTTP_URL", "http://localhost:8888")
        self._client = HindsightClient(base_url=self._cloud_base_url, api_key=self._cloud_api_key)

    def _bank_id_for(self, user_id: str | None) -> str:
        if self._per_unit and user_id is not None:
            return f"{self._bank_id}-u{user_id}"
        return self._bank_id
