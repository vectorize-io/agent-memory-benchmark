import asyncio
import os
import time
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

# Workaround: hindsight-client passes async_= but the model expects var_async=
# This monkey-patch fixes RetainRequest so async_=True actually works.
try:
    from hindsight_client_api.models.retain_request import RetainRequest as _RR
    _orig_init = _RR.__init__
    def _patched_init(self, *args, **kwargs):
        if "async_" in kwargs and "var_async" not in kwargs:
            kwargs["var_async"] = kwargs.pop("async_")
        _orig_init(self, *args, **kwargs)
    _RR.__init__ = _patched_init
except Exception:
    pass


# Workaround: hindsight-client 0.9.2 lazily imports RecallRequestTagGroupsInner from
# hindsight_client_api.models.recall_request_tag_groups_inner when a recall passes
# tag_groups, but codegen never emitted that module — RecallRequest.tag_groups is in fact
# typed as MentalModelTriggerInputTagGroupsInner, the identical generated union. Register
# the missing module as an alias so tag_groups recall works.
try:
    import sys as _sys
    import hindsight_client_api.models as _hs_models
    if not hasattr(_hs_models, "recall_request_tag_groups_inner"):
        import types as _types
        from hindsight_client_api.models.mental_model_trigger_input_tag_groups_inner import (
            MentalModelTriggerInputTagGroupsInner as _TagGroupsInner,
        )
        _mod = _types.ModuleType("hindsight_client_api.models.recall_request_tag_groups_inner")
        _mod.RecallRequestTagGroupsInner = _TagGroupsInner
        _sys.modules[_mod.__name__] = _mod
except Exception:
    pass


def _as_recall_response(payload: dict):
    """Wrap a raw /memories/recall JSON body in the attribute shape the client returns.

    Used for recall parameters the installed hindsight-client cannot express — it has
    repeatedly lagged the API (tag_groups shipped broken in 0.9.2, query_tag_gate is newer
    still), and a benchmark should not be blocked on client codegen."""
    from types import SimpleNamespace

    def result(d: dict) -> SimpleNamespace:
        return SimpleNamespace(
            id=d.get("id"), text=d.get("text") or "", type=d.get("type"),
            entities=d.get("entities"), context=d.get("context"),
            occurred_start=d.get("occurred_start"), occurred_end=d.get("occurred_end"),
            mentioned_at=d.get("mentioned_at"), document_id=d.get("document_id"),
            metadata=d.get("metadata"), chunk_id=d.get("chunk_id"),
            tags=d.get("tags"), scores=d.get("scores"),
        )

    chunks = {k: SimpleNamespace(text=(v or {}).get("text", "")) for k, v in (payload.get("chunks") or {}).items()}
    return SimpleNamespace(
        results=[result(r) for r in (payload.get("results") or [])],
        chunks=chunks,
        model_dump=lambda: payload,
    )


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
    return [
        Document(id=r.id, content=c, source_ids=_source_ids(r))
        for r, c in zip(results, _format_results(results, chunks))
    ]


def _source_ids(r) -> list[str] | None:
    """The ingested Document.id(s) a recall result came from.

    Ingestion stamps every item with `document_id` and `metadata.doc_id` (see
    _doc_to_items), so recall results can usually be traced back to the benchmark
    document that produced them. Datasets scored on retrieved document identity
    (PrecisionMemBench) need this; answer-based datasets ignore it."""
    ids = []
    meta_doc_id = (getattr(r, "metadata", None) or {}).get("doc_id")
    if meta_doc_id:
        ids.append(meta_doc_id)
    if r.document_id and r.document_id not in ids:
        ids.append(r.document_id)
    return ids or None


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

    supports_filters = True

    def __init__(self):
        self._bank_id = "bench"
        self._dataset: str | None = None
        self._category: str | None = None
        self._default_user_id = "omb-bench-default"
        self._client = None  # set by subclass
        self._async_client = None  # lazily created (cloud only)
        self._per_unit = False
        self._resume = os.environ.get("AMB_RESUME", "").lower() in ("1", "true")
        self._stale_failed_ops: dict[str, set] = {}
        self._extraction_labels: list[dict] | None = None
        self._recall_fields_supported: set[str] = set()

    def _bank_id_for(self, user_id: str | None) -> str:
        if self._per_unit and user_id is not None:
            return f"{self._bank_id}-u{user_id}"
        return self._bank_id

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._bank_id, self._dataset, self._category = _bank_id_from_store_dir(store_dir)
        self._per_unit = unit_ids is not None

    def set_extraction_labels(self, labels: list[dict] | None) -> None:
        """Controlled vocabularies the extraction LLM should classify each fact against.

        Applied to the bank as `entity_labels`; entries with tag=True have their chosen value
        written onto the fact as a tag, which recall can then hard-filter on (a SQL WHERE
        across all four retrieval strategies rather than a post-filter)."""
        self._extraction_labels = labels

    # ── Bank creation (sync) ──────────────────────────────────────────────────

    _BEAM_RETAIN_MISSION = (
        "Extract ALL factual claims the user makes about themselves, their project, "
        "and their experience — including NEGATIVE statements (e.g. 'I have never done X', "
        "'I don't know Y', 'I haven't used Z'). Negative self-assessments and denials "
        "are as important as positive ones. Also preserve contradictions: if the user "
        "says opposite things at different points, extract BOTH statements as separate facts. "
        "Preserve specific numbers, dates, versions, and quantities exactly as stated."
    )

    # PrecisionMemBench seeds one belief per document and scores which beliefs come
    # back. Default extraction fans a belief out into 2-3 facts (each an independent
    # retrieval unit, so a split belief occupies several budget slots) and drops the
    # alias/shorthand tokens the benchmark queries by (k8s, GHA, moongoose).
    _PMB_RETAIN_MISSION = (
        "Each input document is ONE belief record about the user, written as: canonical "
        "name, then its alias and shorthand forms, then the statement, then why it matters. "
        "Extract EXACTLY ONE fact per input document — never split a document into several "
        "facts, even when it states more than one thing; keep them together in a single fact. "
        "Preserve the canonical name and EVERY alias and shorthand form VERBATIM in the fact "
        "text, including abbreviations, acronyms, short forms and misspellings (for example "
        "k8s, kube, GHA, TS, Mongo, moongoose) — these are how the user refers to the concept "
        "and the fact must stay findable by those exact strings. Do not generalise, do not "
        "merge records, and do not add framing such as 'Involving: user'."
    )

    def _bank_kwargs(self, bank_id: str | None = None) -> dict:
        kwargs: dict = dict(enable_observations=False)
        if self._dataset == "beam":
            kwargs["retain_mission"] = self._BEAM_RETAIN_MISSION
        elif self._dataset == "precisionmembench":
            kwargs["retain_mission"] = self._PMB_RETAIN_MISSION
        # Explicit override always wins (mission A/B experiments).
        env_mission = os.environ.get("AMB_RETAIN_MISSION")
        if env_mission:
            kwargs["retain_mission"] = env_mission
        # Optional override of the retain extraction strategy, e.g.
        # AMB_RETAIN_EXTRACTION_MODE=chunks to store raw chunks instead of
        # extracting concise facts ('concise' is the server default).
        extraction_mode = os.environ.get("AMB_RETAIN_EXTRACTION_MODE")
        if extraction_mode:
            kwargs["retain_extraction_mode"] = extraction_mode
        return kwargs

    def _create_bank(self, bank_id: str, force_reset: bool = True) -> None:
        kwargs = self._bank_kwargs(bank_id=bank_id)
        if force_reset:
            try:
                self._client.banks.delete(bank_id=bank_id)
            except Exception:
                pass
        self._client.create_bank(bank_id=bank_id, name=f"Benchmark Bank ({bank_id})", **kwargs)
        self._apply_extraction_labels(bank_id)

    def _apply_extraction_labels(self, bank_id: str) -> None:
        """Push the dataset's controlled vocabularies onto the bank as `entity_labels`.

        Labels with tag=True make the extraction LLM stamp its chosen value onto each fact as
        a tag, so recall can hard-filter on a classification the LLM produced at retain time."""
        if not self._extraction_labels:
            return
        import logging
        from hindsight_client_api.api.banks_api import BanksApi
        from hindsight_client_api.models.bank_config_update import BankConfigUpdate
        from hindsight_client.hindsight_client import _run_async
        try:
            api = BanksApi(self._client._api_client)
            _run_async(api.update_bank_config(
                bank_id=bank_id,
                bank_config_update=BankConfigUpdate(updates={"entity_labels": self._extraction_labels}),
            ))
            logging.getLogger(__name__).info(
                f"Bank {bank_id}: applied {len(self._extraction_labels)} entity label group(s)")
        except Exception as e:
            raise RuntimeError(
                f"Bank {bank_id}: failed to apply entity_labels — the run would silently score "
                f"without the classification tags it depends on: {e}"
            ) from e

    # Server-side ingestion bookkeeping. The async path used to wait per-operation with a
    # 5-minute cap and then continue regardless — on a large split (10M: ~560 docs/bank) that
    # means answering questions against a half-filled bank. These wait on the whole bank and
    # refuse to proceed unless every queued operation actually completed.

    # Retry budget for server-side operation failures. Retries are spaced out: the backend
    # tends to be unavailable for minutes, so firing every attempt within one 15s poll cycle
    # burns the whole budget on a single outage.
    _OP_RETRY_LIMIT = int(os.environ.get("AMB_OP_RETRY_LIMIT", 5))
    _OP_RETRY_COOLDOWN_S = int(os.environ.get("AMB_OP_RETRY_COOLDOWN_S", 180))

    @staticmethod
    def _ops_api(client):
        from hindsight_client_api.api.operations_api import OperationsApi
        return OperationsApi(client._api_client)

    async def _count_ops(self, client, bank_id: str, status: str | None = None) -> int:
        kwargs = dict(bank_id=bank_id, limit=1)
        if status:
            kwargs["status"] = status
        resp = await asyncio.wait_for(self._ops_api(client).list_operations(**kwargs), timeout=60)
        return resp.total or 0

    async def _in_flight_ops(self, client, bank_id: str) -> tuple[int, int]:
        """Return (in_flight, failed). in_flight covers pending AND processing, which a
        status filter cannot express — derive it from the totals instead."""
        total     = await self._count_ops(client, bank_id)
        completed = await self._count_ops(client, bank_id, "completed")
        failed    = await self._count_ops(client, bank_id, "failed")
        return max(0, total - completed - failed), failed

    async def _retry_failed_ops(self, client, bank_id: str, retried: dict) -> tuple[int, int, int]:
        """Re-queue failed operations server-side. Returns (requeued, unrecoverable, cooling)."""
        ops_api = self._ops_api(client)
        requeued = unrecoverable = cooling = 0
        offset = 0
        while True:
            resp = await asyncio.wait_for(
                ops_api.list_operations(bank_id=bank_id, status="failed", limit=100, offset=offset),
                timeout=60,
            )
            ops = resp.operations or []
            if not ops:
                break
            now = time.monotonic()
            for op in ops:
                if op.id in self._stale_failed_ops.get(bank_id, ()):
                    continue
                attempts, last = retried.get(op.id, (0, 0.0))
                if attempts >= self._OP_RETRY_LIMIT:
                    unrecoverable += 1
                    continue
                if attempts and now - last < self._OP_RETRY_COOLDOWN_S:
                    cooling += 1  # wait out the outage before spending another attempt
                    continue
                try:
                    await asyncio.wait_for(
                        ops_api.retry_operation(bank_id=bank_id, operation_id=op.id), timeout=60)
                    retried[op.id] = (attempts + 1, now)
                    requeued += 1
                except Exception:
                    retried[op.id] = (attempts + 1, now)
                    cooling += 1
            offset += len(ops)
            if offset >= (resp.total or 0):
                break
        return requeued, unrecoverable, cooling

    async def _throttle_submissions(self, client, bank_id: str, max_in_flight: int) -> None:
        """Hold off submitting more batches while the server's queue is deep.

        Firing every batch at once is what made the backend start refusing connections;
        the client cannot go faster than the worker drains anyway."""
        import logging
        _log = logging.getLogger(__name__)
        while True:
            try:
                in_flight, _ = await self._in_flight_ops(client, bank_id)
            except Exception:
                return  # bookkeeping must never block ingestion
            if in_flight <= max_in_flight:
                return
            _log.info(f"Bank {bank_id}: {in_flight} ops in flight (cap {max_in_flight}) — pausing submissions")
            await asyncio.sleep(15)

    async def _await_bank_ingest(self, client, bank_id: str) -> None:
        """Block until every operation queued for this bank has completed.

        Retries server-side failures, and raises rather than letting the run score a
        partially-ingested bank."""
        import logging
        _log = logging.getLogger(__name__)
        max_wait_s = int(os.environ.get("AMB_INGEST_MAX_WAIT_S", 8 * 3600))
        poll = 15
        retried: dict = {}
        start = time.monotonic()
        while time.monotonic() - start < max_wait_s:
            in_flight, failed = await self._in_flight_ops(client, bank_id)
            if failed:
                requeued, unrecoverable, cooling = await self._retry_failed_ops(
                    client, bank_id, retried)
                _log.warning(f"Bank {bank_id}: {failed} failed op(s) — re-queued {requeued}, "
                             f"cooling off {cooling}, unrecoverable {unrecoverable}")
                if unrecoverable:
                    raise RuntimeError(
                        f"Bank {bank_id}: {unrecoverable} operation(s) still failing after "
                        f"{self._OP_RETRY_LIMIT} retries. Aborting to avoid scoring with "
                        f"incomplete ingestion."
                    )
                if requeued or cooling:
                    await asyncio.sleep(poll if requeued else self._OP_RETRY_COOLDOWN_S // 4)
                    continue
                # every failure predates this ingest — its documents were re-sent
            if in_flight == 0:
                stored = await self._count_ops(client, bank_id, "completed")
                _log.info(f"Bank {bank_id}: ingestion complete "
                          f"({stored} ops, {time.monotonic() - start:.0f}s)")
                return
            _log.info(f"Bank {bank_id}: {in_flight} ops in flight "
                      f"({time.monotonic() - start:.0f}s elapsed)")
            await asyncio.sleep(poll)
        raise RuntimeError(
            f"Bank {bank_id}: timed out after {max_wait_s}s waiting for ingestion "
            f"(raise AMB_INGEST_MAX_WAIT_S). Aborting to avoid scoring with incomplete ingestion."
        )

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
        kwargs = self._bank_kwargs(bank_id=bank_id)
        if not self._resume:
            try:
                await client.adelete_bank(bank_id=bank_id)
            except Exception:
                pass
        try:
            await client.acreate_bank(bank_id=bank_id, name=f"Benchmark Bank ({bank_id})", **kwargs)
        except Exception:
            if not self._resume:
                raise  # under AMB_RESUME the bank usually already exists
        await self._aapply_extraction_labels(client, bank_id)

    async def _aapply_extraction_labels(self, client, bank_id: str) -> None:
        """Async twin of _apply_extraction_labels."""
        if not self._extraction_labels:
            return
        import logging
        from hindsight_client_api.api.banks_api import BanksApi
        from hindsight_client_api.models.bank_config_update import BankConfigUpdate
        try:
            api = BanksApi(client._api_client)
            await api.update_bank_config(
                bank_id=bank_id,
                bank_config_update=BankConfigUpdate(updates={"entity_labels": self._extraction_labels}),
            )
            logging.getLogger(__name__).info(
                f"Bank {bank_id}: applied {len(self._extraction_labels)} entity label group(s)")
        except Exception as e:
            raise RuntimeError(
                f"Bank {bank_id}: failed to apply entity_labels — the run would silently score "
                f"without the classification tags it depends on: {e}"
            ) from e

    async def _snapshot_failed_ops(self, client, bank_id: str) -> set:
        """Ids of ops already marked failed BEFORE this ingest started.

        Their documents get re-sent, so charging these stale records against this run's
        retry budget would abort a perfectly healthy resume."""
        stale: set = set()
        offset = 0
        try:
            while True:
                resp = await asyncio.wait_for(
                    self._ops_api(client).list_operations(
                        bank_id=bank_id, status="failed", limit=100, offset=offset),
                    timeout=60,
                )
                ops = resp.operations or []
                if not ops:
                    break
                stale.update(o.id for o in ops)
                offset += len(ops)
                if offset >= (resp.total or 0):
                    break
        except Exception:
            return set()
        return stale

    async def _retained_document_ids(self, client, bank_id: str) -> set:
        """Document ids the server has already retained for this bank (AMB_RESUME).

        A crash mid-ingest otherwise means re-sending an entire 10M conversation."""
        done: set = set()
        offset = 0
        try:
            while True:
                resp = await asyncio.wait_for(
                    self._ops_api(client).list_operations(
                        bank_id=bank_id, status="completed", limit=100, offset=offset),
                    timeout=60,
                )
                ops = resp.operations or []
                if not ops:
                    break
                done.update(o.document_id for o in ops if o.document_id)
                offset += len(ops)
                if offset >= (resp.total or 0):
                    break
        except Exception:
            return set()  # bank absent or listing failed — ingest everything
        return done

    # ── Item builders ─────────────────────────────────────────────────────────

    def _doc_to_items(self, doc: Document) -> list[dict]:
        """Convert a Document to a list of retain items."""
        content = doc.content.replace("\x00", "")
        base: dict = {}
        tags: list[str] = []
        if not self._per_unit:
            tags.append(f"user:{doc.user_id or self._default_user_id}")
        # Dataset-supplied labels (Document.tags) become filterable tags alongside the
        # user scoping tag — recall can then hard-filter on them via tag_groups.
        for t in (doc.tags or []):
            if t not in tags:
                tags.append(t)
        if tags:
            base["tags"] = tags
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
        import logging as _logging
        _log = _logging.getLogger(__name__)

        if not self._per_unit:
            self._create_bank(self._bank_id, force_reset=not self._resume)

        _BATCH_SIZE = 20
        # Cap how deep the server-side queue is allowed to get. Submitting all batches at
        # once made the backend start refusing connections on large splits.
        _MAX_IN_FLIGHT = int(os.environ.get("AMB_MAX_IN_FLIGHT_OPS", 60))
        created: set[str] = set()
        operation_ids: list[tuple[str, str]] = []

        # Collect all items across all documents first, grouped by bank_id,
        # then batch across documents so fewer (larger) operations are created.
        # This makes each operation durable in async_operations and resumable on restart.
        items_by_bank: dict[str, list[dict]] = {}
        for doc in documents:
            bank_id = self._bank_id_for(doc.user_id)
            if self._per_unit and bank_id not in created:
                self._create_bank(bank_id, force_reset=not self._resume)
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

            _use_async = True
            for i in range(0, len(all_items), _BATCH_SIZE):
                batch = all_items[i:i + _BATCH_SIZE]
                doc_label = batch[0].get("document_id", "?") if len(batch) == 1 else f"batch {i // _BATCH_SIZE + 1}"
                for attempt in range(5):
                    try:
                        resp = self._client.retain_batch(
                            bank_id=bank_id,
                            items=batch,
                            retain_async=_use_async,
                        )
                        if _use_async and resp is not None and getattr(resp, "var_async", False):
                            op_id = getattr(resp, "operation_id", None)
                            if op_id:
                                operation_ids.append((bank_id, op_id))
                        if not _use_async:
                            _log.info(f"[retain] {doc_label} done ({i+1}/{len(all_items)})")
                        break
                    except Exception as e:
                        err = str(e)
                        etype = type(e).__name__
                        if ("duplicate key" in err or "duplicate document_ids" in err
                                or "violates foreign key constraint" in err
                                or "empty response" in err):
                            # Skip: already ingested / duplicate / FK race.
                            break
                        if "Cannot connect" in err:
                            # Daemon down — wait longer before retrying.
                            if attempt < 4:
                                time.sleep(30)
                            else:
                                _log.warning(f"retain_batch: daemon down after 5 attempts, skipping: {err[:200]}")
                                break
                        elif "Timeout" in etype or "Timeout" in err or "CancelledError" in err:
                            # Transient LLM/server timeout — retry with backoff.
                            if attempt < 3:
                                _log.warning(f"retain_batch timeout (attempt {attempt+1}/4), retrying in 30s: {err[:100]}")
                                time.sleep(30)
                            else:
                                _log.warning(f"retain_batch: timeout after 4 attempts, skipping batch: {err[:200]}")
                                break
                        elif attempt < 4:
                            time.sleep(10)
                        else:
                            # Last resort: skip any unrecognised transient error rather than killing the run.
                            _log.warning(f"retain_batch unhandled error (skipping batch): {etype}: {err[:200]}")
                            break

        # Wait for async extraction to finish before returning.
        # Critical for large documents: retain_async=True returns immediately but
        # the daemon extracts facts in the background. Without waiting, retrieval
        # right after ingest finds an empty bank.
        if operation_ids or items_by_bank:
            banks_to_check = list(items_by_bank.keys())
            max_wait_s = 28800  # 8 hours max (10m docs have 17K+ chunks per doc)
            poll_interval = 10
            start = time.monotonic()
            _log.info(f"Waiting for extraction to complete on {len(banks_to_check)} bank(s)…")
            for bank_id in banks_to_check:
                deadline = start + max_wait_s
                while time.monotonic() < deadline:
                    try:
                        import httpx as _httpx
                        base_url = self._client._api_client.configuration.host
                        # Check for failed operations first
                        r_failed = _httpx.get(f"{base_url}/v1/default/banks/{bank_id}/operations?status=failed&limit=1", timeout=15)
                        if r_failed.status_code == 200 and r_failed.json().get("total", 0) > 0:
                            failed_count = r_failed.json()["total"]
                            raise RuntimeError(
                                f"Bank {bank_id}: {failed_count} failed operation(s) detected. "
                                f"Aborting to avoid scoring with incomplete ingestion."
                            )
                        # Use lightweight operations query instead of full stats (which JOINs millions of links)
                        r = _httpx.get(f"{base_url}/v1/default/banks/{bank_id}/operations?status=pending&limit=1", timeout=15)
                        if r.status_code == 200:
                            pending = r.json().get("total", 0)
                        else:
                            pending = -1  # unknown
                        if pending == 0:
                            r2 = _httpx.get(f"{base_url}/v1/default/banks/{bank_id}/memories/list?limit=1", timeout=15)
                            total = r2.json().get("total", 0) if r2.status_code == 200 else 0
                            _log.info(f"Bank {bank_id}: extraction complete ({total} facts, 0 pending)")
                            break
                        elapsed = time.monotonic() - start
                        _log.info(f"Bank {bank_id}: {pending} pending ops ({elapsed:.0f}s)")
                    except RuntimeError:
                        raise
                    except Exception as e:
                        elapsed = time.monotonic() - start
                        _log.info(f"Bank {bank_id}: still extracting… ({elapsed:.0f}s, {e.__class__.__name__})")
                    time.sleep(poll_interval)
                else:
                    raise RuntimeError(
                        f"Bank {bank_id}: timed out waiting for extraction after {max_wait_s}s. "
                        f"Aborting to avoid scoring with incomplete ingestion."
                    )

    # ── Recall kwargs ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_tag_groups(filters: dict | None, user_tag: str | None) -> list | None:
        """Provider-neutral filter -> Hindsight `tag_groups` (groups are AND-ed).

        {"all": [t]}   -> {"tags": [t], "match": "all_strict"}
        {"any": [[..]]}-> one {"tags": [...], "match": "any_strict"} leaf per group (OR within)
        {"none": [t]}  -> {"not": {"tags": [t], "match": "any_strict"}}
        Everything lands as a SQL WHERE across all four retrieval strategies rather than a
        post-filter, so excluded memories never enter ranking."""
        if not filters:
            return None
        groups: list = []
        if user_tag:
            groups.append({"tags": [user_tag], "match": "any_strict"})
        if filters.get("narrow_any"):
            # OR of alternative narrowings — each its own leaf, so they can resolve
            # differently (an exact leaf and a fuzzy one over the same query).
            leaves = [
                {"tags": list(spec["tags"]), "match": "any_strict",
                 **({"resolve": spec["resolve"]} if spec.get("resolve") else {})}
                for spec in filters["narrow_any"] if spec.get("tags")
            ]
            if leaves:
                groups.append(leaves[0] if len(leaves) == 1 else {"or": leaves})
        if filters.get("narrow"):
            groups.append({"tags": list(filters["narrow"]), "match": "any_strict"})
        if filters.get("all"):
            groups.append({"tags": list(filters["all"]), "match": "all_strict"})
        for grp in filters.get("any") or []:
            if grp:
                groups.append({"tags": list(grp), "match": "any_strict"})
        if filters.get("none"):
            groups.append({"not": {"tags": list(filters["none"]), "match": "any_strict"}})
        return groups or None

    def _recall_kwargs(self, query: str, user_id: str | None, query_timestamp: str | None, include_chunks: bool = True, max_chunk_tokens: int | None = None, filters: dict | None = None) -> dict:
        is_lifebench = self._dataset == "lifebench"
        is_personamem = self._dataset == "personamem"
        is_beam = self._dataset == "beam"
        if max_chunk_tokens is None:
            if is_personamem:
                max_chunk_tokens = 10240
            elif is_beam:
                max_chunk_tokens = 8192
            elif is_lifebench:
                max_chunk_tokens = 16384
            else:
                max_chunk_tokens = 16384
        if is_personamem:
            max_tokens = 4096
        elif is_lifebench:
            max_tokens = 16384
        elif is_beam:
            max_tokens = 12288
        else:
            max_tokens = 32768
        # Generic overrides — a dataset scored on *which* memories come back rather
        # than on answer quality (precisionmembench) needs a much tighter recall
        # budget than the answer-generation defaults above.
        env_max = os.environ.get("AMB_RECALL_MAX_TOKENS")
        if env_max:
            max_tokens = int(env_max)
        env_chunk = os.environ.get("AMB_RECALL_MAX_CHUNK_TOKENS")
        if env_chunk:
            max_chunk_tokens = int(env_chunk)
        if max_chunk_tokens == 0:
            include_chunks = False
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
        gate = (filters or {}).get("query_gate")
        if gate:
            # Server-side: restrict recall to the identities the query names, and abstain
            # when it names none. Nothing to compute here — the bank owns the vocabulary.
            self._require_recall_field("query_tag_gate")
            kwargs["query_tag_gate"] = dict(gate)
        user_tag = f"user:{user_id}" if (user_id is not None and not self._per_unit) else None
        tag_groups = self._to_tag_groups(filters, user_tag)
        if tag_groups is not None:
            # tag_groups supersedes tags/tags_match: it can express the AND-of-OR-and-NOT that
            # a single flat tag list cannot.
            kwargs["tag_groups"] = tag_groups
        elif user_tag:
            kwargs["tags"] = [user_tag]
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

    def _require_recall_field(self, field: str) -> None:
        """Fail loudly when the server predates a recall parameter the dataset needs.

        An unknown field is accepted and ignored (HTTP 200), so without this check the run
        would quietly score as if the filter had been applied when it never was — the worst
        kind of benchmark result. Checked once per process against the live schema."""
        if field in self._recall_fields_supported:
            return
        import httpx

        base, headers = self._recall_endpoint()
        try:
            spec = httpx.get(f"{base}/openapi.json", headers=headers, timeout=30).json()
            props = spec["components"]["schemas"]["RecallRequest"]["properties"]
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: cannot verify that {base} supports recall field {field!r}: {e}"
            ) from e
        if field not in props:
            raise RuntimeError(
                f"{self.name}: the server at {base} does not support the recall field "
                f"{field!r}, and ignores unknown fields silently — this run would score as "
                f"though the filter applied when it did not. Point HINDSIGHT_CLOUD_URL at a "
                f"server that has it."
            )
        self._recall_fields_supported.add(field)

    # Recall parameters the installed hindsight-client has no argument for. When one is
    # present the request goes out over raw HTTP instead of the typed client.
    _CLIENT_UNSUPPORTED_RECALL_KEYS = ("query_tag_gate",)

    def _needs_http_recall(self, kwargs: dict) -> bool:
        if any(k in kwargs for k in self._CLIENT_UNSUPPORTED_RECALL_KEYS):
            return True
        # `resolve` on a tag_groups leaf postdates the installed client's model, which drops
        # unknown fields silently — the request would go out as an exact match and the run
        # would score as though fuzzy resolution had been applied.
        def _uses_resolve(g):
            if not isinstance(g, dict):
                return False
            if "resolve" in g:
                return True
            return any(_uses_resolve(x) for x in (g.get("or") or g.get("and") or []))
        return any(_uses_resolve(g) for g in (kwargs.get("tag_groups") or []))

    def _recall_endpoint(self) -> tuple[str, dict]:
        """(url, headers) for the bank-agnostic recall endpoint. Cloud/HTTP providers only."""
        base = getattr(self, "_cloud_base_url", None)
        if not base:
            raise RuntimeError(
                f"{self.name} cannot issue raw recall requests — no base URL. The dataset asked "
                f"for a recall parameter the installed hindsight-client cannot express."
            )
        key = getattr(self, "_cloud_api_key", None)
        return base.rstrip("/"), ({"Authorization": f"Bearer {key}"} if key else {})

    def _recall_http(self, kwargs: dict):
        import httpx

        base, headers = self._recall_endpoint()
        body = {k: v for k, v in kwargs.items() if k != "bank_id"}
        r = httpx.post(
            f"{base}/v1/default/banks/{kwargs['bank_id']}/memories/recall",
            headers=headers, json=body, timeout=120,
        )
        r.raise_for_status()
        return _as_recall_response(r.json())

    async def _arecall_http(self, kwargs: dict):
        import httpx

        base, headers = self._recall_endpoint()
        body = {k: v for k, v in kwargs.items() if k != "bank_id"}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{base}/v1/default/banks/{kwargs['bank_id']}/memories/recall",
                headers=headers, json=body,
            )
            r.raise_for_status()
            return _as_recall_response(r.json())

    # ── Sync retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, filters: dict | None = None) -> tuple[list[Document], dict | None]:
        import logging
        _log = logging.getLogger(__name__)
        for attempt in range(3):
            try:
                _kw = self._recall_kwargs(query, user_id, query_timestamp, filters=filters)
                response = self._recall_http(_kw) if self._needs_http_recall(_kw) else self._client.recall(**_kw)
                break
            except Exception as e:
                if attempt < 2:
                    _log.warning(f"recall failed (attempt {attempt+1}/3, retrying in 10s): {e}")
                    time.sleep(10)
                else:
                    _log.warning(f"recall failed after 3 attempts (returning empty): {e}")
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
        # Allow overriding the hindsight-api binary with a local project path
        # e.g. HINDSIGHT_API_PATH=/path/to/hindsight-api
        custom_api_path = os.environ.get("HINDSIGHT_API_PATH")
        if custom_api_path:
            from hindsight_embed.daemon_embed_manager import DaemonEmbedManager
            _custom = custom_api_path
            DaemonEmbedManager._find_api_command = lambda self: ["uv", "run", "--project", _custom, "hindsight-api"]
        from hindsight import HindsightEmbedded
        self._client = HindsightEmbedded(
            profile=f"omb-{self._bank_id}",
            llm_provider="gemini",
            llm_model="gemini-2.5-flash-lite",
            llm_api_key=self._api_key,
            idle_timeout=0,  # Disable idle timeout to prevent daemon from shutting down during long runs
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

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, filters: dict | None = None):
        import logging
        _log = logging.getLogger(__name__)
        kwargs = self._recall_kwargs(query, user_id, query_timestamp, filters=filters)
        for attempt in range(3):
            try:
                response = await self._client.arecall(**kwargs)
                break
            except Exception as e:
                if attempt < 2:
                    _log.warning(f"async_recall failed (attempt {attempt+1}/3, retrying in 10s): {e}")
                    await asyncio.sleep(10)
                else:
                    _log.warning(f"async_recall failed after 3 attempts (returning empty): {e}")
                    return [], None
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

    def ingest(self, documents: list[Document]) -> None:
        """Sync entry point used by the runner's batch (non-isolated) path.

        _HindsightBase.ingest polls the bank's operations endpoint with a bare httpx
        call that carries no API key — harmless against a local daemon, but against
        the cloud API every poll 401s, is swallowed as "still extracting", and the
        run hangs until the 8-hour deadline. Delegate to the authenticated async path
        instead, dropping the cached async client afterwards because it is bound to
        the event loop this call is about to close."""
        try:
            asyncio.run(self.async_ingest(documents))
        finally:
            self._async_client = None

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
        # Cap how deep the server-side queue is allowed to get. Submitting all batches at
        # once made the backend start refusing connections on large splits.
        _MAX_IN_FLIGHT = int(os.environ.get("AMB_MAX_IN_FLIGHT_OPS", 60))
        created: set[str] = set()
        operation_ids: list[tuple[str, str]] = []

        _SUBMIT_ATTEMPTS = int(os.environ.get("AMB_SUBMIT_ATTEMPTS", 8))

        async def _submit(bank_id: str, batch: list[dict]) -> None:
            """Send one retain batch, retrying transient failures.

            The backend can be unavailable for minutes at a time, so keep trying for
            ~5 minutes rather than giving up after a couple of quick retries."""
            for attempt in range(_SUBMIT_ATTEMPTS):
                try:
                    resp = await asyncio.wait_for(
                        client.aretain_batch(
                            bank_id=bank_id,
                            items=batch,
                            retain_async=True,
                        ),
                        timeout=600,
                    )
                    break
                except Exception:
                    if attempt < _SUBMIT_ATTEMPTS - 1:
                        await asyncio.sleep(min(60, 10 * (attempt + 1)))
                    else:
                        raise

            if resp.var_async:
                if not resp.operation_id:
                    raise RuntimeError(
                        f"Server processed retain asynchronously but returned no operation_id "
                        f"for bank={bank_id}. Cannot wait for extraction to complete."
                    )
                operation_ids.append((bank_id, resp.operation_id))

        # Accumulate items ACROSS documents so each request carries a full batch —
        # a document yields a single item, so batching within one document never
        # groups anything (561 docs used to mean 561 single-item requests).
        pending: dict[str, list[dict]] = {}

        import logging
        _log = logging.getLogger(__name__)
        already: dict[str, set] = {}

        for doc in documents:
            bank_id = self._bank_id_for(doc.user_id)
            if self._per_unit and bank_id not in created:
                await self._acreate_bank(client, bank_id)
                created.add(bank_id)
                if self._resume:
                    self._stale_failed_ops[bank_id] = await self._snapshot_failed_ops(client, bank_id)
                    if self._stale_failed_ops[bank_id]:
                        _log.info(f"Bank {bank_id}: ignoring {len(self._stale_failed_ops[bank_id])} "
                                  f"pre-existing failed op(s) from an earlier attempt")
                    already[bank_id] = await self._retained_document_ids(client, bank_id)
                    if already[bank_id]:
                        _log.info(f"Bank {bank_id}: resuming — {len(already[bank_id])} document(s) "
                                  f"already retained, skipping them")

            if doc.id in already.get(bank_id, ()):
                continue

            batch = pending.setdefault(bank_id, [])
            batch.extend(self._doc_to_items(doc))
            while len(batch) >= _BATCH_SIZE:
                await self._throttle_submissions(client, bank_id, _MAX_IN_FLIGHT)
                await _submit(bank_id, batch[:_BATCH_SIZE])
                del batch[:_BATCH_SIZE]

        for bank_id, batch in pending.items():
            if batch:
                await _submit(bank_id, batch)

        # Wait per bank until the server has drained everything we queued.
        for bank_id in dict.fromkeys(b for b, _ in operation_ids):
            await self._await_bank_ingest(client, bank_id)

    async def async_retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None, filters: dict | None = None) -> tuple[list[Document], dict | None]:
        import logging
        _log = logging.getLogger(__name__)
        client = self._get_async_client()
        # Retry transient server errors (500/503/504 from an overloaded backend):
        # without this a single blip aborts the whole run mid-sweep.
        delay = 2
        for attempt in range(5):
            try:
                _kw = self._recall_kwargs(query, user_id, query_timestamp, filters=filters)
                _call = self._arecall_http(_kw) if self._needs_http_recall(_kw) else client.arecall(**_kw)
                response = await asyncio.wait_for(_call, timeout=300)
                # Narrow-then-widen: the narrowing group asked for memories the query names.
                # If nothing carries those names the query may still be about something the
                # ranking can find (a misspelled name), so retry without the restriction and
                # let a score floor decide between "here it is" and "nothing relevant".
                _fb = (filters or {}).get("fallback")
                if _fb and not (response.results or []):
                    _wide = {k: v for k, v in (filters or {}).items() if k != "narrow"}
                    _kw2 = self._recall_kwargs(query, user_id, query_timestamp, filters=_wide)
                    if _fb.get("min_final") is not None:
                        _kw2["min_scores"] = {"final": _fb["min_final"]}
                    _call2 = self._arecall_http(_kw2) if self._needs_http_recall(_kw2) else client.arecall(**_kw2)
                    response = await asyncio.wait_for(_call2, timeout=300)
                break
            except asyncio.TimeoutError:
                _log.warning(f"async_retrieve timed out for query={query[:60]!r}")
                return [], None
            except Exception as e:
                if attempt == 4:
                    _log.error(f"async_retrieve failed after 5 attempts ({e.__class__.__name__}) "
                               f"for query={query[:60]!r} — returning empty context")
                    return [], None
                _log.warning(f"async_retrieve {e.__class__.__name__} (attempt {attempt + 1}/5), "
                             f"retrying in {delay}s")
                await asyncio.sleep(delay)
                delay *= 2
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
