import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .utils import count_tokens

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

_CONCURRENCY = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from .dataset.base import Dataset
from .judge import GeminiJudge
from .memory.base import MemoryProvider
from .modes.base import ResponseMode
from .models import EvalSummary, QueryResult

console = Console()


def _score_mcq(answer: str, gold_answers: list[str]) -> tuple[bool, str]:
    """Exact letter match — no LLM needed for multiple-choice questions."""
    def norm(s: str) -> str:
        return s.strip().lower().strip("(). ")[:1]

    answer_letter = norm(answer)
    for gold in gold_answers:
        if norm(gold) == answer_letter:
            return True, "letter match"
    return False, f"expected one of {gold_answers!r}, got {answer!r}"


class EvalRunner:
    def __init__(self, output_dir: Path = Path("outputs")):
        self.output_dir = output_dir
        self._judge = GeminiJudge()

    def _get_judge(self, dataset: Dataset) -> "GeminiJudge":
        dataset_llm = dataset.default_judge_llm() if hasattr(dataset, "default_judge_llm") else None
        if dataset_llm is not None:
            return GeminiJudge(llm=dataset_llm)
        return self._judge

    def run(
        self,
        dataset: Dataset,
        split: str,
        memory: MemoryProvider,
        mode: ResponseMode,
        category: str | None = None,
        query_limit: int | None = None,
        query_id: str | None = None,
        doc_limit: int | None = None,
        oracle: bool = False,
        skip_ingestion: bool = False,
        skip_ingested: bool = False,
        skip_retrieval: bool = False,
        skip_answer: bool = False,
        only_failed: bool = False,
        show_raw: bool = False,
        run_name: str | None = None,
        description: str | None = None,
    ) -> EvalSummary:
        effective_name = run_name or memory.name
        cat_label = f"  [bold]Category:[/bold] {category}" if category else ""
        name_label = f"  [bold]Run:[/bold] {effective_name}" if run_name else ""
        console.print(f"\n[bold]Dataset:[/bold] {dataset.name}  [bold]Split:[/bold] {split}{cat_label}  [bold]Memory:[/bold] {memory.name}{name_label}  [bold]Mode:[/bold] {mode.name}\n")

        task_type = dataset.task_type
        if task_type == "mcq":
            console.print("[dim]Task type: multiple-choice (exact letter match scoring)[/dim]")

        if oracle and not dataset.supports_oracle():
            raise ValueError(f"Dataset '{dataset.name}' does not support oracle mode (no gold_ids)")

        memory.initialize()
        store_dir = self.output_dir / dataset.name / effective_name / "_store" / split / (category or "all")

        # Parse comma-separated categories; load query_limit queries per category
        cats: list[str | None] = (
            [c.strip() for c in category.split(",") if c.strip()]
            if category else [None]
        )

        console.print("[dim]Loading queries...[/dim]")
        if len(cats) > 1:
            seen_ids: set[str] = set()
            queries = []
            for cat in cats:
                for q in dataset.load_queries(split, category=cat, limit=query_limit):
                    if q.id not in seen_ids:
                        seen_ids.add(q.id)
                        queries.append(q)
            # doc loading must not be filtered by category when combining multiple
            doc_category: str | None = None
        else:
            queries = dataset.load_queries(split, category=cats[0], limit=query_limit)
            doc_category = cats[0]

        if query_id:
            queries = [q for q in queries if q.id == query_id]
            console.print(f"  1 query loaded [dim](--query-id)[/dim]")
        elif only_failed:
            prev = self._load_previous(dataset.name, split, effective_name, mode.name)
            failed_ids = {r["query_id"] for r in prev.get("results", []) if not r.get("correct")}
            queries = [q for q in queries if q.id in failed_ids]
            console.print(f"  {len(queries)} failed queries loaded [dim](--only-failed)[/dim]")
        else:
            console.print(f"  {len(queries)} queries loaded")

        console.print("[dim]Loading documents...[/dim]")
        if oracle:
            gold_ids = {gid for q in queries for gid in q.gold_ids}
            if not gold_ids:
                raise ValueError("Oracle mode requested but no gold_ids found in queries")
            documents = dataset.load_documents(split, category=doc_category, ids=gold_ids)
            console.print(f"  {len(documents)} gold documents loaded [dim](oracle mode)[/dim]")
        elif dataset.isolation_unit is not None and query_limit is not None:
            # For isolated datasets with a query limit, only load docs for the queried units
            # to avoid loading the entire dataset into memory unnecessarily.
            query_user_ids = {q.user_id for q in queries if q.user_id}
            documents = dataset.load_documents(split, category=doc_category, limit=doc_limit, user_ids=query_user_ids)
            console.print(f"  {len(documents)} documents loaded")
        else:
            documents = dataset.load_documents(split, category=doc_category, limit=doc_limit)
            console.print(f"  {len(documents)} documents loaded")

        # Pass isolation unit IDs to prepare() so providers can set up per-unit storage
        # (e.g. one bank per episode) before ingestion begins.
        unit_ids: set[str] | None = None
        if dataset.isolation_unit is not None:
            unit_ids = {uid for doc in documents if (uid := dataset.get_isolation_id(doc)) is not None}

        memory.prepare(store_dir, unit_ids=unit_ids)

        stored_contexts: dict[str, str] = {}
        stored_answers: dict[str, str] = {}
        if skip_retrieval or skip_answer:
            prev = self._load_previous(dataset.name, split, effective_name, mode.name)
            stored_contexts = {r["query_id"]: r["context"] for r in prev.get("results", []) if r.get("context")}
            if skip_answer:
                stored_answers = {r["query_id"]: r["answer"] for r in prev.get("results", []) if r.get("answer")}
                console.print(f"[dim]Skipping retrieval+answer — re-judging {len(stored_answers)} cached answers.[/dim]\n")
            else:
                console.print(f"[dim]Skipping retrieval — using {len(stored_contexts)} cached contexts.[/dim]\n")

        def _prompt_fn(query, context, meta=None):
            return dataset.build_rag_prompt(query, context, task_type, split, category, meta)

        async def _process_one(q) -> QueryResult:
            t_start = time.perf_counter()
            logger.info("[query:%s] start — %s", q.id, q.query[:80])
            meta = {**q.meta, "_prompt_fn": _prompt_fn}
            if skip_answer:
                from .models import AnswerResult
                ctx = stored_contexts.get(q.id, "")
                ans = stored_answers.get(q.id, "")
                answer_result = AnswerResult(answer=ans, reasoning="", context=ctx, retrieve_time_ms=0.0)
            elif skip_retrieval:
                ctx = stored_contexts.get(q.id, "")
                answer_result = await asyncio.to_thread(mode.answer_from_context, q.query, ctx, task_type, meta)
            else:
                answer_result = await mode.async_answer(q.query, memory, task_type=task_type, user_id=q.user_id, meta=meta)
            logger.info("[query:%s] answer done in %.1fs (retrieve=%.0fms)", q.id, time.perf_counter() - t_start, answer_result.retrieve_time_ms)

            if not answer_result.context:
                correct, judge_reason = False, "empty context — no memories retrieved"
            elif task_type == "mcq":
                correct, judge_reason = _score_mcq(answer_result.answer, q.gold_answers)
            else:
                # Use per-query judge if dataset supports it (e.g. LongMemEval has per-category prompts)
                if hasattr(dataset, "get_judge_prompt_fn"):
                    judge_prompt_fn = dataset.get_judge_prompt_fn(
                        q.meta.get("question_type") or q.meta.get("category"),
                        meta=q.meta,
                    )
                else:
                    judge_prompt_fn = dataset.build_judge_prompt
                judge = await asyncio.to_thread(self._get_judge(dataset).score, q.query, answer_result.answer, q.gold_answers, judge_prompt_fn)
                correct, judge_reason = judge.correct, judge.reason
            logger.info("[query:%s] done in %.1fs — correct=%s", q.id, time.perf_counter() - t_start, correct)

            return QueryResult(
                query_id=q.id,
                query=q.query,
                answer=answer_result.answer,
                reasoning=answer_result.reasoning,
                context=answer_result.context,
                context_tokens=count_tokens(answer_result.context),
                retrieve_time_ms=answer_result.retrieve_time_ms,
                gold_answers=q.gold_answers,
                correct=correct,
                judge_reason=judge_reason,
                meta=q.meta,
                raw_response=answer_result.raw_response,
                category_axes=dataset.get_result_categories(q.meta),
            )

        ingestion_ms = 0.0
        ingested_docs_count = 0

        if dataset.isolation_unit is not None:
            # Unit-sequential mode: ingest one unit's document then answer all its queries
            # before moving to the next unit. This matches the original AMA-bench setup and
            # gives natural per-unit isolation without shared-bank tag scoping.
            queries_by_unit: dict[str, list] = {}
            for q in queries:
                if q.user_id:
                    queries_by_unit.setdefault(q.user_id, []).append(q)

            # Group all docs by unit, keeping only units that have at least one query
            docs_by_unit: dict[str, list] = {}
            for doc in documents:
                uid = dataset.get_isolation_id(doc)
                if uid is not None and uid in queries_by_unit:
                    docs_by_unit.setdefault(uid, []).append(doc)

            # Determine which units are already done (have results in previous run)
            already_done_units: set[str] = set()
            if skip_ingested:
                prev = self._load_previous(dataset.name, split, effective_name, mode.name)
                for r in prev.get("results", []):
                    uid = r.get("meta", {}).get("sample_id") or r.get("meta", {}).get("user_id")
                    if uid:
                        already_done_units.add(uid)
                if already_done_units:
                    console.print(f"[dim]Skipping {len(already_done_units)} already-ingested units: {', '.join(sorted(already_done_units))}[/dim]")

            if skip_ingestion:
                console.print(f"[dim]Skipping ingestion (--skip-ingestion).[/dim]\n")
                ingestion_ms = self._load_previous_ingestion_ms(dataset.name, split, effective_name, mode.name)
                ingested_docs_count = self._load_previous_ingested_docs(dataset.name, split, effective_name, mode.name)
            else:
                pending_units = len(docs_by_unit) - len(already_done_units)
                console.print(f"[dim]Ingesting {pending_units} units into {memory.name} (unit-sequential)...[/dim]")

            # Pre-load previous results for skip-ingested fast-path
            _prev_by_unit: dict[str, list[QueryResult]] = {}
            if already_done_units:
                prev_data = self._load_previous(dataset.name, split, effective_name, mode.name)
                import dataclasses
                _qr_fields = {f.name for f in dataclasses.fields(QueryResult)}
                for r in prev_data.get("results", []):
                    uid = r.get("meta", {}).get("sample_id") or r.get("meta", {}).get("user_id")
                    if uid in already_done_units:
                        _prev_by_unit.setdefault(uid, []).append(
                            QueryResult(**{k: v for k, v in r.items() if k in _qr_fields})
                        )

            async def _run_unit_sequential(progress, task_id):
                nonlocal ingestion_ms, ingested_docs_count
                concurrency = getattr(memory, "concurrency", _CONCURRENCY)
                sem = asyncio.Semaphore(concurrency)
                all_results = []

                for unit_id, unit_docs in docs_by_unit.items():
                    if unit_id in already_done_units:
                        unit_prev = _prev_by_unit.get(unit_id, [])
                        all_results.extend(unit_prev)
                        progress.advance(task_id, len(unit_prev))
                        continue

                    if not skip_ingestion:
                        t0 = time.perf_counter()
                        await memory.async_ingest(unit_docs)
                        ingestion_ms += (time.perf_counter() - t0) * 1000
                        ingested_docs_count += len(unit_docs)

                    unit_queries = queries_by_unit.get(unit_id, [])
                    unit_results = [None] * len(unit_queries)

                    async def bounded(i, q):
                        async with sem:
                            unit_results[i] = await _process_one(q)
                            progress.advance(task_id)

                    await asyncio.gather(*[bounded(i, q) for i, q in enumerate(unit_queries)])
                    all_results.extend(unit_results)
                    # Save incrementally after each unit to survive crashes
                    partial = EvalSummary(
                        dataset=dataset.name, split=split, category=category,
                        memory_provider=memory.name, run_name=effective_name,
                        mode=mode.name, oracle=oracle,
                        total_queries=len(all_results),
                        correct=sum(1 for r in all_results if r and r.correct),
                        accuracy=0.0, ingestion_time_ms=round(ingestion_ms, 1),
                        ingested_docs=ingested_docs_count,
                        description=description, answer_llm=mode.llm_id,
                        judge_llm=self._get_judge(dataset)._llm.model_id, results=[r for r in all_results if r],
                    )
                    self._save(partial)

                return all_results

            with Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(),
                          TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
                task_id = progress.add_task("Evaluating", total=len(queries))
                results = asyncio.run(_run_unit_sequential(progress, task_id))

        else:
            # Batch mode: ingest all documents upfront, then answer all queries.
            if skip_ingestion:
                console.print(f"[dim]Skipping ingestion (--skip-ingestion).[/dim]\n")
                ingestion_ms = self._load_previous_ingestion_ms(dataset.name, split, effective_name, mode.name)
                ingested_docs_count = self._load_previous_ingested_docs(dataset.name, split, effective_name, mode.name)
            else:
                console.print(f"[dim]Ingesting into {memory.name}...[/dim]")
                t0 = time.perf_counter()
                memory.ingest(documents)
                ingestion_ms = (time.perf_counter() - t0) * 1000
                ingested_docs_count = len(documents)
                console.print(f"  ingested in {ingestion_ms:.0f}ms ({ingestion_ms / len(documents):.1f}ms/doc avg)\n")

            async def _run_all(progress, task_id):
                concurrency = getattr(memory, "concurrency", _CONCURRENCY)
                sem = asyncio.Semaphore(concurrency)
                results = [None] * len(queries)

                async def bounded(i, q):
                    async with sem:
                        results[i] = await _process_one(q)
                        progress.advance(task_id)

                await asyncio.gather(*[bounded(i, q) for i, q in enumerate(queries)])
                return results

            with Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(),
                          TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
                task_id = progress.add_task("Evaluating", total=len(queries))
                results = asyncio.run(_run_all(progress, task_id))

        if show_raw:
            for r in results:
                if r.raw_response is not None:
                    console.rule(f"[dim]raw response — query {r.query_id}[/dim]")
                    console.print_json(json.dumps(r.raw_response))

        if only_failed:
            results = self._merge(results, dataset.name, split, effective_name, mode.name)

        correct_count = sum(1 for r in results if r.correct)
        summary = EvalSummary(
            dataset=dataset.name,
            split=split,
            category=category,
            memory_provider=memory.name,
            run_name=effective_name,
            mode=mode.name,
            oracle=oracle,
            total_queries=len(results),
            correct=correct_count,
            accuracy=correct_count / len(results) if results else 0.0,
            ingestion_time_ms=round(ingestion_ms, 1),
            ingested_docs=ingested_docs_count,
            description=description,
            answer_llm=mode.llm_id,
            judge_llm=self._get_judge(dataset)._llm.model_id,
            results=results,
        )
        self._save(summary)
        memory.cleanup()
        return summary

    def _merge(self, new_results: list[QueryResult], dataset: str, split: str, memory: str, mode: str) -> list[QueryResult]:
        """Merge new results into existing ones — new entries win by query_id."""
        prev = self._load_previous(dataset, split, memory, mode)
        old_by_id = {r["query_id"]: r for r in prev.get("results", [])}
        new_by_id = {r.query_id: r for r in new_results}
        old_by_id.update(new_by_id)  # new results win
        return [
            new_by_id.get(qid) or QueryResult(**r)
            for qid, r in old_by_id.items()
        ]

    def _output_path(self, dataset: str, split: str, run_name: str, mode: str) -> Path:
        return self.output_dir / dataset / run_name / mode / f"{split}.json"

    def _load_previous(self, dataset: str, split: str, memory: str, mode: str) -> dict:
        path = self._output_path(dataset, split, memory, mode)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def _load_previous_ingestion_ms(self, dataset: str, split: str, memory: str, mode: str) -> float:
        return self._load_previous(dataset, split, memory, mode).get("ingestion_time_ms", 0.0)

    def _load_previous_ingested_docs(self, dataset: str, split: str, memory: str, mode: str) -> int:
        return self._load_previous(dataset, split, memory, mode).get("ingested_docs", 0)

    def _save(self, summary: EvalSummary) -> None:
        path = self._output_path(summary.dataset, summary.split, summary.run_name, summary.mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always merge: new results overwrite existing ones by query_id, old ones are kept
        merged = self._merge(summary.results, summary.dataset, summary.split, summary.run_name, summary.mode)
        d = asdict(summary)
        results_dicts      = [asdict(r) for r in merged]
        d["total_queries"] = len(merged)
        d["correct"]       = sum(1 for r in merged if r.correct)
        d["accuracy"]      = d["correct"] / d["total_queries"] if merged else 0.0
        rec_times  = [r["retrieve_time_ms"] for r in results_dicts if r.get("retrieve_time_ms") is not None]
        ctx_tokens = [r["context_tokens"]   for r in results_dicts if r.get("context_tokens")   is not None]
        d["avg_retrieve_time_ms"] = round(sum(rec_times)  / len(rec_times),  1) if rec_times  else None
        d["avg_context_tokens"]   = round(sum(ctx_tokens) / len(ctx_tokens), 1) if ctx_tokens else None
        d["results"]       = results_dicts
        path.write_text(json.dumps(d, indent=2))
        console.print(f"\n[green]Saved → {path}[/green]")
