from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    content: str                         # plain-text representation (used by BM25 etc.)
    user_id: str | None = None           # namespace for per-user isolation
    messages: list[dict] | None = None   # structured conversation turns, e.g. for Mem0
    timestamp: str | None = None         # ISO-8601 datetime when the document occurred
    context: str | None = None           # optional hint about document provenance (passed to Hindsight)


@dataclass
class Query:
    id: str
    query: str
    gold_ids: list[str]
    gold_answers: list[str]
    user_id: str | None = None  # scope retrieval to this user/namespace
    meta: dict = field(default_factory=dict)  # dataset-specific metadata (e.g. persona_id)


@dataclass
class AnswerResult:
    answer: str
    reasoning: str
    context: str          # the raw string injected into the LLM prompt
    retrieve_time_ms: float
    raw_response: dict | None = None  # raw provider response, free-form


@dataclass
class JudgeResult:
    correct: bool
    reason: str


@dataclass
class QueryResult:
    query_id: str
    query: str
    answer: str
    reasoning: str
    context: str
    context_tokens: int        # estimated tokens in the injected context (chars / 4)
    retrieve_time_ms: float    # wall time for memory.retrieve() only
    gold_answers: list[str]
    correct: bool
    judge_reason: str
    meta: dict = field(default_factory=dict)  # propagated from Query.meta
    raw_response: dict | None = None  # raw provider response, free-form
    category_axes: dict[str, list[str]] = field(default_factory=dict)  # axis → values, e.g. {"conversation": ["conv-26"], "question_type": ["temporal"]}


@dataclass
class EvalSummary:
    dataset: str
    split: str                    # e.g. "32k", "bitcoin" — was "domain"
    category: str | None          # optional sub-group, e.g. question_type
    memory_provider: str
    run_name: str        # output directory name; equals memory_provider unless --name was given
    mode: str
    oracle: bool
    total_queries: int
    correct: int
    accuracy: float
    ingestion_time_ms: float  # wall time for memory.ingest()
    ingested_docs: int
    description: str | None = None    # optional free-text annotation for this run
    answer_llm: str | None = None     # e.g. "gemini:gemini-2.5-flash-lite"
    judge_llm: str | None = None      # e.g. "gemini:gemini-2.5-flash-lite"
    results: list[QueryResult] = field(default_factory=list)
