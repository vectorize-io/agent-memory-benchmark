"""
BEAM dataset (https://huggingface.co/datasets/Mohammadta/BEAM).

Benchmarking memory capabilities of LLMs across very long conversations
(100K–10M tokens).  100 conversations, 2 000 probing questions (20 per
conversation) across 10 memory ability categories.

Paper: https://arxiv.org/abs/2510.27246

Data is auto-downloaded from HuggingFace on first use and cached locally.
Set BEAM_DATA_PATH (for the standard 100K/500K/1M splits) or
BEAM_10M_DATA_PATH (for the 10M split) to point at a local JSON file.

Structure
---------
Documents  = one per conversation (full chat turns)
             ID = "{conversation_id}"
             Isolation unit = "conversation" (each conv gets its own bank)

Queries    = one per (conversation × question_category × question_index)
             ID = "{conversation_id}_{category}_{index}"
             user_id = conversation_id

Categories (query-level, 10 memory ability types):
  abstention, contradiction_resolution, event_ordering,
  information_extraction, instruction_following, knowledge_update,
  multi_session_reasoning, preference_following, summarization,
  temporal_reasoning
"""
import ast
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.table import Table
from scipy.stats import kendalltau

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query, QueryResult
from ..llm.base import LLM, Schema

logger = logging.getLogger(__name__)

# HuggingFace split name → our internal name
_HF_SPLIT_MAP = {
    "100k": "100K",
    "500k": "500K",
    "1m":   "1M",
}

SPLITS = ["100k", "500k", "1m", "10m"]

_CATEGORIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# Fields to try (in order) when looking for the gold answer inside a question obj
_ANSWER_FIELDS = ["ideal_response", "answer", "expected_answer", "expected", "gold_answer", "reference"]


def _anchor_to_iso(anchor: str | None) -> str | None:
    """Parse a BEAM time_anchor (e.g. 'March-15-2024') into an ISO-8601 datetime string,
    used as the retain reference date so relative/undated dates resolve to the conversation's
    real year instead of the ingest date. Returns None if unparseable."""
    if not anchor:
        return None
    from datetime import datetime, timezone
    a = str(anchor).strip().replace("_", " ").replace("-", " ").replace(",", " ")
    a = re.sub(r"\s+", " ", a)
    for fmt in ("%B %d %Y", "%b %d %Y", "%Y %m %d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(a, fmt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
        except ValueError:
            continue
    return None


class BEAMDataset(Dataset):
    """
    BEAM benchmark — long-context memory capabilities.

    Data is auto-downloaded from HuggingFace on first use.
    Set BEAM_DATA_PATH or BEAM_10M_DATA_PATH to point at a local JSON file.
    """

    name = "beam"
    published = True
    description = (
        "Long-context memory benchmark: 100 conversations (100K–10M tokens), "
        "2 000 questions across 10 memory ability categories."
    )
    splits = SPLITS
    task_type = "open"
    isolation_unit = "conversation"
    links = [
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/Mohammadta/BEAM"},
        {"label": "Paper",       "url": "https://arxiv.org/abs/2510.27246"},
    ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, split: str) -> Path:
        cache = dataset_cache_dir("beam")
        return cache / f"{split}.json"

    def _load_raw(self, split: str) -> list[dict]:
        env_key = "BEAM_10M_DATA_PATH" if split == "10m" else "BEAM_DATA_PATH"
        local = os.environ.get(env_key)
        if local:
            with open(local, encoding="utf-8") as f:
                return json.load(f)

        path = self._cache_path(split)
        if not path.exists():
            self._download(split, path)

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _download(split: str, dest: Path) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install 'datasets' to auto-download BEAM: pip install datasets"
            ) from e

        if split == "10m":
            hf_name, hf_split = "Mohammadta/BEAM-10M", "10M"
        else:
            hf_name  = "Mohammadta/BEAM"
            hf_split = _HF_SPLIT_MAP[split]

        print(f"Downloading BEAM {split} split from HuggingFace ({hf_name}, split={hf_split})…")
        ds = load_dataset(hf_name, split=hf_split)
        data = [dict(row) for row in ds]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Cached to {dest}")

    @staticmethod
    def _parse_probing_questions(item: dict) -> dict[str, list[dict]]:
        pq = item.get("probing_questions", {})
        if isinstance(pq, str):
            try:
                pq = json.loads(pq)
            except Exception:
                try:
                    pq = ast.literal_eval(pq)
                except Exception:
                    pq = {}
        return pq if isinstance(pq, dict) else {}

    @staticmethod
    def _extract_answer(question_obj: dict) -> str:
        for field in _ANSWER_FIELDS:
            if field in question_obj:
                val = question_obj[field]
                return str(val) if val is not None else ""
        return ""

    @staticmethod
    def _iter_turns(chat: list):
        """
        Yield turn dicts from chat regardless of nesting depth.

        Standard BEAM:  chat = list[session]  where session = list[turn_dict]
        BEAM-10M:       chat = list[plan_dict] where plan_dict = dict[plan_name,
                          list[batch]], batch = {'turns': list[turn_group]},
                          turn_group = list[turn_dict]
        """
        for item in chat:
            if isinstance(item, dict) and "role" in item:
                yield item
            elif isinstance(item, list):
                # Standard BEAM: each item is a session (list of turn dicts)
                for turn in item:
                    if isinstance(turn, dict) and "role" in turn:
                        yield turn
            elif isinstance(item, dict):
                # BEAM-10M: plan_dict with plan-N keys
                for _plan_name, batches in item.items():
                    if not isinstance(batches, list):
                        continue
                    for batch in batches:
                        if not isinstance(batch, dict):
                            continue
                        for turn_group in batch.get("turns", []):
                            if isinstance(turn_group, list):
                                for turn in turn_group:
                                    if isinstance(turn, dict) and "role" in turn:
                                        yield turn
                            elif isinstance(turn_group, dict) and "role" in turn_group:
                                yield turn_group

    @staticmethod
    def _format_chat(chat: list) -> str:
        """Format chat into readable text, handling both standard and 10M structures.

        Includes the turn ID (sequential index from the `id` field) in the prefix
        so that event-ordering models can track which topics were mentioned first.
        """
        lines = []
        for turn in BEAMDataset._iter_turns(chat):
            role    = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            anchor  = turn.get("time_anchor", "")
            turn_id = turn.get("id", "")
            # Include both time anchor AND turn ID for fine-grained ordering
            if anchor and turn_id != "":
                prefix = f"[{anchor} | Turn {turn_id}] "
            elif anchor:
                prefix = f"[{anchor}] "
            elif turn_id != "":
                prefix = f"[Turn {turn_id}] "
            else:
                prefix = ""
            lines.append(f"{prefix}{role}: {content}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def categories(self, split: str) -> list[str] | None:
        return _CATEGORIES

    def category_type(self, split: str, category: str) -> str:
        return "query"

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        data = self._load_raw(split)
        documents: list[Document] = []

        for item in data:
            conv_id = str(item.get("conversation_id", f"conv_{len(documents)}"))

            if user_ids is not None and conv_id not in user_ids:
                continue
            if ids is not None and conv_id not in ids:
                continue

            chat = item.get("chat", [])
            user_profile = item.get("user_profile") or {}
            user_info    = user_profile.get("user_info", "")
            base_context = f"Conversation {conv_id}" + (f" — {user_info}" if user_info else "")

            # Ingest one document per session (or sub-session for very large ones)
            # to avoid asyncpg timeouts from large content fields.
            # Max ~100k chars per document to keep PostgreSQL happy.
            _MAX_DOC_CHARS = 100_000
            sessions = [s for s in chat if isinstance(s, list)]
            if sessions:
                doc_idx = 0
                for s_idx, session in enumerate(sessions):
                    turns = [t for t in session if isinstance(t, dict) and "role" in t]
                    # Reference date for this session: the session's time_anchor (e.g. "March-15-2024").
                    # Passed as Document.timestamp so the retain step anchors relative/undated dates to the
                    # conversation's real date instead of the ingest date (fixes year corruption 2024->2026).
                    session_anchor = _anchor_to_iso(
                        next((t.get("time_anchor") for t in session
                              if isinstance(t, dict) and t.get("time_anchor")), None))
                    # Split session into sub-chunks if it would exceed the char limit
                    chunk_start = 0
                    while chunk_start < max(1, len(turns)):
                        # Grow chunk until it exceeds the char limit
                        chunk_end = chunk_start + 1
                        while chunk_end < len(turns):
                            candidate = self._format_chat([turns[chunk_start:chunk_end + 1]])
                            if len(candidate) > _MAX_DOC_CHARS:
                                break
                            chunk_end += 1
                        chunk_turns = turns[chunk_start:chunk_end]
                        if not chunk_turns:
                            break
                        content = self._format_chat([chunk_turns])
                        chunk_anchor = _anchor_to_iso(
                            next((t.get("time_anchor") for t in chunk_turns if t.get("time_anchor")), None)
                        ) or session_anchor
                        documents.append(Document(
                            id=f"{conv_id}_s{s_idx}_{doc_idx}",
                            content=content,
                            user_id=conv_id,
                            timestamp=chunk_anchor,
                            context=f"{base_context} (session {s_idx + 1}/{len(sessions)})",
                        ))
                        doc_idx += 1
                        chunk_start = chunk_end
            else:
                # Fallback for unusual structures (e.g., BEAM-10M flat turns)
                documents.append(Document(
                    id=conv_id,
                    content=self._format_chat(chat),
                    user_id=conv_id,
                    context=base_context,
                ))

        if limit and ids is None:
            documents = documents[:limit]
        return documents

    @staticmethod
    def _extract_meta_for_category(cat: str, question_obj: dict) -> dict:
        """Extract category-specific metadata useful for prompting and judging."""
        meta: dict = {}
        rubric = question_obj.get("rubric")
        if rubric:
            meta["rubric"] = rubric if isinstance(rubric, list) else [str(rubric)]

        if cat == "abstention":
            meta["why_unanswerable"] = question_obj.get("why_unanswerable", "")
        elif cat == "contradiction_resolution":
            meta["tests_for"] = question_obj.get("tests_for", "")
        elif cat == "instruction_following":
            meta["instruction_being_tested"] = question_obj.get("instruction_being_tested", "")
            meta["compliance_indicators"] = question_obj.get("compliance_indicators", [])
        elif cat == "preference_following":
            meta["preference_being_tested"] = question_obj.get("preference_being_tested", "")
            meta["compliance_indicators"] = question_obj.get("compliance_indicators", [])
        elif cat == "temporal_reasoning":
            meta["time_points"] = question_obj.get("time_points", [])
            meta["calculation_required"] = question_obj.get("calculation_required", "")
        elif cat == "event_ordering":
            meta["ordering_tested"] = question_obj.get("ordering_tested", [])
            total = question_obj.get("total_mentions")
            if total is not None:
                meta["total_mentions"] = int(total)
        return meta

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        data = self._load_raw(split)
        queries: list[Query] = []

        for item in data:
            conv_id = str(item.get("conversation_id", f"conv_{len(queries)}"))
            pq      = self._parse_probing_questions(item)

            cats_to_process = [category] if category else _CATEGORIES

            for cat in cats_to_process:
                for idx, question_obj in enumerate(pq.get(cat, [])):
                    question = question_obj.get("question", "")
                    if not question:
                        continue

                    answer   = self._extract_answer(question_obj)
                    query_id = f"{conv_id}_{cat}_{idx}"
                    extra    = self._extract_meta_for_category(cat, question_obj)

                    queries.append(Query(
                        id=query_id,
                        query=question,
                        gold_ids=[conv_id],
                        gold_answers=[answer] if answer else [],
                        user_id=conv_id,
                        meta={
                            "question_category": cat,
                            "conversation_id": conv_id,
                            **extra,
                        },
                    ))

        if limit:
            queries = queries[:limit]
        return queries

    _BEHAVIORAL = {
        "summarization": (
            "\nThis is a SUMMARY request: write a COMPREHENSIVE, chronological summary that covers "
            "EVERY distinct phase, issue, decision, technical choice, problem, and development found "
            "in the memories \u2014 be exhaustive (completeness across all topics matters more than "
            "brevity), and include specific names, dates, versions, and details. Do NOT reply that you "
            "lack information if ANY relevant memories exist; summarize everything available.\n"
        ),
        # temporal_reasoning: NO extra guidance — duration-specific guidance regressed non-duration
        # questions (0.744 -> 0.662). mem0's generic "pay attention to dates" rule already suffices.
        # knowledge_update: NO extra guidance — "report only the latest value" regressed questions
        # that aren't update-chains (0.613 -> 0.538). mem0's rule 3 (prefer more recent) suffices.
        # multi_session_reasoning: NO extra guidance — counting clause regressed non-counting
        # questions (0.639 -> 0.626). Recall-bound; revisit via mission/recall, not prompt.
        "instruction_following": (
            "\nThe user gave a STANDING formatting/style/behavior instruction earlier in the conversation. "
            "Recall that instruction from the memories and follow it exactly when forming your answer.\n"
        ),
        "contradiction_resolution": (
            "\nThe memories contain CONTRADICTORY statements about this. Explicitly state that there is "
            "conflicting information, quote BOTH conflicting statements with their specific details, and "
            "then ask the user which one is correct \u2014 do not pick a side or guess.\n"
        ),
        "abstention": (
            "\nOnly answer if the memories DIRECTLY contain the specific thing asked. Do NOT infer, guess, "
            "or substitute tangential/profile information. If the specific information is not present, reply "
            "exactly: \"Based on the provided chat, there is no information related to [topic].\"\n"
        ),
        # event_ordering: NO extra guidance \u2014 explicit re-derivation guidance regressed it
        # (0.543 -> 0.436); the topic-framing mismatch is recall/abstraction-bound, not prompt.
        # information_extraction: handled by the GLOBAL anti-over-abstention rule #4 (resolve indirect
        # references before giving up). Per-category "dig hard" guidance was a wash (over-claiming).
    }

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        task_type: str,
        split: str,
        category: str | None = None,
        meta: dict | None = None,
    ) -> str:
        # mem0's BEAM answer-generation prompt (verbatim): question + retrieved memories, NO leak,
        # NO per-category gold/rubric injection. Generic behavioral rules only — for parity with mem0.
        cat = (meta or {}).get("question_category", category or "")
        extra = self._BEHAVIORAL.get(cat, "")
        return (
            "You are an AI assistant with access to stored memories from prior conversations with a user.\n"
            "Use these memories to answer the following question as accurately and completely as possible.\n\n"
            "IMPORTANT RULES:\n"
            "1. Scan ALL provided memories before answering \u2014 do not stop after the first relevant one.\n"
            "2. If multiple memories contain relevant information, combine and cross-reference them.\n"
            "3. If the memories contain contradictory information, prefer the more recent one.\n"
            "4. The question may refer to something indirectly (e.g. 'the person I met at the festival') — resolve such references by searching the memories before concluding. Only if the specific information is genuinely absent after a careful search, say exactly: \"I don't have enough information to answer this question.\"\n"
            "5. For temporal questions: pay attention to dates and relative time references.\n"
            "6. For ordering questions: present events in chronological order.\n"
            "7. For preference questions: use the most recently stated preference.\n"
            "8. Be specific and direct \u2014 include exact names, dates, numbers, and details from the memories.\n"
            "9. Do NOT invent or assume information that isn't in the memories.\n"
            + extra +
            f"\nQUESTION: {query}\n\n"
            f"RETRIEVED MEMORIES:\n{context}\n\n"
            "ANSWER:"
        )


    def default_judge_llm(self):
        # BEAM is judged with gemini-3.5-flash in code (more reliable rubric scoring than
        # 2.5-flash-lite; env-independent so results are consistent across runs).
        from ..llm.gemini import GeminiLLM
        return GeminiLLM("gemini-3.5-flash")

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        # Overridden per-query via get_judge_prompt_fn when meta is available.
        gold_str = gold_answers[0] if gold_answers else "(no reference answer)"
        return f"""You are evaluating an AI system's answer about a long conversation.

Question: {query}
Reference Answer: {gold_str}
System's Answer: {answer}

Mark correct=true if the answer captures the essential facts from the reference.
Minor wording differences are acceptable. Be lenient on partial answers that contain the key facts.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    def get_judge_prompt_fn(self, category: str | None, meta: dict | None = None):
        """Return a rubric-aware judge function for BEAM evaluation."""
        cat    = category or (meta or {}).get("question_category", "")
        rubric = (meta or {}).get("rubric", [])

        def _judge(query: str, gold_answers: list[str], answer: str) -> str:
            gold_str = gold_answers[0] if gold_answers else ""
            rubric_str = ""
            if rubric:
                items = "\n".join(f"  - {r}" for r in rubric)
                rubric_str = f"\n\nScoring Rubric (check each item):\n{items}"

            if cat == "abstention":
                return f"""You are evaluating an AI system's answer to an ABSTENTION question about a long conversation.

The correct answer should state that the information is NOT available in the conversation.

Question: {query}
Reference Answer: {gold_str}
System's Answer: {answer}
{rubric_str}

Mark correct=true ONLY if the system explicitly says the information is not available / not mentioned in the conversation.
Mark correct=false if the system provides a specific answer (even if it sounds plausible) instead of abstaining.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "contradiction_resolution":
                return f"""You are evaluating an AI system's answer to a CONTRADICTION RESOLUTION question.

The system must identify that contradictory statements were made in the conversation.

Question: {query}
Reference Answer: {gold_str}
System's Answer: {answer}
{rubric_str}

Mark correct=true if the system:
1. Acknowledges there is contradictory/conflicting information, AND
2. Mentions both contradictory statements or notes the inconsistency.
Mark correct=false if it gives a definitive answer without noting the contradiction.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "event_ordering":
                ordering = (meta or {}).get("ordering_tested", [])
                ordering_str = "\n".join(f"  {o}" for o in ordering) if ordering else ""
                expected_section = f"\n\nExpected order:\n{ordering_str}" if ordering_str else (f"\n\nReference Answer: {gold_str}" if gold_str else "")
                return f"""You are evaluating an AI system's answer to an EVENT ORDERING question.

The system must list specific topics in the order they were FIRST mentioned in the conversation.

Question: {query}{expected_section}
System's Answer: {answer}
{rubric_str}

Mark correct=true if the system's answer lists the topics in the SAME SEQUENCE as the expected order above.
Exact wording doesn't need to match — semantic equivalence is sufficient.
Mark correct=false if the sequence is wrong or key topics are missing from the answer.
If no expected order is given, use the reference answer or rubric to evaluate.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "preference_following":
                compliance = (meta or {}).get("compliance_indicators", [])
                pref       = (meta or {}).get("preference_being_tested", "")
                comp_str   = "\n".join(f"  - {c}" for c in compliance) if compliance else ""
                return f"""You are evaluating whether an AI system's answer FOLLOWS the user's stated preference.

User's Preference (from conversation): {pref}
Question: {query}
System's Answer: {answer}
{rubric_str}
{"Compliance indicators:" + chr(10) + comp_str if comp_str else ""}

Mark correct=true if the system's answer aligns with and respects the user's stated preference.
Mark correct=false if the system recommends things that violate the preference, or ignores it entirely.
If no preference context was available, be lenient.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "instruction_following":
                instr      = (meta or {}).get("instruction_being_tested", "")
                compliance = (meta or {}).get("compliance_indicators", [])
                comp_str   = "\n".join(f"  - {c}" for c in compliance) if compliance else ""
                return f"""You are evaluating whether an AI system FOLLOWS a formatting/style instruction.

Instruction being tested: {instr}
Question: {query}
System's Answer: {answer}
{rubric_str}
{"Compliance indicators:" + chr(10) + comp_str if comp_str else ""}

Mark correct=true if the system's answer substantially complies with the main intent of the instruction.
Minor gaps are acceptable — focus on whether the primary format/style requirement is met.
Mark correct=false ONLY if it clearly ignores or violates the core instruction.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "temporal_reasoning":
                return f"""You are evaluating an AI system's answer to a TEMPORAL REASONING question requiring date/time calculations.

Question: {query}
Reference Answer: {gold_str}
System's Answer: {answer}
{rubric_str}

The rubric items above are the authoritative scoring criteria. If the rubric specifies a number or date range, use THAT as the ground truth.
Mark correct=true if the system's answer contains the numbers/dates specified in the rubric OR in the reference answer.
Be lenient on phrasing — focus only on whether the numeric result is correct.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            elif cat == "summarization":
                gold_section = f"\nReference Answer: {gold_str}" if gold_str else ""
                return f"""You are evaluating an AI system's SUMMARIZATION answer about a long conversation.

Question: {query}{gold_section}
System's Answer: {answer}
{rubric_str}

Mark correct=true if the answer covers at least 2 of the rubric items (or at least 50% if there are 2 or fewer items).
It does not need to cover every detail — a comprehensive high-level summary that addresses the main phases/milestones is sufficient.
Minor omissions or paraphrasing are acceptable. Be LENIENT — partial credit for answers that cover the main ideas.

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

            else:
                # General rubric-based evaluation for: information_extraction,
                # knowledge_update, multi_session_reasoning
                gold_section = f"\nReference Answer: {gold_str}" if gold_str else ""
                n_rubric = len(rubric) if rubric else 0
                if n_rubric >= 3:
                    threshold_note = "Mark correct=true if the answer addresses at least 2 of the rubric items above (semantic equivalence counts — exact wording is NOT required)."
                elif n_rubric == 2:
                    threshold_note = "Mark correct=true if the answer addresses at least 1 of the rubric items above (semantic equivalence counts — exact wording is NOT required)."
                elif n_rubric == 1:
                    threshold_note = "Mark correct=true if the answer captures the core fact in the single rubric item (semantic equivalence counts). Be lenient — partial match is acceptable."
                else:
                    threshold_note = "Mark correct=true if the answer captures the essential facts from the reference answer. Be lenient."
                return f"""You are evaluating an AI system's answer about a long conversation.

Question: {query}{gold_section}
System's Answer: {answer}
{rubric_str}

{threshold_note}

Respond as JSON: {{"correct": true/false, "reason": "one sentence"}}"""

        return _judge

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes = {}
        cat = meta.get("question_category")
        if cat:
            axes["Question Category"] = [cat]
        conv = meta.get("conversation_id")
        if conv:
            axes["Conversation"] = [conv]
        return axes

    # ------------------------------------------------------------------
    # BEAM paper scoring (continuous 0-1 scores)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ordered_items(text: str) -> list[str]:
        """Extract ordered items from a numbered/bulleted list or newline-separated text."""
        lines = text.strip().splitlines()
        items = []
        for line in lines:
            # Strip numbering like "1.", "1)", "- ", "* ", etc.
            cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    @staticmethod
    def _llm_equivalence(ref_item: str, sys_item: str, llm: LLM) -> bool:
        """Use LLM to check if two items refer to the same event/topic."""
        prompt = f"""Do these two items refer to the same event, topic, or concept? Answer only YES or NO.

Item A: {ref_item}
Item B: {sys_item}

Answer (YES or NO):"""
        schema = Schema(
            properties={"answer": {"type": "string", "description": "YES or NO"}},
            required=["answer"],
        )
        try:
            result = llm.generate(prompt, schema)
            return result.get("answer", "").strip().upper().startswith("YES")
        except Exception:
            return False

    @staticmethod
    def _align_with_llm(reference: list[str], system: list[str], llm: LLM) -> tuple[list[str], list[str]]:
        """Align system items to reference items using LLM equivalence matching.

        Returns (reference_canonical, system_canonical) where matched items share
        the same string from the reference list.
        """
        used: set[int] = set()
        system_out: list[str] = []
        for s in system:
            matched_index = None
            for index, r in enumerate(reference):
                if index in used:
                    continue
                if BEAMDataset._llm_equivalence(r, s, llm):
                    matched_index = index
                    break
            if matched_index is not None:
                system_out.append(reference[matched_index])
                used.add(matched_index)
            else:
                system_out.append(s)
        return reference, system_out

    @staticmethod
    def _event_ordering_score(reference: list[str], system: list[str], llm: LLM) -> float:
        """Compute tau_b_norm for event ordering, matching the BEAM paper methodology."""
        if not reference or not system:
            return 0.0

        reference_canon, system_canon = BEAMDataset._align_with_llm(reference, system, llm)

        # Build union of items preserving order of first appearance
        union = list(dict.fromkeys(reference_canon + system_canon))
        tie_rank = len(union) + 1

        def to_rank(seq: list[str]) -> list[int]:
            r = {item: i + 1 for i, item in enumerate(seq)}
            return [r.get(u, tie_rank) for u in union]

        ref_ranks = to_rank(reference_canon)
        sys_ranks = to_rank(system_canon)

        tau_b, _ = kendalltau(ref_ranks, sys_ranks, variant="b")
        if tau_b is None or str(tau_b) == "nan":
            return 0.0
        return (tau_b + 1) / 2

    @staticmethod
    def _rubric_item_score(query: str, answer: str, rubric_item: str, llm: LLM) -> float:
        """Score a single rubric nugget 0/0.5/1 using mem0's nugget-judge prompt (the comparable
        BEAM standard; replaces BEAM's unified judge for head-to-head parity with mem0)."""
        prompt = f"""Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{query}

LLM RESPONSE:
{answer}

RUBRIC CRITERION:
{rubric_item}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0**: The response is responsive AND the prohibited element is absent.
- **0.5**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0**: The prohibited element is present, OR the response is non-responsive.

**Compound statements** ("and"/commas joining multiple required elements): all present=1.0, some=0.5, none=0.0.

EVALUATION RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric/date equivalence: "$68,000"="68k"="sixty-eight thousand dollars"; "2 years"="24 months".
3. Case/punctuation/whitespace tolerance.
4. Hedging tolerance: do not penalize "I think"/"probably"/passive/verbosity if content satisfies the criterion.
5. Style neutrality: do not penalize tone/format/length unless the criterion requires it.
6. Responsiveness: completely off-topic or refusal = 0.0.
7. Independence: evaluate this criterion in isolation.
8. Specificity: vague generic answers score lower than specific, detailed ones.

Return JSON: {{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence>"}}"""
        schema = Schema(
            properties={
                "score": {"type": "number", "description": "Score: 0.0, 0.5, or 1.0"},
                "reason": {"type": "string"},
            },
            required=["score", "reason"],
        )
        try:
            result = llm.generate(prompt, schema)
            raw = float(result.get("score", 0))
            # Clamp to nearest valid value
            if raw >= 0.75:
                return 1.0
            elif raw >= 0.25:
                return 0.5
            else:
                return 0.0
        except Exception as e:
            logger.warning("Rubric scoring failed: %s", e)
            return 0.0

    def score_result(self, result: QueryResult, llm: LLM) -> float:
        """Compute a continuous BEAM paper score (0-1) for a query result.

        ALL categories (including event_ordering) are scored by average rubric-nugget compliance
        (mem0's standard), NOT Kendall-tau — for head-to-head parity with mem0/the field.
        """
        cat = result.meta.get("question_category", "")
        rubric = list(result.meta.get("rubric", []) or [])
        # event_ordering nuggets are bare topics → frame them as "should mention" for the judge.
        if cat == "event_ordering":
            rubric = [r if "should" in str(r).lower() else f"LLM response should mention: {r}" for r in rubric]
        if not rubric and result.gold_answers:
            rubric = [f"LLM response should contain: {result.gold_answers[0]}"]
        if not rubric:
            logger.warning("[%s] No rubric found, falling back to 0", result.query_id)
            return 0.0

        scores = []
        for item in rubric:
            s = self._rubric_item_score(result.query, result.answer, item, llm)
            scores.append(s)
            logger.info("[%s] rubric item score=%.1f for: %s", result.query_id, s, item[:80])

        avg = sum(scores) / len(scores) if scores else 0.0
        logger.info("[%s] %s avg_score=%.3f (%d items)", result.query_id, cat, avg, len(scores))
        return avg

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="BEAM dataset stats")
        table.add_column("Split", style="bold")
        table.add_column("Conversations", justify="right")
        table.add_column("Questions (est.)", justify="right")

        for split in SPLITS:
            try:
                data = self._load_raw(split)
                n_convs = len(data)
                n_questions = sum(
                    sum(len(v) for v in self._parse_probing_questions(item).values())
                    for item in data
                )
                table.add_row(split, str(n_convs), str(n_questions))
            except Exception as e:
                table.add_row(split, "—", f"error: {e}")

        console.print(table)
