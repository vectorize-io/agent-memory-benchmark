"""
LoComo with its broken questions removed.

Some LoComo questions cannot be answered from the conversation they ask about: the gold
answer names something the conversation never says (a book title that appears in no turn
and no image caption), the gold contradicts the conversation (a count that misses a
visit the conversation mentions), or the question names the wrong person (it asks about
Melanie's bowl; the bowl is Caroline's). A system that answers those correctly is guessing, and one
that answers them faithfully is marked wrong.

Each removed question is listed in ``locomo_fixed_exclusions.json`` with the reason, checked
against the conversation's turns and image captions, which are what gets ingested. A question
is removed only when the conversation cannot support its gold answer, never because it is
hard: an answer that needs two turns joined, or one stated only in an image caption, stays.

Same conversations, sessions, dates and judge as ``locomo``; only the query list differs.
"""
import json
from pathlib import Path

from .locomo import LoComoDataset
from ..models import Query

_EXCLUSIONS_PATH = Path(__file__).with_name("locomo_fixed_exclusions.json")


def load_exclusions() -> dict[str, str]:
    """``{query_id: reason}`` for every removed question."""
    return json.loads(_EXCLUSIONS_PATH.read_text(encoding="utf-8"))


class LoComoFixedDataset(LoComoDataset):
    name = "locomo-fixed"
    published = True
    description = (
        "LoComo without the questions its conversations cannot answer "
        "(gold answer absent from the conversation, gold contradicting it, or the question naming the wrong person)."
    )

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        excluded = load_exclusions()
        queries = [q for q in super().load_queries(split, category) if q.id not in excluded]
        return queries[:limit] if limit else queries
