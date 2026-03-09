import re
from enum import Enum


class QueryType(str, Enum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    NAVIGATION = "navigation"
    OTHER = "other"


def classify_query(query: str) -> QueryType:
    """
    Lightweight, rule-based query classifier used to steer retrieval.

    This is intentionally simple and cheap; it can be upgraded to an LLM-based
    classifier later without changing the public interface.
    """
    q = (query or "").strip().lower()

    if not q:
        return QueryType.OTHER

    # Definition-style questions
    if re.search(r"\bwhat is\b", q) or "define " in q:
        return QueryType.DEFINITION

    # Comparison questions
    if "difference between" in q or " differ from " in q or " vs " in q:
        return QueryType.COMPARISON

    # Summarization requests
    if "summarize" in q or "summary of" in q or "overview of" in q:
        return QueryType.SUMMARIZATION

    # Navigation / locating information
    if q.startswith("where ") or "which section" in q or "located" in q:
        return QueryType.NAVIGATION

    # Reasoning / how / why
    if q.startswith("how ") or q.startswith("why ") or "explain how" in q:
        return QueryType.REASONING

    return QueryType.OTHER

