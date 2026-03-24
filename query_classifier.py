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

    definition_patterns = [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bdefine\b",
        r"\bmeaning of\b",
        r"\bstands for\b",
        r"\brefers to\b",
    ]
    comparison_patterns = [
        r"\bdifference between\b",
        r"\bdiffer from\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bbetter than\b",
        r"\bsimilarities\b",
    ]
    summarization_patterns = [
        r"\bsummarize\b",
        r"\bsummary of\b",
        r"\boverview of\b",
        r"\btl;dr\b",
        r"\bin short\b",
        r"\bkey points\b",
    ]
    navigation_patterns = [
        r"^where\b",
        r"\bwhich section\b",
        r"\bwhere in\b",
        r"\blocated\b",
        r"\bpage\b",
        r"\bchapter\b",
        r"\bfind\b",
    ]
    reasoning_patterns = [
        r"^how\b",
        r"^why\b",
        r"\bexplain how\b",
        r"\bexplain why\b",
        r"\breason\b",
        r"\bimpact\b",
        r"\bcause\b",
    ]

    # Check comparison first to avoid misclassifying "what is the difference..."
    if any(re.search(pattern, q) for pattern in comparison_patterns):
        return QueryType.COMPARISON

    # Definition-style questions
    if any(re.search(pattern, q) for pattern in definition_patterns):
        return QueryType.DEFINITION

    # Summarization requests
    if any(re.search(pattern, q) for pattern in summarization_patterns):
        return QueryType.SUMMARIZATION

    # Navigation / locating information
    if any(re.search(pattern, q) for pattern in navigation_patterns):
        return QueryType.NAVIGATION

    # Reasoning / how / why
    if any(re.search(pattern, q) for pattern in reasoning_patterns):
        return QueryType.REASONING

    return QueryType.OTHER
