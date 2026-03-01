"""
parse_sanitizer.py
===================
Guards against small SLM hallucinating structured fields
that the user never mentioned.

Rules:
- chapter_number must only be kept if the raw query contains
  an explicit numeric chapter reference
- activity_number / exercise_number same rule
- topic is always kept as-is
"""

import re
from typing import Any


def _has_chapter_mention(query: str) -> bool:
    """True if the user explicitly mentioned a chapter number."""
    return bool(re.search(
        r'\b(chapter|ch\.?)\s*\d+\b',
        query,
        re.IGNORECASE
    ))

def _has_activity_mention(query: str) -> bool:
    return bool(re.search(
        r'\bactivity\s*\d+\b',
        query,
        re.IGNORECASE
    ))

def _has_exercise_mention(query: str) -> bool:
    return bool(re.search(
        r'\bexercise\s*[\d.]+\b',
        query,
        re.IGNORECASE
    ))


def sanitize(parsed: dict, raw_query: str) -> dict:
    """
    Remove fields the SLM filled in but the user never mentioned.
    Returns a cleaned copy of the parsed dict.
    """
    cleaned = dict(parsed)

    if not _has_chapter_mention(raw_query):
        if cleaned.get("chapter_number") is not None:
            cleaned["chapter_number"] = None

    if not _has_activity_mention(raw_query):
        if cleaned.get("activity_number") is not None:
            cleaned["activity_number"] = None

    if not _has_exercise_mention(raw_query):
        if cleaned.get("exercise_number") is not None:
            cleaned["exercise_number"] = None

    return cleaned
