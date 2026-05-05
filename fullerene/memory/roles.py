"""Deterministic memory-role and query-intent classification for Memory v2.

Classification stays small and inspectable:

- No model calls, embeddings, or external NLP libraries.
- All matching is lowercase and respects token boundaries where useful.
- Roles describe *why* a memory was stored (preference / fact / question /
  task / feedback / outcome / unknown) so retrieval can prefer support
  memories over repeated questions.
- Query intents describe *why* the current event was asked (recommendation /
  planning / factual / unknown) so hybrid retrieval can apply role-aware
  bonuses and penalties.

Domain detection lives in :mod:`fullerene.memory.inference` because it shares
the deterministic tag rules.
"""

from __future__ import annotations

import re
from enum import Enum


class MemoryRole(str, Enum):
    """High-level reason a memory exists."""

    PREFERENCE = "preference"
    FACT = "fact"
    QUESTION = "question"
    TASK = "task"
    FEEDBACK = "feedback"
    OUTCOME = "outcome"
    UNKNOWN = "unknown"


class QueryIntent(str, Enum):
    """High-level intent behind the current event being asked about."""

    RECOMMENDATION = "recommendation"
    PLANNING = "planning"
    FACTUAL = "factual"
    UNKNOWN = "unknown"


PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi\s+(?:really\s+)?like\s+", re.IGNORECASE),
    re.compile(r"\bi\s+(?:really\s+)?love\s+", re.IGNORECASE),
    re.compile(r"\bi\s+(?:really\s+)?enjoy\s+", re.IGNORECASE),
    re.compile(r"\bi\s+(?:really\s+)?prefer\s+", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+into\s+", re.IGNORECASE),
    re.compile(r"\bmy\s+favorite\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:really\s+)?(?:hate|dislike|avoid)\s+", re.IGNORECASE),
    re.compile(r"\bi\s+do(?:\s+not|n['\u2019]t)\s+like\s+", re.IGNORECASE),
)

QUESTION_STARTERS: tuple[str, ...] = (
    "what",
    "how",
    "why",
    "should",
    "can",
    "could",
    "would",
    "where",
    "when",
    "who",
    "which",
    "do you",
    "did you",
    "is there",
    "are there",
)

TASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi\s+need\s+to\s+", re.IGNORECASE),
    re.compile(r"\bi\s+should\s+", re.IGNORECASE),
    re.compile(r"\bi\s+have\s+to\s+", re.IGNORECASE),
    re.compile(r"\bi\s+must\s+", re.IGNORECASE),
    re.compile(r"\bremember\s+to\s+", re.IGNORECASE),
    re.compile(r"\bdon['\u2019]t\s+forget\s+to\s+", re.IGNORECASE),
    re.compile(r"\btodo\b", re.IGNORECASE),
    re.compile(r"\bto[-_\s]?do\b", re.IGNORECASE),
)

FEEDBACK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthat\s+worked\b", re.IGNORECASE),
    re.compile(r"\bthat\s+(?:was|is)\s+(?:right|correct|wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\b(?:that['\u2019]s|its|it['\u2019]s)\s+(?:right|correct|wrong|incorrect|good|bad)\b", re.IGNORECASE),
    re.compile(r"\b(?:nice\s+job|good\s+job|well\s+done|thanks)\b", re.IGNORECASE),
    re.compile(r"\b(?:bad|wrong|broken|failed|incorrect|fail)\b", re.IGNORECASE),
)

OUTCOME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:succeeded|success|completed|done|finished|shipped)\b", re.IGNORECASE),
    re.compile(r"\b(?:failed|failure|errored|crashed|broke)\b", re.IGNORECASE),
    re.compile(r"\boutcome\s+was\b", re.IGNORECASE),
    re.compile(r"\bresult(?:s|ed)?\s+(?:was|in|were)\b", re.IGNORECASE),
)


RECOMMENDATION_PHRASES: tuple[str, ...] = (
    "what should i",
    "what should we",
    "what kind of",
    "what type of",
    "what book",
    "what movie",
    "what should",
    "recommend",
    "recommendation",
    "suggest",
    "suggestion",
    "any ideas",
    "help me pick",
    "help me choose",
)

PLANNING_PHRASES: tuple[str, ...] = (
    "what next",
    "next step",
    "next steps",
    "what are the next steps",
    "how should i",
    "how should we",
    "make a plan",
    "plan this",
    "break this down",
    "what should i do next",
    "what should we do next",
)

FACTUAL_PHRASES: tuple[str, ...] = (
    "what is",
    "what are",
    "who is",
    "who are",
    "what do you know",
    "tell me about",
    "explain",
    "define",
    "describe",
)


def _normalize(content: str) -> str:
    return (
        content.strip()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )


def _starts_with_question_word(text: str) -> bool:
    lowered = text.lower().lstrip()
    for starter in QUESTION_STARTERS:
        if lowered == starter or lowered.startswith(starter + " ") or lowered.startswith(starter + "'"):
            return True
    return False


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_memory_role(content: str) -> MemoryRole:
    """Return the deterministic memory role for ``content``.

    Order of precedence is intentional: preferences win over questions because
    a sentence like "I like to read books" is a preference even if it contains
    a question-shaped fragment elsewhere; tasks beat feedback because "I need
    to fix this bug" is a task; explicit outcome language wins over generic
    feedback wording. Unknown is the safe default for short factual asides.
    """
    if not content or not content.strip():
        return MemoryRole.UNKNOWN

    text = _normalize(content)

    if _matches_any(text, PREFERENCE_PATTERNS):
        return MemoryRole.PREFERENCE

    if _matches_any(text, TASK_PATTERNS):
        return MemoryRole.TASK

    if _matches_any(text, OUTCOME_PATTERNS):
        return MemoryRole.OUTCOME

    if _matches_any(text, FEEDBACK_PATTERNS):
        return MemoryRole.FEEDBACK

    stripped = text.rstrip()
    if stripped.endswith("?") or _starts_with_question_word(stripped):
        return MemoryRole.QUESTION

    if _is_factual_statement(text):
        return MemoryRole.FACT

    return MemoryRole.UNKNOWN


def _is_factual_statement(text: str) -> bool:
    """Detect simple `X is Y` factual phrasing.

    Kept narrow on purpose: this is only used to upgrade some statements from
    UNKNOWN to FACT when the sentence asserts something. False negatives are
    fine; UNKNOWN works as a safe fallback.
    """
    factual_pattern = re.compile(
        r"^\s*[a-z0-9][a-z0-9\s,'-]*\s+(?:is|are|was|were)\s+",
        re.IGNORECASE,
    )
    return bool(factual_pattern.search(text))


def classify_query_intent(content: str) -> QueryIntent:
    """Return the deterministic query intent for the current event.

    Used by hybrid retrieval to decide which role bonus/penalty to apply.
    """
    if not content or not content.strip():
        return QueryIntent.UNKNOWN

    lowered = _normalize(content).lower()

    for phrase in RECOMMENDATION_PHRASES:
        if phrase in lowered:
            return QueryIntent.RECOMMENDATION

    for phrase in PLANNING_PHRASES:
        if phrase in lowered:
            return QueryIntent.PLANNING

    for phrase in FACTUAL_PHRASES:
        if phrase in lowered:
            return QueryIntent.FACTUAL

    return QueryIntent.UNKNOWN


def coerce_memory_role(value: object) -> MemoryRole:
    """Coerce an arbitrary value (string or enum) into :class:`MemoryRole`."""
    if isinstance(value, MemoryRole):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        for role in MemoryRole:
            if role.value == candidate:
                return role
    return MemoryRole.UNKNOWN


def coerce_query_intent(value: object) -> QueryIntent:
    """Coerce an arbitrary value (string or enum) into :class:`QueryIntent`."""
    if isinstance(value, QueryIntent):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        for intent in QueryIntent:
            if intent.value == candidate:
                return intent
    return QueryIntent.UNKNOWN
