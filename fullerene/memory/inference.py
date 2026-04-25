"""Deterministic tag and salience inference for Fullerene Memory v1.

These rules are intentionally simple, lowercase, and transparent:

- No model calls, embeddings, vector indices, or external NLP libraries.
- All matching is case-insensitive and respects token boundaries.
- Tag rules and salience signals are declared as plain tuples so they can
  be inspected, audited, or extended without any clever framework code.

Future affect / prosody work may influence salience, but is not implemented
here. See ai/project/architecture.md for the Memory roadmap.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from fullerene.memory.models import normalize_tags

# Lowercase triggers per tag. The tuple ordering controls the inferred-tag
# output order so behavior is deterministic across runs and platforms.
TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "communication",
        ("email", "emails", "inbox", "inboxes", "message", "messages"),
    ),
    ("authority", ("boss", "manager", "managers", "lead", "leads")),
    ("urgent", ("urgent", "asap", "immediately", "now")),
    ("hard-rule-candidate", ("never", "always", "don't ever", "must")),
    ("bug", ("bug", "bugs", "error", "errors", "failing", "broken")),
    ("verification", ("test", "tests", "verify", "validation")),
    ("memory", ("memory", "remember", "forgot")),
    (
        "goals",
        ("goal", "goals", "objective", "objectives", "priority", "priorities"),
    ),
    (
        "policy",
        ("policy", "policies", "rule", "rules", "permission", "permissions"),
    ),
)

# Direct user instruction cues. Only contribute when the source event is a
# user message (instructions from the user, not generic system notes).
INSTRUCTION_WORDS: tuple[str, ...] = (
    "please",
    "make sure",
    "remember",
    "do not",
    "don't",
    "ensure",
)

# Generic emphasis / strong-language intensifiers.
STRONG_LANGUAGE_WORDS: tuple[str, ...] = (
    "very",
    "really",
    "extremely",
    "critically",
    "absolutely",
    "definitely",
)

# Correction / negative-feedback cues.
CORRECTION_WORDS: tuple[str, ...] = (
    "wrong",
    "incorrect",
    "mistake",
    "actually",
    "instead",
    "correction",
)

# Salience constants exposed so tests and docs can refer to them by name.
BASE_SALIENCE: float = 0.3
USER_INSTRUCTION_BOOST: float = 0.2
STRONG_LANGUAGE_BOOST: float = 0.2
HARD_RULE_BOOST: float = 0.2
URGENT_BOOST: float = 0.1
CORRECTION_BOOST: float = 0.2

HARD_RULE_TAG = "hard-rule-candidate"
URGENT_TAG = "urgent"


def _normalize_text(text: str) -> str:
    """Lowercase and fold smart quotes so triggers like ``don't`` always match."""
    return (
        text.lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )


def _trigger_pattern(trigger: str) -> re.Pattern[str]:
    # Match the trigger only when it is not adjacent to another alphanumeric
    # character, so "lead" does not fire on "leader" and "boss" does not fire
    # on "embossed". Spaces and punctuation inside the trigger are allowed.
    return re.compile(rf"(?<![a-z0-9]){re.escape(trigger)}(?![a-z0-9])")


_TAG_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (tag, tuple(_trigger_pattern(trigger) for trigger in triggers))
    for tag, triggers in TAG_RULES
)
_INSTRUCTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _trigger_pattern(word) for word in INSTRUCTION_WORDS
)
_STRONG_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _trigger_pattern(word) for word in STRONG_LANGUAGE_WORDS
)
_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _trigger_pattern(word) for word in CORRECTION_WORDS
)


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def infer_tags(content: str) -> list[str]:
    """Return deterministic tags inferred from ``content``.

    The output order follows :data:`TAG_RULES` so callers can rely on stable
    ordering across runs.
    """
    if not content:
        return []
    text = _normalize_text(content)
    inferred: list[str] = []
    for tag, patterns in _TAG_PATTERNS:
        if _matches_any(text, patterns):
            inferred.append(tag)
    return inferred


def merge_tags(*tag_groups: Iterable[str] | None) -> list[str]:
    """Merge multiple tag iterables into a single normalized, de-duplicated list.

    Earlier groups win the position when duplicates appear. Use this to combine
    explicit metadata-supplied tags with inferred tags without overwriting the
    caller's intent.
    """
    combined: list[str] = []
    for group in tag_groups:
        if not group:
            continue
        combined.extend(group)
    return normalize_tags(combined)


def compute_salience(
    *,
    content: str,
    tags: Iterable[str],
    is_user_message: bool,
    base: float = BASE_SALIENCE,
) -> float:
    """Score the importance of a memory record deterministically.

    Signals (each contributes independently, then the sum is clamped):

    - base salience (default 0.3)
    - direct user instruction language (+0.2, only on user messages)
    - strong / emphasis language (+0.2)
    - ``hard-rule-candidate`` tag (+0.2)
    - ``urgent`` tag (+0.1)
    - correction / negative-feedback language (+0.2)

    The result is always clamped to the inclusive range ``[0.0, 1.0]``.
    """
    text = _normalize_text(content)
    tag_set = {tag for tag in tags}

    score = float(base)
    if is_user_message and _matches_any(text, _INSTRUCTION_PATTERNS):
        score += USER_INSTRUCTION_BOOST
    if _matches_any(text, _STRONG_PATTERNS):
        score += STRONG_LANGUAGE_BOOST
    if HARD_RULE_TAG in tag_set:
        score += HARD_RULE_BOOST
    if URGENT_TAG in tag_set:
        score += URGENT_BOOST
    if _matches_any(text, _CORRECTION_PATTERNS):
        score += CORRECTION_BOOST

    return _clamp_unit(score)


def explain_salience(
    *,
    content: str,
    tags: Iterable[str],
    is_user_message: bool,
    base: float = BASE_SALIENCE,
) -> dict[str, float]:
    """Return a per-signal breakdown for the salience score.

    Useful for debugging / inspection ("why is this memory salient?") without
    re-implementing the scoring logic in callers.
    """
    text = _normalize_text(content)
    tag_set = {tag for tag in tags}

    breakdown: dict[str, float] = {"base": float(base)}
    if is_user_message and _matches_any(text, _INSTRUCTION_PATTERNS):
        breakdown["user_instruction"] = USER_INSTRUCTION_BOOST
    if _matches_any(text, _STRONG_PATTERNS):
        breakdown["strong_language"] = STRONG_LANGUAGE_BOOST
    if HARD_RULE_TAG in tag_set:
        breakdown["hard_rule_candidate"] = HARD_RULE_BOOST
    if URGENT_TAG in tag_set:
        breakdown["urgent_tag"] = URGENT_BOOST
    if _matches_any(text, _CORRECTION_PATTERNS):
        breakdown["correction"] = CORRECTION_BOOST

    raw_total = sum(breakdown.values())
    breakdown["total"] = _clamp_unit(raw_total)
    return breakdown


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))
