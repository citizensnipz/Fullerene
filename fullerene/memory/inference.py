"""Deterministic tag and salience inference for Fullerene Memory v1/v2.

These rules stay intentionally small and inspectable:

- No model calls, embeddings, vector indices, or external NLP libraries.
- Matching is lowercase, deterministic, and respects token boundaries.
- Salience is derived from event kind plus the effective tag set only.
- Memory v2 adds light-weight domain detection on top of the same rule
  table; domains are coarse buckets such as ``reading_books`` or
  ``project_software`` used for role-aware hybrid retrieval. They are *not*
  hardcoded special cases per topic; new domains follow the same shape.

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
    ("correction", ("don't", "wrong", "missed", "skipped", "failed")),
    (
        "book",
        (
            "book",
            "books",
            "novel",
            "novels",
            "fiction",
            "non-fiction",
            "nonfiction",
            "autobiography",
            "autobiographies",
            "biography",
            "biographies",
        ),
    ),
    ("read", ("read", "reading", "reads", "reader")),
    (
        "sci-fi",
        ("sci-fi", "scifi", "sci fi", "science fiction"),
    ),
    (
        "outdoors",
        ("outdoor", "outdoors", "outside", "beach", "ocean", "water"),
    ),
    ("scuba", ("scuba", "diving", "dive", "snorkel", "snorkeling")),
    ("surf", ("surf", "surfing", "surfboard")),
    (
        "fullerene",
        ("fullerene",),
    ),
    (
        "code",
        ("code", "coding", "codebase", "github", "git"),
    ),
    (
        "planner",
        ("planner", "planning",),
    ),
    (
        "context",
        ("context", "context window", "contextwindow"),
    ),
    (
        "behavior",
        ("behavior", "behavioral",),
    ),
    (
        "nexus",
        ("nexus",),
    ),
    (
        "task",
        ("task", "tasks", "todo", "to-do",),
    ),
    (
        "work",
        ("work", "working", "finish", "complete", "next steps"),
    ),
)


# Domain definitions for Memory v2. A domain is a bucket of related tags;
# the first tag list whose intersection with inferred tags is non-empty
# wins. The ordering controls precedence so behavior stays deterministic.
DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "reading_books",
        ("book", "read", "sci-fi"),
    ),
    (
        "outdoors_water",
        ("scuba", "surf", "outdoors"),
    ),
    (
        "project_software",
        (
            "fullerene",
            "code",
            "planner",
            "context",
            "behavior",
            "nexus",
            "memory",
            "policy",
            "goals",
        ),
    ),
    (
        "task_work",
        ("task", "work", "verification"),
    ),
)

# Salience constants exposed so tests and docs can refer to them by name.
BASE_SALIENCE: float = 0.3
USER_MESSAGE_BOOST: float = 0.2
HARD_RULE_BOOST: float = 0.2
URGENT_BOOST: float = 0.1
CORRECTION_BOOST: float = 0.2
AUTHORITY_BOOST: float = 0.1
COMMUNICATION_BOOST: float = 0.05

HARD_RULE_TAG = "hard-rule-candidate"
URGENT_TAG = "urgent"
CORRECTION_TAG = "correction"
AUTHORITY_TAG = "authority"
COMMUNICATION_TAG = "communication"


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


def infer_domain(content: str, tags: Iterable[str] | None = None) -> str | None:
    """Return the deterministic domain bucket inferred from ``content``/``tags``.

    Domains group related tags into coarse buckets such as ``reading_books``
    or ``project_software`` so role-aware hybrid retrieval can compare
    across phrasings without hardcoding individual topics.

    Returns ``None`` when no domain matches.
    """
    tag_set: set[str] = set()
    if tags is not None:
        for raw_tag in tags:
            cleaned = str(raw_tag).strip().lower()
            if cleaned:
                tag_set.add(cleaned)
    for inferred_tag in infer_tags(content):
        tag_set.add(inferred_tag)
    if not tag_set:
        return None
    for domain, domain_tags in DOMAIN_RULES:
        if any(domain_tag in tag_set for domain_tag in domain_tags):
            return domain
    return None


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
    - user message (+0.2)
    - ``hard-rule-candidate`` tag (+0.2)
    - ``urgent`` tag (+0.1)
    - ``correction`` tag (+0.2)
    - ``authority`` tag (+0.1)
    - ``communication`` tag (+0.05)

    The result is always clamped to the inclusive range ``[0.0, 1.0]``.
    """
    tag_set = set(merge_tags(tags, infer_tags(content)))

    score = float(base)
    if is_user_message:
        score += USER_MESSAGE_BOOST
    if HARD_RULE_TAG in tag_set:
        score += HARD_RULE_BOOST
    if URGENT_TAG in tag_set:
        score += URGENT_BOOST
    if CORRECTION_TAG in tag_set:
        score += CORRECTION_BOOST
    if AUTHORITY_TAG in tag_set:
        score += AUTHORITY_BOOST
    if COMMUNICATION_TAG in tag_set:
        score += COMMUNICATION_BOOST

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
    tag_set = set(merge_tags(tags, infer_tags(content)))

    breakdown: dict[str, float] = {"base": float(base)}
    if is_user_message:
        breakdown["user_message"] = USER_MESSAGE_BOOST
    if HARD_RULE_TAG in tag_set:
        breakdown["hard_rule_candidate_tag"] = HARD_RULE_BOOST
    if URGENT_TAG in tag_set:
        breakdown["urgent_tag"] = URGENT_BOOST
    if CORRECTION_TAG in tag_set:
        breakdown["correction_tag"] = CORRECTION_BOOST
    if AUTHORITY_TAG in tag_set:
        breakdown["authority_tag"] = AUTHORITY_BOOST
    if COMMUNICATION_TAG in tag_set:
        breakdown["communication_tag"] = COMMUNICATION_BOOST

    raw_total = sum(breakdown.values())
    breakdown["total"] = _clamp_unit(raw_total)
    return breakdown


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))
