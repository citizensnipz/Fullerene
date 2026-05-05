"""Hybrid memory retrieval scoring for Fullerene Memory v2.

Memory v2 keeps SQLite as the source of truth and layers a deterministic
hybrid score on top of:

- ``semantic_similarity``  (0.35) - cosine over stored embeddings; ``0.0``
  when embeddings are unavailable.
- ``tag_overlap``          (0.20) - Jaccard-style overlap of inferred and
  explicit tags between event and memory.
- ``salience``             (0.15) - the memory's stored deterministic
  salience score from Memory v1.
- ``recency``              (0.10) - smooth time-decay over days since the
  memory was created.
- ``domain_match``         (0.10) - 1.0 when the inferred event domain
  equals the memory domain, else 0.0.
- ``role_bonus``           (0.10) - small bonus for role pairs that the
  intent classifier prefers (e.g. recommendation queries + preference
  memories in the same domain).
- ``role_penalty``         (subtracted) - prior-question penalties so
  retrieval does not over-rank repeated questions.

The final score is sortable; it can exceed ``1.0`` when every component
fires, but ordering is what matters for retrieval. ``explain_hybrid_score``
returns the per-component breakdown so the runtime can keep retrieval
inspectable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from fullerene.memory.embeddings import cosine_similarity
from fullerene.memory.inference import infer_domain
from fullerene.memory.models import MemoryRecord
from fullerene.memory.roles import (
    MemoryRole,
    QueryIntent,
    classify_memory_role,
    classify_query_intent,
    coerce_memory_role,
)
from fullerene.memory.scoring import extract_event_tags, recency_score, tokenize
from fullerene.nexus.models import Event


HYBRID_SEMANTIC_WEIGHT = 0.35
HYBRID_TAG_WEIGHT = 0.20
HYBRID_SALIENCE_WEIGHT = 0.15
HYBRID_RECENCY_WEIGHT = 0.10
HYBRID_DOMAIN_WEIGHT = 0.10
HYBRID_ROLE_BONUS_WEIGHT = 0.10

# Concrete bonus/penalty magnitudes. The role bonus weight above caps how
# much the bonus can move the final score, but each preset records the raw
# magnitude before the cap so the breakdown stays transparent.
ROLE_BONUS_RECOMMENDATION_PREFERENCE = 0.25
ROLE_BONUS_PLANNING_TASK = 0.20
ROLE_BONUS_FACTUAL_FACT = 0.15
ROLE_BONUS_GENERIC_OUTCOME = 0.05

ROLE_PENALTY_RECOMMENDATION_QUESTION = 0.25
ROLE_PENALTY_DUPLICATE_QUESTION = 0.30
ROLE_PENALTY_OLD_UNANSWERED_QUESTION = 0.20

# Soft bound on how stale a question memory has to be (in days) before the
# duplicate-question penalty escalates. Avoids penalizing the just-stored
# event itself when it is the only question record of its kind.
QUESTION_RECENCY_DUPLICATE_DAYS = 7.0


def _domain_for_event(event: Event) -> str | None:
    return infer_domain(event.content, extract_event_tags(event))


def _domain_for_memory(memory: MemoryRecord) -> str | None:
    explicit = memory.domain
    if explicit:
        return explicit
    return infer_domain(memory.content, memory.tags)


def _memory_role(memory: MemoryRecord) -> MemoryRole:
    direct = coerce_memory_role(memory.role)
    if direct != MemoryRole.UNKNOWN:
        return direct
    metadata_role = memory.metadata.get("memory_role") if isinstance(memory.metadata, dict) else None
    if metadata_role is not None:
        coerced = coerce_memory_role(metadata_role)
        if coerced != MemoryRole.UNKNOWN:
            return coerced
    return classify_memory_role(memory.content)


def _semantic_similarity_value(
    event_vector: Sequence[float] | None,
    memory_vector: Sequence[float] | None,
) -> float:
    if not event_vector or not memory_vector:
        return 0.0
    return max(0.0, cosine_similarity(event_vector, memory_vector))


def _tag_overlap(event_tags: set[str], memory_tags: Iterable[str]) -> tuple[float, list[str]]:
    memory_set = set(memory_tags)
    if not event_tags:
        return 0.0, []
    shared = sorted(event_tags & memory_set)
    overlap = len(shared) / len(event_tags)
    return overlap, shared


def _question_age_days(
    memory: MemoryRecord,
    now: datetime | None = None,
) -> float:
    current = now or datetime.now(timezone.utc)
    delta = current - memory.created_at
    return max(delta.total_seconds() / 86400.0, 0.0)


def _question_has_answer(memory: MemoryRecord) -> bool:
    metadata = memory.metadata if isinstance(memory.metadata, dict) else {}
    if metadata.get("answered") is True:
        return True
    if metadata.get("answer"):
        return True
    if metadata.get("response"):
        return True
    return False


def explain_hybrid_score(
    event: Event,
    memory: MemoryRecord,
    *,
    event_vector: Sequence[float] | None = None,
    memory_vector: Sequence[float] | None = None,
    query_intent: QueryIntent | None = None,
    event_domain: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a per-component breakdown of the Memory v2 hybrid score.

    The dictionary keys are stable so callers (tests, debug output, prompt
    grounding) can rely on them without re-implementing the scoring math.
    """
    intent = query_intent if query_intent is not None else classify_query_intent(event.content)
    event_tags = extract_event_tags(event)
    semantic = _semantic_similarity_value(event_vector, memory_vector)
    tag_overlap, shared_tags = _tag_overlap(event_tags, memory.tags)
    salience = float(memory.salience)
    recency = recency_score(memory.created_at, now=now)
    derived_event_domain = (
        event_domain if event_domain is not None else _domain_for_event(event)
    )
    memory_domain = _domain_for_memory(memory)
    domain_match = (
        1.0
        if derived_event_domain
        and memory_domain
        and derived_event_domain == memory_domain
        else 0.0
    )
    role = _memory_role(memory)

    role_bonus_raw, role_bonus_reason = _role_bonus(
        intent=intent,
        role=role,
        domain_match=bool(domain_match),
        memory=memory,
    )
    role_penalty_raw, role_penalty_reason = _role_penalty(
        intent=intent,
        role=role,
        memory=memory,
        event=event,
        now=now,
    )

    semantic_component = semantic * HYBRID_SEMANTIC_WEIGHT
    tag_component = tag_overlap * HYBRID_TAG_WEIGHT
    salience_component = salience * HYBRID_SALIENCE_WEIGHT
    recency_component = recency * HYBRID_RECENCY_WEIGHT
    domain_component = domain_match * HYBRID_DOMAIN_WEIGHT
    # The bonus weight caps how much the role bonus can move the final
    # score; the raw magnitude is reported separately for inspection.
    role_bonus_component = min(role_bonus_raw, 1.0) * HYBRID_ROLE_BONUS_WEIGHT

    total = (
        semantic_component
        + tag_component
        + salience_component
        + recency_component
        + domain_component
        + role_bonus_component
        - role_penalty_raw
    )

    return {
        "semantic_similarity": semantic,
        "tag_overlap": tag_overlap,
        "salience": salience,
        "recency": recency,
        "domain_match": domain_match,
        "role_bonus_raw": role_bonus_raw,
        "role_penalty_raw": role_penalty_raw,
        "semantic_component": semantic_component,
        "tag_component": tag_component,
        "salience_component": salience_component,
        "recency_component": recency_component,
        "domain_component": domain_component,
        "role_bonus_component": role_bonus_component,
        "role_penalty_component": -role_penalty_raw,
        "shared_tags": shared_tags,
        "event_domain": derived_event_domain,
        "memory_domain": memory_domain,
        "memory_role": role.value,
        "query_intent": intent.value,
        "role_bonus_reason": role_bonus_reason,
        "role_penalty_reason": role_penalty_reason,
        "total": total,
    }


def hybrid_score(
    event: Event,
    memory: MemoryRecord,
    *,
    event_vector: Sequence[float] | None = None,
    memory_vector: Sequence[float] | None = None,
    query_intent: QueryIntent | None = None,
    event_domain: str | None = None,
    now: datetime | None = None,
) -> float:
    breakdown = explain_hybrid_score(
        event,
        memory,
        event_vector=event_vector,
        memory_vector=memory_vector,
        query_intent=query_intent,
        event_domain=event_domain,
        now=now,
    )
    return float(breakdown["total"])


def hybrid_sort_key(
    event: Event,
    memory: MemoryRecord,
    *,
    event_vector: Sequence[float] | None = None,
    memory_vector: Sequence[float] | None = None,
    query_intent: QueryIntent | None = None,
    event_domain: str | None = None,
    now: datetime | None = None,
) -> tuple[float, float, str]:
    """Return a stable sort key for hybrid retrieval ordering."""
    return (
        hybrid_score(
            event,
            memory,
            event_vector=event_vector,
            memory_vector=memory_vector,
            query_intent=query_intent,
            event_domain=event_domain,
            now=now,
        ),
        memory.created_at.timestamp(),
        memory.id,
    )


def _role_bonus(
    *,
    intent: QueryIntent,
    role: MemoryRole,
    domain_match: bool,
    memory: MemoryRecord,
) -> tuple[float, str | None]:
    del memory  # currently unused; reserved for future domain-specific tweaks
    if intent == QueryIntent.RECOMMENDATION and role == MemoryRole.PREFERENCE and domain_match:
        return ROLE_BONUS_RECOMMENDATION_PREFERENCE, "recommendation_preference_same_domain"
    if intent == QueryIntent.RECOMMENDATION and role == MemoryRole.PREFERENCE:
        # Even without a strict domain match, a preference still helps a
        # recommendation more than a random fact would.
        return ROLE_BONUS_RECOMMENDATION_PREFERENCE / 2, "recommendation_preference_other_domain"
    if intent == QueryIntent.PLANNING and role == MemoryRole.TASK and domain_match:
        return ROLE_BONUS_PLANNING_TASK, "planning_task_same_domain"
    if intent == QueryIntent.FACTUAL and role == MemoryRole.FACT:
        return ROLE_BONUS_FACTUAL_FACT, "factual_fact"
    if role == MemoryRole.OUTCOME:
        # Outcomes are mildly useful to most queries because they record
        # what worked or did not.
        return ROLE_BONUS_GENERIC_OUTCOME, "generic_outcome"
    return 0.0, None


def _role_penalty(
    *,
    intent: QueryIntent,
    role: MemoryRole,
    memory: MemoryRecord,
    event: Event,
    now: datetime | None,
) -> tuple[float, str | None]:
    if role != MemoryRole.QUESTION:
        return 0.0, None
    has_answer = _question_has_answer(memory)
    age_days = _question_age_days(memory, now=now)

    if (
        intent == QueryIntent.RECOMMENDATION
        and not has_answer
        and _is_duplicate_question(event, memory)
    ):
        return ROLE_PENALTY_DUPLICATE_QUESTION, "duplicate_recent_question_for_recommendation"
    if intent == QueryIntent.RECOMMENDATION and not has_answer:
        return ROLE_PENALTY_RECOMMENDATION_QUESTION, "prior_question_for_recommendation"
    if not has_answer and age_days >= QUESTION_RECENCY_DUPLICATE_DAYS:
        return ROLE_PENALTY_OLD_UNANSWERED_QUESTION, "old_unanswered_question"
    if not has_answer and _is_duplicate_question(event, memory):
        return ROLE_PENALTY_DUPLICATE_QUESTION, "duplicate_recent_question"
    return 0.0, None


def _is_duplicate_question(event: Event, memory: MemoryRecord) -> bool:
    """Return True when the memory looks like the same question as ``event``.

    Used as a tie-breaker so a query like "What kind of book should I read
    next?" does not boost an identical prior memory to the top of retrieval.
    The matcher is conservative on purpose: very high token overlap between
    the event and the memory content.
    """
    event_tokens = tokenize(event.content)
    memory_tokens = tokenize(memory.content)
    if not event_tokens or not memory_tokens:
        return False
    shared = event_tokens & memory_tokens
    union = event_tokens | memory_tokens
    if not union:
        return False
    jaccard = len(shared) / len(union)
    return jaccard >= 0.6
