"""Deterministic scoring helpers for Fullerene memory retrieval.

Memory v1 keeps retrieval scoring fully transparent:

- ``keyword_overlap`` - fraction of event tokens that also appear in the memory.
- ``tag_overlap``     - fraction of event tags that also appear on the memory.
- ``salience``        - the memory's stored deterministic salience score.
- ``recency``         - smooth time-decay over days since the memory was created.

The components are linearly combined with fixed weights. A separate
``explain_score`` helper returns the per-component breakdown so retrieval
results stay easy to debug without re-implementing the math elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
from datetime import datetime, timezone
from typing import Any

from fullerene.memory.models import MemoryRecord, normalize_tags
from fullerene.nexus.models import Event

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

KEYWORD_WEIGHT = 0.5
TAG_WEIGHT = 0.2
SALIENCE_WEIGHT = 0.2
RECENCY_WEIGHT = 0.1


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if len(token) >= 2
    }


def extract_event_tags(event: Event) -> set[str]:
    raw_tags = event.metadata.get("tags", [])
    if not isinstance(raw_tags, Iterable) or isinstance(raw_tags, (str, bytes)):
        return set()
    return set(normalize_tags(raw_tags))


def recency_score(created_at: datetime, now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    age_seconds = max((current - created_at).total_seconds(), 0.0)
    age_days = age_seconds / 86400.0
    return 1.0 / (1.0 + age_days)


def score_memory_record(
    event: Event,
    memory: MemoryRecord,
    now: datetime | None = None,
) -> float:
    breakdown = explain_score(event, memory, now=now)
    return float(breakdown["total"])


def explain_score(
    event: Event,
    memory: MemoryRecord,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a per-component breakdown of the retrieval score.

    The dict keys are stable so tests and inspectors can rely on them. Token
    and tag overlap sets are sorted lists for deterministic ordering.
    """
    event_tokens = tokenize(event.content)
    memory_tokens = tokenize(memory.content)
    shared_tokens = event_tokens & memory_tokens
    keyword_overlap = (
        len(shared_tokens) / len(event_tokens) if event_tokens else 0.0
    )

    event_tags = extract_event_tags(event)
    memory_tags = set(memory.tags)
    shared_tags = event_tags & memory_tags
    tag_overlap = len(shared_tags) / len(event_tags) if event_tags else 0.0

    recency = recency_score(memory.created_at, now=now)

    keyword_component = keyword_overlap * KEYWORD_WEIGHT
    tag_component = tag_overlap * TAG_WEIGHT
    salience_component = float(memory.salience) * SALIENCE_WEIGHT
    recency_component = recency * RECENCY_WEIGHT

    total = (
        keyword_component
        + tag_component
        + salience_component
        + recency_component
    )

    return {
        "keyword_overlap": keyword_overlap,
        "tag_overlap": tag_overlap,
        "salience": float(memory.salience),
        "recency": recency,
        "keyword_component": keyword_component,
        "tag_component": tag_component,
        "salience_component": salience_component,
        "recency_component": recency_component,
        "shared_tokens": sorted(shared_tokens),
        "shared_tags": sorted(shared_tags),
        "total": total,
    }


def score_sort_key(event: Event, memory: MemoryRecord) -> tuple[float, float, str]:
    return (
        score_memory_record(event, memory),
        memory.created_at.timestamp(),
        memory.id,
    )
