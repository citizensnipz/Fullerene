"""Deterministic scoring helpers for Fullerene memory retrieval."""

from __future__ import annotations

from collections.abc import Iterable
import re
from datetime import datetime, timezone

from fullerene.memory.models import MemoryRecord, normalize_tags
from fullerene.nexus.models import Event

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
    event_tokens = tokenize(event.content)
    memory_tokens = tokenize(memory.content)
    shared_tokens = event_tokens & memory_tokens
    keyword_score = len(shared_tokens) / len(event_tokens) if event_tokens else 0.0

    event_tags = extract_event_tags(event)
    memory_tags = set(memory.tags)
    shared_tags = event_tags & memory_tags
    tag_score = len(shared_tags) / len(event_tags) if event_tags else 0.0

    return (
        (keyword_score * 0.5)
        + (tag_score * 0.2)
        + (memory.salience * 0.2)
        + (recency_score(memory.created_at, now=now) * 0.1)
    )


def score_sort_key(event: Event, memory: MemoryRecord) -> tuple[float, float, str]:
    return (
        score_memory_record(event, memory),
        memory.created_at.timestamp(),
        memory.id,
    )
