"""Belief models for deterministic Fullerene world model storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from fullerene.memory.models import normalize_tags


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _parse_datetime(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


class BeliefStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


class BeliefSource(str, Enum):
    USER = "user"
    SYSTEM = "system"
    MEMORY = "memory"
    GOAL = "goal"


@dataclass(slots=True)
class Belief:
    id: str = field(default_factory=lambda: uuid4().hex)
    claim: str = ""
    confidence: float = 0.5
    status: BeliefStatus = BeliefStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    source: BeliefSource = BeliefSource.USER
    source_event_id: str | None = None
    source_memory_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = self._validate_confidence(self.confidence)
        self.tags = normalize_tags(self.tags)

    @staticmethod
    def _validate_confidence(value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "confidence": self.confidence,
            "status": self.status.value,
            "tags": list(self.tags),
            "source": self.source.value,
            "source_event_id": self.source_event_id,
            "source_memory_id": self.source_memory_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Belief":
        return cls(
            id=data["id"],
            claim=data.get("claim", ""),
            confidence=data.get("confidence", 0.5),
            status=BeliefStatus(data.get("status", BeliefStatus.ACTIVE.value)),
            tags=data.get("tags", []),
            source=BeliefSource(data.get("source", BeliefSource.USER.value)),
            source_event_id=data.get("source_event_id"),
            source_memory_id=data.get("source_memory_id"),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
