"""Memory models for deterministic Fullerene memory storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4


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


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags or ():
        cleaned = str(tag).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class MemoryRecord:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    source_event_id: str | None = None
    salience: float = 0.5
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.salience = self._validate_score("salience", self.salience)
        self.confidence = self._validate_score("confidence", self.confidence)
        self.tags = normalize_tags(self.tags)

    @staticmethod
    def _validate_score(field_name: str, value: float) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "memory_type": self.memory_type.value,
            "content": self.content,
            "source_event_id": self.source_event_id,
            "salience": self.salience,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            memory_type=MemoryType(data["memory_type"]),
            content=data.get("content", ""),
            source_event_id=data.get("source_event_id"),
            salience=data.get("salience", 0.5),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
