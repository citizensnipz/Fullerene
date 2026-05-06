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


class MemoryLayer(str, Enum):
    WORKING = "working"
    LONG_TERM = "long_term"


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
    # Memory v2: deterministic role + domain fields. ``role`` defaults to
    # the string "unknown" so legacy callers and snapshot rows that pre-date
    # Memory v2 still round-trip cleanly. ``domain`` is optional.
    role: str = "unknown"
    domain: str | None = None
    memory_layer: MemoryLayer = MemoryLayer.LONG_TERM
    # Memory v3: optional thematic community / cluster assignment (SQLite column).
    community_id: str | None = None

    def __post_init__(self) -> None:
        self.salience = self._validate_score("salience", self.salience)
        self.confidence = self._validate_score("confidence", self.confidence)
        self.tags = normalize_tags(self.tags)
        self.role = self._normalize_role(self.role)
        self.domain = self._normalize_domain(self.domain)
        if not isinstance(self.memory_layer, MemoryLayer):
            self.memory_layer = MemoryLayer(str(self.memory_layer).strip().lower())
        if self.community_id is not None:
            cleaned_c = str(self.community_id).strip()
            self.community_id = cleaned_c or None

    @staticmethod
    def _validate_score(field_name: str, value: float) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return score

    @staticmethod
    def _normalize_role(raw_role: object) -> str:
        if raw_role is None:
            return "unknown"
        cleaned = str(raw_role).strip().lower()
        return cleaned or "unknown"

    @staticmethod
    def _normalize_domain(raw_domain: object) -> str | None:
        if raw_domain is None:
            return None
        cleaned = str(raw_domain).strip().lower()
        return cleaned or None

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
            "role": self.role,
            "domain": self.domain,
            "memory_layer": self.memory_layer.value,
            "community_id": self.community_id,
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
            role=data.get("role", "unknown") or "unknown",
            domain=data.get("domain"),
            memory_layer=data.get("memory_layer", MemoryLayer.LONG_TERM.value),
            community_id=data.get("community_id"),
        )
