"""Goal models for deterministic Fullerene goals storage."""

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


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class GoalSource(str, Enum):
    USER = "user"
    SYSTEM = "system"


@dataclass(slots=True)
class Goal:
    id: str = field(default_factory=lambda: uuid4().hex)
    description: str = ""
    priority: float = 0.5
    status: GoalStatus = GoalStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    source: GoalSource = GoalSource.USER
    reinforcement_score: float = 0.0
    activation_count: int = 0
    last_activated_at: datetime | None = None
    last_activated_event_id: str | None = None
    last_reinforced_at: datetime | None = None
    completion_score: float = 0.0
    paused_reason: str | None = None
    completed_reason: str | None = None
    completed_at: datetime | None = None
    evidence_event_ids: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    stale_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.priority = self._validate_priority(self.priority)
        self.tags = normalize_tags(self.tags)
        self.reinforcement_score = self._clamp01(self.reinforcement_score)
        self.activation_count = max(int(self.activation_count), 0)
        self.completion_score = self._clamp01(self.completion_score)
        self.stale_score = self._clamp01(self.stale_score)
        self.evidence_event_ids = [
            str(event_id).strip()
            for event_id in self.evidence_event_ids
            if str(event_id).strip()
        ][:20]

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _validate_priority(value: float) -> float:
        priority = float(value)
        if not 0.0 <= priority <= 1.0:
            raise ValueError("priority must be between 0.0 and 1.0")
        return priority

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source.value,
            "reinforcement_score": self.reinforcement_score,
            "activation_count": self.activation_count,
            "last_activated_at": (
                self.last_activated_at.isoformat() if self.last_activated_at else None
            ),
            "last_activated_event_id": self.last_activated_event_id,
            "last_reinforced_at": (
                self.last_reinforced_at.isoformat() if self.last_reinforced_at else None
            ),
            "completion_score": self.completion_score,
            "paused_reason": self.paused_reason,
            "completed_reason": self.completed_reason,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "evidence_event_ids": list(self.evidence_event_ids),
            "blocked_reason": self.blocked_reason,
            "stale_score": self.stale_score,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            priority=data.get("priority", 0.5),
            status=GoalStatus(data.get("status", GoalStatus.ACTIVE.value)),
            tags=data.get("tags", []),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            source=GoalSource(data.get("source", GoalSource.USER.value)),
            reinforcement_score=data.get("reinforcement_score", 0.0),
            activation_count=data.get("activation_count", 0),
            last_activated_at=(
                _parse_datetime(data["last_activated_at"])
                if data.get("last_activated_at")
                else None
            ),
            last_activated_event_id=data.get("last_activated_event_id"),
            last_reinforced_at=(
                _parse_datetime(data["last_reinforced_at"])
                if data.get("last_reinforced_at")
                else None
            ),
            completion_score=data.get("completion_score", 0.0),
            paused_reason=data.get("paused_reason"),
            completed_reason=data.get("completed_reason"),
            completed_at=(
                _parse_datetime(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            evidence_event_ids=data.get("evidence_event_ids", []),
            blocked_reason=data.get("blocked_reason"),
            stale_score=data.get("stale_score", 0.0),
            metadata=data.get("metadata", {}),
        )
