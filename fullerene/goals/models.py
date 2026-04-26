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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.priority = self._validate_priority(self.priority)
        self.tags = normalize_tags(self.tags)

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
            metadata=data.get("metadata", {}),
        )
