"""Models for Fullerene Context v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


STATIC_RECENT_EPISODIC_V0 = "static_recent_episodic_v0"


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


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


class ContextItemType(str, Enum):
    EVENT = "event"
    MEMORY = "memory"
    GOAL = "goal"
    BELIEF = "belief"
    POLICY = "policy"
    SYSTEM = "system"


@dataclass(slots=True)
class ContextItem:
    id: str = field(default_factory=lambda: uuid4().hex)
    item_type: ContextItemType = ContextItemType.MEMORY
    content: str = ""
    source_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_type": self.item_type.value,
            "content": self.content,
            "source_id": self.source_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextItem":
        return cls(
            id=data["id"],
            item_type=ContextItemType(data["item_type"]),
            content=data.get("content", ""),
            source_id=data.get("source_id"),
            created_at=_parse_datetime(data.get("created_at")),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ContextWindow:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    items: list[ContextItem] = field(default_factory=list)
    max_items: int = 5
    strategy: str = STATIC_RECENT_EPISODIC_V0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_items = max(int(self.max_items), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
            "max_items": self.max_items,
            "strategy": self.strategy,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextWindow":
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            items=[ContextItem.from_dict(item) for item in data.get("items", [])],
            max_items=data.get("max_items", 5),
            strategy=data.get("strategy", STATIC_RECENT_EPISODIC_V0),
            metadata=data.get("metadata", {}),
        )
