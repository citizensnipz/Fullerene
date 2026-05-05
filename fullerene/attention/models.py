"""Inspectable models for Fullerene Attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


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


class AttentionSource(str, Enum):
    MEMORY = "memory"
    GOAL = "goal"
    BELIEF = "belief"
    EXECUTION = "execution"
    EVENT = "event"
    SYSTEM = "system"


class AttentionMode(str, Enum):
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"


ATTENTION_STRATEGY_FIXED_WEIGHT_V0 = "fixed_weight_competition_v0"


@dataclass(slots=True)
class AttentionItem:
    id: str
    source: AttentionSource
    content: str
    source_id: str | None = None
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    dominant_component: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.content = str(self.content or "")
        self.score = self._validate_unit("score", self.score)
        self.components = {
            str(name): self._validate_unit(str(name), value)
            for name, value in dict(self.components or {}).items()
        }
        self.dominant_component = (
            str(self.dominant_component).strip()
            if self.dominant_component is not None and str(self.dominant_component).strip()
            else None
        )
        self.metadata = dict(self.metadata or {})

    @staticmethod
    def _validate_unit(field_name: str, value: float) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "source_id": self.source_id,
            "content": self.content,
            "score": self.score,
            "components": dict(self.components),
            "dominant_component": self.dominant_component,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionItem":
        return cls(
            id=data["id"],
            source=AttentionSource(data["source"]),
            source_id=data.get("source_id"),
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            components=data.get("components", {}),
            dominant_component=data.get("dominant_component"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class AttentionBroadcast:
    id: str
    created_at: datetime
    item_id: str
    source: AttentionSource
    content: str
    score: float
    mode: AttentionMode
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    recipients: list[str] = field(default_factory=list)
    conflict_ids: list[str] = field(default_factory=list)
    repeated_count: int = 0
    pressure_contribution: float = 0.0
    source_id: str | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.item_id = str(self.item_id)
        self.content = str(self.content or "")
        self.score = AttentionItem._validate_unit("score", self.score)
        self.components = {
            str(name): AttentionItem._validate_unit(str(name), value)
            for name, value in dict(self.components or {}).items()
        }
        self.metadata = dict(self.metadata or {})
        self.recipients = [str(recipient) for recipient in list(self.recipients or [])]
        self.conflict_ids = [str(conflict_id) for conflict_id in list(self.conflict_ids or [])]
        self.repeated_count = max(int(self.repeated_count), 0)
        self.pressure_contribution = AttentionItem._validate_unit(
            "pressure_contribution",
            self.pressure_contribution,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "item_id": self.item_id,
            "source": self.source.value,
            "source_id": self.source_id,
            "content": self.content,
            "score": self.score,
            "mode": self.mode.value,
            "components": dict(self.components),
            "metadata": _serialize_value(self.metadata),
            "recipients": list(self.recipients),
            "conflict_ids": list(self.conflict_ids),
            "repeated_count": self.repeated_count,
            "pressure_contribution": self.pressure_contribution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionBroadcast":
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            item_id=data["item_id"],
            source=AttentionSource(data["source"]),
            source_id=data.get("source_id"),
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            mode=AttentionMode(data["mode"]),
            components=data.get("components", {}),
            metadata=data.get("metadata", {}),
            recipients=data.get("recipients", []),
            conflict_ids=data.get("conflict_ids", []),
            repeated_count=data.get("repeated_count", 0),
            pressure_contribution=data.get("pressure_contribution", 0.0),
        )


@dataclass(slots=True)
class AttentionConflict:
    id: str
    item_ids: list[str]
    score_delta: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.item_ids = [str(item_id) for item_id in list(self.item_ids or [])]
        self.score_delta = AttentionItem._validate_unit("score_delta", self.score_delta)
        self.reason = str(self.reason or "").strip()
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_ids": list(self.item_ids),
            "score_delta": self.score_delta,
            "reason": self.reason,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionConflict":
        return cls(
            id=data["id"],
            item_ids=data.get("item_ids", []),
            score_delta=data.get("score_delta", 0.0),
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class AttentionHistoryEntry:
    broadcast_id: str
    item_id: str
    source: AttentionSource
    score: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str | None = None

    def __post_init__(self) -> None:
        self.broadcast_id = str(self.broadcast_id)
        self.item_id = str(self.item_id)
        self.score = AttentionItem._validate_unit("score", self.score)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "broadcast_id": self.broadcast_id,
            "item_id": self.item_id,
            "source": self.source.value,
            "source_id": self.source_id,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionHistoryEntry":
        return cls(
            broadcast_id=data["broadcast_id"],
            item_id=data["item_id"],
            source=AttentionSource(data["source"]),
            source_id=data.get("source_id"),
            score=data.get("score", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class AttentionResult:
    focus_items: list[AttentionItem] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    dominant_source: AttentionSource | None = None
    strategy: str = ATTENTION_STRATEGY_FIXED_WEIGHT_V0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.focus_items = list(self.focus_items or [])
        self.scores = {
            str(item_id): AttentionItem._validate_unit("score", value)
            for item_id, value in dict(self.scores or {}).items()
        }
        if self.dominant_source is None and self.focus_items:
            self.dominant_source = self.focus_items[0].source
        self.strategy = (
            str(self.strategy or ATTENTION_STRATEGY_FIXED_WEIGHT_V0).strip()
            or ATTENTION_STRATEGY_FIXED_WEIGHT_V0
        )
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_items": [item.to_dict() for item in self.focus_items],
            "scores": dict(self.scores),
            "dominant_source": (
                self.dominant_source.value if self.dominant_source is not None else None
            ),
            "strategy": self.strategy,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionResult":
        raw_dominant_source = data.get("dominant_source")
        return cls(
            focus_items=[
                AttentionItem.from_dict(item)
                for item in data.get("focus_items", [])
            ],
            scores=data.get("scores", {}),
            dominant_source=(
                AttentionSource(raw_dominant_source)
                if raw_dominant_source is not None
                else None
            ),
            strategy=data.get("strategy", ATTENTION_STRATEGY_FIXED_WEIGHT_V0),
            metadata=data.get("metadata", {}),
        )
