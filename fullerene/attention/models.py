"""Inspectable models for Fullerene Attention v0."""

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
