"""Inspectable models for Fullerene Affect v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0 = "deterministic_vad_novelty_v0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _parse_datetime(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _clamp_signed(value: float) -> float:
    return round(max(-1.0, min(float(value), 1.0)), 3)


def _clamp_unit(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


@dataclass(slots=True)
class AffectState:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    novelty: float = 0.5
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.valence = _clamp_signed(self.valence)
        self.arousal = _clamp_unit(self.arousal)
        self.dominance = _clamp_unit(self.dominance)
        self.novelty = _clamp_unit(self.novelty)
        self.components = dict(self.components or {})
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "novelty": self.novelty,
            "components": _serialize_value(self.components),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AffectState":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.5),
            novelty=data.get("novelty", 0.5),
            components=data.get("components", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class AffectResult:
    current_state: AffectState
    history: list[AffectState] = field(default_factory=list)
    strategy: str = AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.history = list(self.history or [])
        self.strategy = (
            str(self.strategy or AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0).strip()
            or AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0
        )
        self.reasons = [
            str(reason).strip() for reason in self.reasons if str(reason).strip()
        ]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state.to_dict(),
            "history": [state.to_dict() for state in self.history],
            "strategy": self.strategy,
            "reasons": list(self.reasons),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AffectResult":
        return cls(
            current_state=AffectState.from_dict(data["current_state"]),
            history=[
                AffectState.from_dict(state)
                for state in data.get("history", [])
            ],
            strategy=data.get(
                "strategy",
                AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0,
            ),
            reasons=data.get("reasons", []),
            metadata=data.get("metadata", {}),
        )
