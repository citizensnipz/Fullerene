"""Inspectable models for Fullerene Learning v0."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
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


class SignalType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FAILURE = "failure"
    NEUTRAL = "neutral"


class SignalSource(str, Enum):
    USER_FEEDBACK = "user_feedback"
    EXECUTION_RESULT = "execution_result"
    GOAL_LIFECYCLE = "goal_lifecycle"
    SYSTEM = "system"


class AdjustmentStatus(str, Enum):
    APPLIED = "applied"
    PROPOSED = "proposed"
    SKIPPED = "skipped"


class AdjustmentTarget(str, Enum):
    MEMORY_SALIENCE = "memory_salience"
    GOAL_PRIORITY = "goal_priority"
    BEHAVIOR_CONFIDENCE = "behavior_confidence"
    BEHAVIOR_THRESHOLD = "behavior_threshold"


@dataclass(slots=True)
class LearningSignal:
    id: str = dataclass_field(default_factory=lambda: uuid4().hex)
    created_at: datetime = dataclass_field(default_factory=utcnow)
    signal_type: SignalType = SignalType.NEUTRAL
    source: SignalSource = SignalSource.SYSTEM
    magnitude: float = 0.0
    source_event_id: str | None = None
    source_record_id: str | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    reasons: list[str] = dataclass_field(default_factory=list)

    def __post_init__(self) -> None:
        self.magnitude = _clamp_unit(self.magnitude)
        self.metadata = dict(self.metadata or {})
        self.reasons = [str(reason).strip() for reason in self.reasons if str(reason).strip()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "signal_type": self.signal_type.value,
            "source": self.source.value,
            "magnitude": self.magnitude,
            "source_event_id": self.source_event_id,
            "source_record_id": self.source_record_id,
            "metadata": _serialize_value(self.metadata),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningSignal":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            signal_type=SignalType(data["signal_type"]),
            source=SignalSource(data["source"]),
            magnitude=data.get("magnitude", 0.0),
            source_event_id=data.get("source_event_id"),
            source_record_id=data.get("source_record_id"),
            metadata=data.get("metadata", {}),
            reasons=data.get("reasons", []),
        )


@dataclass(slots=True)
class AdjustmentRecord:
    id: str = dataclass_field(default_factory=lambda: uuid4().hex)
    created_at: datetime = dataclass_field(default_factory=utcnow)
    target: AdjustmentTarget = AdjustmentTarget.BEHAVIOR_CONFIDENCE
    target_id: str | None = None
    target_facet: str = "behavior"
    field: str = "confidence"
    old_value: float | None = None
    new_value: float | None = None
    delta: float = 0.0
    status: AdjustmentStatus = AdjustmentStatus.SKIPPED
    source_signal_id: str = ""
    reasons: list[str] = dataclass_field(default_factory=list)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.old_value = _coerce_optional_unit(self.old_value)
        self.new_value = _coerce_optional_unit(self.new_value)
        self.delta = round(float(self.delta), 3)
        self.target_facet = str(self.target_facet or "").strip() or "learning"
        self.field = str(self.field or "").strip() or "value"
        self.metadata = dict(self.metadata or {})
        self.reasons = [str(reason).strip() for reason in self.reasons if str(reason).strip()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "target": self.target.value,
            "target_id": self.target_id,
            "target_facet": self.target_facet,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "status": self.status.value,
            "source_signal_id": self.source_signal_id,
            "reasons": list(self.reasons),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdjustmentRecord":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            target=AdjustmentTarget(data["target"]),
            target_id=data.get("target_id"),
            target_facet=data.get("target_facet", "behavior"),
            field=data.get("field", "confidence"),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            delta=data.get("delta", 0.0),
            status=AdjustmentStatus(data.get("status", AdjustmentStatus.SKIPPED.value)),
            source_signal_id=data.get("source_signal_id", ""),
            reasons=data.get("reasons", []),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class LearningResult:
    signals: list[LearningSignal] = dataclass_field(default_factory=list)
    adjustments: list[AdjustmentRecord] = dataclass_field(default_factory=list)
    proposals: list[AdjustmentRecord] = dataclass_field(default_factory=list)
    applied: list[AdjustmentRecord] = dataclass_field(default_factory=list)
    overall_status: str = "no_signal"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signals = list(self.signals or [])
        self.adjustments = list(self.adjustments or [])
        self.proposals = list(
            self.proposals
            or [record for record in self.adjustments if record.status == AdjustmentStatus.PROPOSED]
        )
        self.applied = list(
            self.applied
            or [record for record in self.adjustments if record.status == AdjustmentStatus.APPLIED]
        )
        self.overall_status = str(self.overall_status or "no_signal").strip() or "no_signal"
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [signal.to_dict() for signal in self.signals],
            "adjustments": [record.to_dict() for record in self.adjustments],
            "proposals": [record.to_dict() for record in self.proposals],
            "applied": [record.to_dict() for record in self.applied],
            "overall_status": self.overall_status,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningResult":
        return cls(
            signals=[
                LearningSignal.from_dict(signal)
                for signal in data.get("signals", [])
            ],
            adjustments=[
                AdjustmentRecord.from_dict(record)
                for record in data.get("adjustments", [])
            ],
            proposals=[
                AdjustmentRecord.from_dict(record)
                for record in data.get("proposals", [])
            ],
            applied=[
                AdjustmentRecord.from_dict(record)
                for record in data.get("applied", [])
            ],
            overall_status=data.get("overall_status", "no_signal"),
            metadata=data.get("metadata", {}),
        )


def _clamp_unit(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


def _coerce_optional_unit(value: float | None) -> float | None:
    if value is None:
        return None
    return _clamp_unit(value)
