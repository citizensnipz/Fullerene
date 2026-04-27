"""Inspectable plan models for Fullerene Planner v0."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


class PlanStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class PlanStepStatus(str, Enum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    COMPLETED = "completed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class PlanStep:
    id: str = field(default_factory=lambda: uuid4().hex)
    description: str = ""
    order: int = 1
    target_type: str = "general"
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    status: PlanStepStatus = PlanStepStatus.PROPOSED
    policy_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.description = str(self.description).strip()
        self.order = max(int(self.order), 1)
        self.target_type = str(self.target_type or "general").strip() or "general"
        self.requires_approval = bool(self.requires_approval)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "order": self.order,
            "target_type": self.target_type,
            "risk_level": self.risk_level.value,
            "requires_approval": self.requires_approval,
            "status": self.status.value,
            "policy_status": self.policy_status,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            order=data.get("order", 1),
            target_type=data.get("target_type", "general"),
            risk_level=RiskLevel(data.get("risk_level", RiskLevel.LOW.value)),
            requires_approval=data.get("requires_approval", False),
            status=PlanStepStatus(data.get("status", PlanStepStatus.PROPOSED.value)),
            policy_status=data.get("policy_status"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class Plan:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    source_event_id: str | None = None
    goal_id: str | None = None
    title: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    confidence: float = 0.0
    pressure: float = 0.0
    status: PlanStatus = PlanStatus.PROPOSED
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.title = str(self.title).strip()
        self.confidence = round(_clamp_unit(self.confidence), 3)
        self.pressure = round(_clamp_unit(self.pressure), 3)
        self.reasons = [str(reason).strip() for reason in self.reasons if str(reason).strip()]
        self.metadata = dict(self.metadata or {})
        self.steps = sorted(list(self.steps), key=lambda step: (step.order, step.id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "source_event_id": self.source_event_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "steps": [step.to_dict() for step in self.steps],
            "confidence": self.confidence,
            "pressure": self.pressure,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            source_event_id=data.get("source_event_id"),
            goal_id=data.get("goal_id"),
            title=data.get("title", ""),
            steps=[PlanStep.from_dict(step) for step in data.get("steps", [])],
            confidence=data.get("confidence", 0.0),
            pressure=data.get("pressure", 0.0),
            status=PlanStatus(data.get("status", PlanStatus.PROPOSED.value)),
            reasons=data.get("reasons", []),
            metadata=data.get("metadata", {}),
        )
