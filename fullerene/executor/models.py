"""Inspectable execution models for Fullerene Executor v0."""

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


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    LIVE = "live"


class ActionType(str, Enum):
    UPDATE_MEMORY = "update_memory"
    UPDATE_GOAL = "update_goal"
    UPDATE_BELIEF = "update_belief"
    EMIT_EVENT = "emit_event"
    NOOP = "noop"


def coerce_action_type(raw_value: Any) -> ActionType | None:
    if isinstance(raw_value, ActionType):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    if not cleaned:
        return None
    try:
        return ActionType(cleaned)
    except ValueError:
        return None


@dataclass(slots=True)
class ExecutionRecord:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    action_type: ActionType = ActionType.NOOP
    plan_id: str | None = None
    plan_step_id: str | None = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    dry_run: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.message = str(self.message or "").strip()
        self.dry_run = bool(self.dry_run)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "action_type": self.action_type.value,
            "plan_id": self.plan_id,
            "plan_step_id": self.plan_step_id,
            "status": self.status.value,
            "dry_run": self.dry_run,
            "message": self.message,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            action_type=ActionType(data.get("action_type", ActionType.NOOP.value)),
            plan_id=data.get("plan_id"),
            plan_step_id=data.get("plan_step_id"),
            status=ExecutionStatus(data.get("status", ExecutionStatus.SUCCESS.value)),
            dry_run=data.get("dry_run", True),
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ExecutionResult:
    plan_id: str | None = None
    records: list[ExecutionRecord] = field(default_factory=list)
    overall_status: ExecutionStatus = ExecutionStatus.SUCCESS
    halted: bool = False
    dry_run: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.records = list(self.records or [])
        self.halted = bool(self.halted)
        self.dry_run = bool(self.dry_run)
        self.reasons = [
            str(reason).strip() for reason in self.reasons if str(reason).strip()
        ]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "records": [record.to_dict() for record in self.records],
            "overall_status": self.overall_status.value,
            "halted": self.halted,
            "dry_run": self.dry_run,
            "reasons": list(self.reasons),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        return cls(
            plan_id=data.get("plan_id"),
            records=[
                ExecutionRecord.from_dict(record)
                for record in data.get("records", [])
            ],
            overall_status=ExecutionStatus(
                data.get("overall_status", ExecutionStatus.SUCCESS.value)
            ),
            halted=data.get("halted", False),
            dry_run=data.get("dry_run", True),
            reasons=data.get("reasons", []),
            metadata=data.get("metadata", {}),
        )
