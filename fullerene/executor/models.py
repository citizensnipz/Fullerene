"""Inspectable execution models for Fullerene Executor v1."""

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
    PENDING_APPROVAL = "pending_approval"
    APPROVAL_TIMEOUT = "approval_timeout"


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    LIVE = "live"


class ActionType(str, Enum):
    UPDATE_MEMORY = "update_memory"
    UPDATE_GOAL = "update_goal"
    UPDATE_BELIEF = "update_belief"
    EMIT_EVENT = "emit_event"
    NOOP = "noop"


@dataclass(slots=True)
class SkillManifestEntry:
    skill_name: str
    version: str
    action_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    requires_approval: bool = False
    dry_run_supported: bool = True
    live_supported: bool = False
    sandbox_required: bool = False
    validator_name: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.skill_name = str(self.skill_name or "").strip().lower()
        self.version = str(self.version or "").strip() or "v1"
        self.action_types = [
            str(action).strip().lower()
            for action in self.action_types
            if str(action).strip()
        ]
        self.target_types = [
            str(target).strip().lower()
            for target in self.target_types
            if str(target).strip()
        ]
        self.requires_approval = bool(self.requires_approval)
        self.dry_run_supported = bool(self.dry_run_supported)
        self.live_supported = bool(self.live_supported)
        self.sandbox_required = bool(self.sandbox_required)
        self.validator_name = (
            str(self.validator_name).strip() if self.validator_name else None
        )
        self.description = str(self.description or "").strip()
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "version": self.version,
            "action_types": list(self.action_types),
            "target_types": list(self.target_types),
            "requires_approval": self.requires_approval,
            "dry_run_supported": self.dry_run_supported,
            "live_supported": self.live_supported,
            "sandbox_required": self.sandbox_required,
            "validator_name": self.validator_name,
            "description": self.description,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifestEntry":
        return cls(
            skill_name=data.get("skill_name", ""),
            version=data.get("version", "v1"),
            action_types=list(data.get("action_types", [])),
            target_types=list(data.get("target_types", [])),
            requires_approval=data.get("requires_approval", False),
            dry_run_supported=data.get("dry_run_supported", True),
            live_supported=data.get("live_supported", False),
            sandbox_required=data.get("sandbox_required", False),
            validator_name=data.get("validator_name"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


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
    action_type: ActionType | None = None
    plan_id: str | None = None
    plan_step_id: str | None = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    dry_run: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    skill_name: str = ""
    skill_version: str = ""
    action_type_name: str = ""
    target_type: str = ""
    policy_status: str | None = None
    approval_status: str | None = None
    sandbox_status: str | None = None
    retryable: bool = False
    requires_replan: bool = False

    def __post_init__(self) -> None:
        self.message = str(self.message or "").strip()
        self.dry_run = bool(self.dry_run)
        self.metadata = dict(self.metadata or {})
        self.skill_name = str(self.skill_name or "").strip().lower()
        self.skill_version = str(self.skill_version or "").strip() or "v0"
        self.action_type_name = str(self.action_type_name or "").strip().lower()
        if not self.action_type_name and self.action_type is not None:
            self.action_type_name = self.action_type.value
        self.target_type = str(self.target_type or "").strip().lower()
        self.policy_status = (
            str(self.policy_status).strip().lower() if self.policy_status else None
        )
        self.approval_status = (
            str(self.approval_status).strip().lower() if self.approval_status else None
        )
        self.sandbox_status = (
            str(self.sandbox_status).strip().lower() if self.sandbox_status else None
        )
        self.retryable = bool(self.retryable)
        self.requires_replan = bool(self.requires_replan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "action_type": self.action_type.value if self.action_type is not None else None,
            "plan_id": self.plan_id,
            "plan_step_id": self.plan_step_id,
            "status": self.status.value,
            "dry_run": self.dry_run,
            "message": self.message,
            "metadata": _serialize_value(self.metadata),
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "action_type_name": self.action_type_name,
            "target_type": self.target_type,
            "policy_status": self.policy_status,
            "approval_status": self.approval_status,
            "sandbox_status": self.sandbox_status,
            "retryable": self.retryable,
            "requires_replan": self.requires_replan,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        raw_action_type = data.get("action_type")
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
            action_type=ActionType(raw_action_type) if raw_action_type else None,
            plan_id=data.get("plan_id"),
            plan_step_id=data.get("plan_step_id"),
            status=ExecutionStatus(data.get("status", ExecutionStatus.SUCCESS.value)),
            dry_run=data.get("dry_run", True),
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
            skill_name=data.get("skill_name", ""),
            skill_version=data.get("skill_version", "v0"),
            action_type_name=data.get("action_type_name", ""),
            target_type=data.get("target_type", ""),
            policy_status=data.get("policy_status"),
            approval_status=data.get("approval_status"),
            sandbox_status=data.get("sandbox_status"),
            retryable=data.get("retryable", False),
            requires_replan=data.get("requires_replan", False),
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
