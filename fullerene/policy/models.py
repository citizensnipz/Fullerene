"""Policy models for deterministic Fullerene policy storage and evaluation."""

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


class PolicyRuleType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    PREFER = "prefer"


class PolicyTargetType(str, Enum):
    INTERNAL_STATE = "internal_state"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL = "shell"
    NETWORK = "network"
    MESSAGE = "message"
    GIT = "git"
    TOOL = "tool"
    DECISION = "decision"
    TAG = "tag"
    GENERAL = "general"


class PolicySource(str, Enum):
    USER = "user"
    SYSTEM = "system"


class PolicyStatus(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
    PREFERRED = "preferred"
    NO_MATCH = "no_match"


class PolicyRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class PolicyEffectiveAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    PREFER = "prefer"
    NONE = "none"


class PolicyApprovalScope(str, Enum):
    PLAN_STEP = "plan_step"
    PLAN = "plan"
    ACTION = "action"
    SESSION = "session"
    NONE = "none"


@dataclass(slots=True)
class PolicyPrecedenceTraceEntry:
    rule_id: str
    rule_name: str
    rule_type: str
    priority: float
    match_score: float | None = None
    match_reason: str | None = None
    conditions_matched: list[str] = field(default_factory=list)
    effect: str = ""
    decisive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.rule_name,
            "rule_type/action": self.rule_type,
            "priority": float(self.priority),
            "match_score": self.match_score,
            "match_reason": self.match_reason,
            "conditions_matched": list(self.conditions_matched),
            "effect": self.effect,
            "decisive": bool(self.decisive),
        }


@dataclass(slots=True)
class PolicyEvaluation:
    """Deterministic v1 policy evaluation (JSON-serializable)."""

    status: str
    effective_action: str
    target_type: str | None
    target: str | None
    risk_level: str

    matched_rule_ids: list[str] = field(default_factory=list)
    matched_rule_names: list[str] = field(default_factory=list)

    decisive_rule_id: str | None = None
    decisive_rule_name: str | None = None
    decisive_reason: str = ""

    approval_required: bool = False
    approval_token_required: bool = False
    approval_token_valid: bool = False
    approval_scope: str = PolicyApprovalScope.NONE.value

    rule_precedence_trace: list[PolicyPrecedenceTraceEntry] = field(
        default_factory=list
    )

    fallback_applied: bool = False
    sandbox_scope: str = ""

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "effective_action": self.effective_action,
            "target_type": self.target_type,
            "target": self.target,
            "risk_level": self.risk_level,
            "matched_rule_ids": list(self.matched_rule_ids),
            "matched_rule_names": list(self.matched_rule_names),
            "decisive_rule_id": self.decisive_rule_id,
            "decisive_rule_name": self.decisive_rule_name,
            "decisive_reason": self.decisive_reason,
            "approval_required": bool(self.approval_required),
            "approval_token_required": bool(self.approval_token_required),
            "approval_token_valid": bool(self.approval_token_valid),
            "approval_scope": self.approval_scope,
            "rule_precedence_trace": [
                entry.to_dict() for entry in self.rule_precedence_trace
            ],
            "fallback_applied": bool(self.fallback_applied),
            "sandbox_scope": self.sandbox_scope,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def coerce_policy_rule_type(raw_value: Any) -> PolicyRuleType | None:
    if isinstance(raw_value, PolicyRuleType):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    if not cleaned:
        return None
    try:
        return PolicyRuleType(cleaned)
    except ValueError:
        return None


def coerce_policy_target_type(raw_value: Any) -> PolicyTargetType | None:
    if isinstance(raw_value, PolicyTargetType):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    if not cleaned:
        return None
    try:
        return PolicyTargetType(cleaned)
    except ValueError:
        return None


def coerce_policy_source(raw_value: Any) -> PolicySource | None:
    if isinstance(raw_value, PolicySource):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    if not cleaned:
        return None
    try:
        return PolicySource(cleaned)
    except ValueError:
        return None


@dataclass(slots=True)
class PolicyRule:
    id: str = field(default_factory=lambda: uuid4().hex)
    name: str = ""
    description: str = ""
    rule_type: PolicyRuleType = PolicyRuleType.ALLOW
    target_type: PolicyTargetType = PolicyTargetType.GENERAL
    target: str = "*"
    conditions: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    enabled: bool = True
    source: PolicySource = PolicySource.USER
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name or self.id).strip() or self.id
        self.description = str(self.description or "").strip()
        self.target = str(self.target or "*").strip() or "*"
        self.conditions = dict(self.conditions or {})
        self.priority = float(self.priority)
        self.enabled = bool(self.enabled)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "target_type": self.target_type.value,
            "target": self.target,
            "conditions": _serialize_value(self.conditions),
            "priority": self.priority,
            "enabled": self.enabled,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyRule":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            rule_type=PolicyRuleType(data.get("rule_type", PolicyRuleType.ALLOW.value)),
            target_type=PolicyTargetType(
                data.get("target_type", PolicyTargetType.GENERAL.value)
            ),
            target=data.get("target", "*"),
            conditions=data.get("conditions", {}),
            priority=data.get("priority", 0.0),
            enabled=data.get("enabled", True),
            source=PolicySource(data.get("source", PolicySource.USER.value)),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
