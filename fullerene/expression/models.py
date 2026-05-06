"""Serializable models for Expression Gate v0 (recommend-only; no prose generation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExpressionMode(str, Enum):
    silent = "silent"
    log_only = "log_only"
    status_only = "status_only"
    short_utterance = "short_utterance"
    ask_user = "ask_user"


class SuggestedIntent(str, Enum):
    status_update = "status_update"
    ask_approval = "ask_approval"
    ask_clarification = "ask_clarification"
    surface_warning = "surface_warning"
    surface_unresolved_pressure = "surface_unresolved_pressure"
    none = "none"


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _bounded_dict(raw: Any, *, max_keys: int = 32, max_depth: int = 3) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    def _walk(obj: Any, d: int) -> Any:
        if d <= 0:
            return "<max_depth>"
        if obj is None or isinstance(obj, (bool, str, int)):
            return obj
        if isinstance(obj, float):
            if obj != obj:
                return 0.0
            return round(float(obj), 6)
        if isinstance(obj, dict):
            return {
                str(k): _walk(v, d - 1)
                for k, v in list(obj.items())[:max_keys]
                if isinstance(k, (str, int))
            }
        if isinstance(obj, (list, tuple)):
            return [_walk(v, d - 1) for v in obj[:24]]
        return str(obj)

    return _walk(dict(raw), max_depth)


@dataclass(slots=True)
class ExpressionRecommendation:
    mode: ExpressionMode
    expression_score: float
    reasons: list[str]
    source_event_id: str
    source_cycle_id: str | None
    source_candidate_id: str | None
    allowed_user_facing: bool
    requires_user_attention: bool
    cooldown_applied: bool
    budget_applied: bool
    suppressed: bool
    suppression_reason: str
    max_words: int
    suggested_intent: SuggestedIntent
    payload: dict[str, Any]
    metadata: dict[str, Any]
    suppression_rules_triggered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "expression_score": _clamp01(self.expression_score),
            "reasons": list(self.reasons),
            "source_event_id": str(self.source_event_id or ""),
            "source_cycle_id": self.source_cycle_id,
            "source_candidate_id": self.source_candidate_id,
            "allowed_user_facing": bool(self.allowed_user_facing),
            "requires_user_attention": bool(self.requires_user_attention),
            "cooldown_applied": bool(self.cooldown_applied),
            "budget_applied": bool(self.budget_applied),
            "suppressed": bool(self.suppressed),
            "suppression_reason": str(self.suppression_reason or ""),
            "suppression_rules_triggered": list(self.suppression_rules_triggered),
            "max_words": max(0, int(self.max_words)),
            "suggested_intent": self.suggested_intent.value,
            "payload": _bounded_dict(self.payload),
            "metadata": _bounded_dict(self.metadata, max_keys=48),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpressionRecommendation":
        mode_raw = data.get("mode", ExpressionMode.silent.value)
        try:
            mode = ExpressionMode(str(mode_raw))
        except ValueError:
            mode = ExpressionMode.silent
        intent_raw = data.get("suggested_intent", SuggestedIntent.none.value)
        try:
            intent = SuggestedIntent(str(intent_raw))
        except ValueError:
            intent = SuggestedIntent.none
        return cls(
            mode=mode,
            expression_score=_clamp01(data.get("expression_score", 0.0)),
            reasons=list(data.get("reasons") or []),
            source_event_id=str(data.get("source_event_id", "")),
            source_cycle_id=data.get("source_cycle_id"),
            source_candidate_id=data.get("source_candidate_id"),
            allowed_user_facing=bool(data.get("allowed_user_facing", False)),
            requires_user_attention=bool(data.get("requires_user_attention", False)),
            cooldown_applied=bool(data.get("cooldown_applied", False)),
            budget_applied=bool(data.get("budget_applied", False)),
            suppressed=bool(data.get("suppressed", False)),
            suppression_reason=str(data.get("suppression_reason", "") or ""),
            suppression_rules_triggered=list(
                data.get("suppression_rules_triggered") or [],
            ),
            max_words=max(0, int(data.get("max_words", 0) or 0)),
            suggested_intent=intent,
            payload=dict(data.get("payload") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class ExpressionBudgetState:
    last_expression_at: str | None
    expression_count_window: int
    last_expression_hash: str | None
    ignored_expression_count: int
    cooldowns: dict[str, Any]
    history: list[dict[str, Any]]
    window_seconds: int = 600
    last_user_facing_at: str | None = None

    def __post_init__(self) -> None:
        self.expression_count_window = max(0, int(self.expression_count_window or 0))
        self.ignored_expression_count = max(0, int(self.ignored_expression_count or 0))
        self.cooldowns = dict(self.cooldowns or {})
        self.history = [dict(item) for item in (self.history or [])][:20]
        self.window_seconds = max(60, int(self.window_seconds or 600))

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_expression_at": self.last_expression_at,
            "expression_count_window": int(self.expression_count_window),
            "last_expression_hash": self.last_expression_hash,
            "ignored_expression_count": int(self.ignored_expression_count),
            "cooldowns": dict(self.cooldowns),
            "history": list(self.history)[:20],
            "window_seconds": int(self.window_seconds),
            "last_user_facing_at": self.last_user_facing_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExpressionBudgetState":
        if not isinstance(data, dict):
            data = {}
        return cls(
            last_expression_at=data.get("last_expression_at"),
            expression_count_window=int(data.get("expression_count_window", 0) or 0),
            last_expression_hash=data.get("last_expression_hash"),
            ignored_expression_count=int(data.get("ignored_expression_count", 0) or 0),
            cooldowns=dict(data.get("cooldowns") or {}),
            history=list(data.get("history") or []),
            window_seconds=int(data.get("window_seconds", 600) or 600),
            last_user_facing_at=data.get("last_user_facing_at"),
        )

    def compact_summary(self) -> dict[str, Any]:
        return {
            "expression_count_window": self.expression_count_window,
            "last_expression_hash": self.last_expression_hash,
            "ignored_expression_count": self.ignored_expression_count,
            "cooldown_keys": sorted([str(k) for k in self.cooldowns.keys()])[:12],
            "history_len": len(self.history),
            "window_seconds": self.window_seconds,
            "has_last_expression_at": bool(self.last_expression_at),
            "has_last_user_facing_at": bool(self.last_user_facing_at),
        }


def expression_mode_rank(mode: ExpressionMode) -> int:
    order = (
        ExpressionMode.silent,
        ExpressionMode.log_only,
        ExpressionMode.status_only,
        ExpressionMode.short_utterance,
        ExpressionMode.ask_user,
    )
    try:
        return order.index(mode)
    except ValueError:
        return 0


def max_expression_mode(a: ExpressionMode, b: ExpressionMode) -> ExpressionMode:
    return a if expression_mode_rank(a) >= expression_mode_rank(b) else b
