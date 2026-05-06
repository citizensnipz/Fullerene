"""Serializable models for latent pressure signal infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any, *, minimum: int = 0) -> int:
    try:
        return max(int(value), minimum)
    except (TypeError, ValueError):
        return minimum


def _parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, str) and raw:
        return datetime.fromisoformat(raw)
    return utcnow()


def _json_dict(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _json_list_of_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


@dataclass(slots=True)
class LatentPressureEntry:
    id: str = field(default_factory=lambda: uuid4().hex)
    source: str = "unknown"
    source_id: str | None = None
    entry_type: str = "unknown"
    description: str = ""
    intensity: float = 0.0
    decay_rate: float = 0.05
    escalation_rate: float = 0.08
    retrigger_count: int = 0
    created_at: datetime = field(default_factory=utcnow)
    last_activated_at: datetime = field(default_factory=utcnow)
    last_decayed_at: datetime = field(default_factory=utcnow)
    last_reactivation_event_type: str | None = None
    last_reactivation_source_id: str | None = None
    tick_reactivation_count: int = 0
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = str(self.source or "unknown").strip().lower() or "unknown"
        self.entry_type = str(self.entry_type or "unknown").strip().lower() or "unknown"
        self.description = str(self.description or "").strip()
        self.intensity = _clamp01(self.intensity)
        self.decay_rate = _clamp01(self.decay_rate)
        self.escalation_rate = _clamp01(self.escalation_rate)
        self.retrigger_count = _coerce_int(self.retrigger_count, minimum=0)
        self.tick_reactivation_count = _coerce_int(self.tick_reactivation_count, minimum=0)
        self.status = str(self.status or "active").strip().lower() or "active"
        self.metadata = _json_dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "entry_type": self.entry_type,
            "description": self.description,
            "intensity": _clamp01(self.intensity),
            "decay_rate": _clamp01(self.decay_rate),
            "escalation_rate": _clamp01(self.escalation_rate),
            "retrigger_count": _coerce_int(self.retrigger_count, minimum=0),
            "created_at": self.created_at.isoformat(),
            "last_activated_at": self.last_activated_at.isoformat(),
            "last_decayed_at": self.last_decayed_at.isoformat(),
            "last_reactivation_event_type": self.last_reactivation_event_type,
            "last_reactivation_source_id": self.last_reactivation_source_id,
            "tick_reactivation_count": _coerce_int(self.tick_reactivation_count, minimum=0),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentPressureEntry":
        return cls(
            id=str(data.get("id") or uuid4().hex),
            source=data.get("source", "unknown"),
            source_id=data.get("source_id"),
            entry_type=data.get("entry_type", "unknown"),
            description=data.get("description", ""),
            intensity=data.get("intensity", 0.0),
            decay_rate=data.get("decay_rate", 0.05),
            escalation_rate=data.get("escalation_rate", 0.08),
            retrigger_count=data.get("retrigger_count", 0),
            created_at=_parse_datetime(data.get("created_at")),
            last_activated_at=_parse_datetime(data.get("last_activated_at")),
            last_decayed_at=_parse_datetime(data.get("last_decayed_at")),
            last_reactivation_event_type=data.get("last_reactivation_event_type"),
            last_reactivation_source_id=data.get("last_reactivation_source_id"),
            tick_reactivation_count=data.get("tick_reactivation_count", 0),
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class LatentPressureResult:
    latent_pressure_total: float = 0.0
    active_entries: list[dict[str, Any]] = field(default_factory=list)
    top_entries: list[dict[str, Any]] = field(default_factory=list)
    created_entries: list[dict[str, Any]] = field(default_factory=list)
    updated_entries: list[dict[str, Any]] = field(default_factory=list)
    decayed_entries: list[dict[str, Any]] = field(default_factory=list)
    resolved_entries: list[dict[str, Any]] = field(default_factory=list)
    ignition_recommended: bool = False
    ignition_reason: str | None = None
    ignition_entry_id: str | None = None
    ignition_entry_type: str | None = None
    skipped_signals: list[dict[str, Any]] = field(default_factory=list)
    skipped_signal_count: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.latent_pressure_total = _clamp01(self.latent_pressure_total)
        self.active_entries = _json_list_of_dicts(self.active_entries)
        self.top_entries = _json_list_of_dicts(self.top_entries)
        self.created_entries = _json_list_of_dicts(self.created_entries)
        self.updated_entries = _json_list_of_dicts(self.updated_entries)
        self.decayed_entries = _json_list_of_dicts(self.decayed_entries)
        self.resolved_entries = _json_list_of_dicts(self.resolved_entries)
        self.ignition_recommended = bool(self.ignition_recommended)
        self.ignition_reason = (
            str(self.ignition_reason).strip() if self.ignition_reason else None
        )
        self.ignition_entry_id = (
            str(self.ignition_entry_id).strip() if self.ignition_entry_id else None
        )
        self.ignition_entry_type = (
            str(self.ignition_entry_type).strip() if self.ignition_entry_type else None
        )
        self.skipped_signals = _json_list_of_dicts(self.skipped_signals)
        self.skipped_signal_count = _coerce_int(
            self.skipped_signal_count or len(self.skipped_signals), minimum=0
        )
        self.skip_reasons = [str(item) for item in self.skip_reasons if str(item).strip()]
        self.reasons = [str(item) for item in self.reasons if str(item).strip()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_pressure_total": _clamp01(self.latent_pressure_total),
            "active_entries": list(self.active_entries),
            "top_entries": list(self.top_entries),
            "created_entries": list(self.created_entries),
            "updated_entries": list(self.updated_entries),
            "decayed_entries": list(self.decayed_entries),
            "resolved_entries": list(self.resolved_entries),
            "ignition_recommended": bool(self.ignition_recommended),
            "ignition_reason": self.ignition_reason,
            "ignition_entry_id": self.ignition_entry_id,
            "ignition_entry_type": self.ignition_entry_type,
            "skipped_signals": list(self.skipped_signals),
            "skipped_signal_count": _coerce_int(self.skipped_signal_count, minimum=0),
            "skip_reasons": list(self.skip_reasons),
            "reasons": list(self.reasons),
            "active_count": len(self.active_entries),
            "total_count": len(self.active_entries) + len(self.resolved_entries),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentPressureResult":
        return cls(
            latent_pressure_total=data.get("latent_pressure_total", 0.0),
            active_entries=data.get("active_entries", []),
            top_entries=data.get("top_entries", []),
            created_entries=data.get("created_entries", []),
            updated_entries=data.get("updated_entries", []),
            decayed_entries=data.get("decayed_entries", []),
            resolved_entries=data.get("resolved_entries", []),
            ignition_recommended=bool(data.get("ignition_recommended", False)),
            ignition_reason=data.get("ignition_reason"),
            ignition_entry_id=data.get("ignition_entry_id"),
            ignition_entry_type=data.get("ignition_entry_type"),
            skipped_signals=data.get("skipped_signals", []),
            skipped_signal_count=data.get("skipped_signal_count", 0),
            skip_reasons=data.get("skip_reasons", []),
            reasons=data.get("reasons", []),
        )

