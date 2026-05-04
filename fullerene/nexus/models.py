"""Typed models used by the Nexus runtime."""

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


class EventType(str, Enum):
    USER_MESSAGE = "user_message"
    SYSTEM_TICK = "system_tick"
    SYSTEM_NOTE = "system_note"
    INTERNAL = "internal"


class DecisionAction(str, Enum):
    WAIT = "wait"
    ASK = "ask"
    ACT = "act"
    RECORD = "record"


@dataclass(slots=True)
class Event:
    event_type: EventType
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "content": self.content,
            "metadata": _serialize_value(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            timestamp=_parse_datetime(data["timestamp"]),
        )


@dataclass(slots=True)
class FacetResult:
    facet_name: str
    summary: str
    proposed_decision: DecisionAction | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facet_name": self.facet_name,
            "summary": self.summary,
            "proposed_decision": (
                self.proposed_decision.value if self.proposed_decision else None
            ),
            "state_updates": _serialize_value(self.state_updates),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FacetResult":
        raw_decision = data.get("proposed_decision")
        return cls(
            facet_name=data["facet_name"],
            summary=data["summary"],
            proposed_decision=DecisionAction(raw_decision) if raw_decision else None,
            state_updates=data.get("state_updates", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class NexusDecision:
    action: DecisionAction
    reason: str
    source_facets: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "source_facets": list(self.source_facets),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusDecision":
        return cls(
            action=DecisionAction(data["action"]),
            reason=data["reason"],
            source_facets=list(data.get("source_facets", [])),
            timestamp=_parse_datetime(data["timestamp"]),
        )


@dataclass(slots=True)
class NexusState:
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    event_count: int = 0
    last_event: Event | None = None
    last_decision: NexusDecision | None = None
    system_pressure: float = 0.0
    facet_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def apply(
        self,
        event: Event,
        facet_results: list[FacetResult],
        decision: NexusDecision,
        *,
        system_pressure: float | None = None,
    ) -> None:
        self.event_count += 1
        self.last_event = event
        self.last_decision = decision
        if system_pressure is not None:
            self.system_pressure = _clamp_unit(system_pressure)
        for result in facet_results:
            if not result.state_updates:
                continue
            facet_bucket = self.facet_state.setdefault(result.facet_name, {})
            facet_bucket.update(result.state_updates)
        self.updated_at = utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "event_count": self.event_count,
            "last_event": self.last_event.to_dict() if self.last_event else None,
            "last_decision": (
                self.last_decision.to_dict() if self.last_decision else None
            ),
            "system_pressure": self.system_pressure,
            "facet_state": _serialize_value(self.facet_state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusState":
        return cls(
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            event_count=data.get("event_count", 0),
            last_event=(
                Event.from_dict(data["last_event"]) if data.get("last_event") else None
            ),
            last_decision=(
                NexusDecision.from_dict(data["last_decision"])
                if data.get("last_decision")
                else None
            ),
            system_pressure=_clamp_unit(data.get("system_pressure", 0.0)),
            facet_state=data.get("facet_state", {}),
        )


@dataclass(slots=True)
class NexusRecord:
    event: Event
    facet_results: list[FacetResult]
    decision: NexusDecision
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)
    record_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "event": self.event.to_dict(),
            "facet_results": [result.to_dict() for result in self.facet_results],
            "decision": self.decision.to_dict(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusRecord":
        return cls(
            record_id=data["record_id"],
            timestamp=_parse_datetime(data["timestamp"]),
            event=Event.from_dict(data["event"]),
            facet_results=[
                FacetResult.from_dict(result) for result in data.get("facet_results", [])
            ],
            decision=NexusDecision.from_dict(data["decision"]),
            metadata=data.get("metadata", {}),
        )


def _clamp_unit(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
