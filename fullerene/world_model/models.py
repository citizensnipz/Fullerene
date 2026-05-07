"""Belief models for deterministic Fullerene world model storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import uuid4

from fullerene.memory.models import normalize_tags


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


class BeliefStatus(str, Enum):
    VALID = "valid"
    CONTRADICTED = "contradicted"
    REDUNDANT = "redundant"
    # Compatibility aliases for v0 rows/tests.
    ACTIVE = "valid"
    STALE = "redundant"
    RETIRED = "redundant"


class BeliefSource(str, Enum):
    USER = "user"
    SYSTEM = "system"
    MEMORY = "memory"
    GOAL = "goal"
    CONTEXT = "context"
    RUNTIME = "runtime"


class BeliefType(str, Enum):
    FACT = "fact"
    CAPABILITY = "capability"
    PREFERENCE = "preference"
    UNKNOWN = "unknown"


# World Model v2 graph artifacts (JSON-serializable via to_dict helpers below).
BELIEF_EDGE_TYPES = frozenset(
    {
        "related",
        "supporting",
        "contradicting",
        "causal",
        "temporal",
        "inferred_from",
    }
)


def stable_belief_edge_id(source_belief_id: str, target_belief_id: str, edge_type: str) -> str:
    import hashlib

    a, b = sorted((str(source_belief_id), str(target_belief_id)))
    raw = f"{a}|{b}|{str(edge_type).strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]  # noqa: S324


def belief_edge_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize edge row for API/verifier consumption."""
    out = dict(row)
    meta = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    prov = out.get("provenance") if isinstance(out.get("provenance"), dict) else {}
    out["metadata"] = _serialize_value(meta)
    out["provenance"] = _serialize_value(prov)
    return out


def belief_community_to_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize_value(v) for k, v in data.items()}


def belief_rule_to_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize_value(v) for k, v in data.items()}


def normalize_statement(text: str) -> str:
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return " ".join(cleaned.split()).strip()


def stable_belief_id(normalized_key: str) -> str:
    import hashlib

    payload = normalized_key.strip().encode("utf-8")
    return hashlib.sha1(payload).hexdigest()  # noqa: S324 deterministic id


def _normalize_sources(raw_sources: Iterable[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for source in raw_sources or ():
        text = str(source).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


@dataclass(slots=True)
class Belief:
    id: str = field(default_factory=lambda: uuid4().hex)
    claim: str = ""
    confidence: float = 0.5
    status: BeliefStatus = BeliefStatus.VALID
    tags: list[str] = field(default_factory=list)
    source: BeliefSource = BeliefSource.USER
    source_event_id: str | None = None
    source_memory_id: str | None = None
    sources: list[str] = field(default_factory=list)
    normalized_key: str = ""
    belief_type: BeliefType = BeliefType.UNKNOWN
    support_count: int = 0
    contradiction_count: int = 0
    last_support_event_id: str | None = None
    last_contradiction_event_id: str | None = None
    last_updated_event_id: str | None = None
    priority: float = 1.0
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = self._validate_confidence(self.confidence)
        self.priority = self._validate_confidence(self.priority)
        self.tags = normalize_tags(self.tags)
        self.sources = _normalize_sources(self.sources)
        if not isinstance(self.belief_type, BeliefType):
            self.belief_type = BeliefType(str(self.belief_type).strip().lower() or "unknown")
        self.normalized_key = self.normalized_key or normalize_statement(self.claim)
        self.support_count = max(int(self.support_count), 0)
        self.contradiction_count = max(int(self.contradiction_count), 0)

    @staticmethod
    def _validate_confidence(value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "confidence": self.confidence,
            "status": self.status.value,
            "tags": list(self.tags),
            "source": self.source.value,
            "source_event_id": self.source_event_id,
            "source_memory_id": self.source_memory_id,
            "sources": list(self.sources),
            "normalized_key": self.normalized_key,
            "belief_type": self.belief_type.value,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "last_support_event_id": self.last_support_event_id,
            "last_contradiction_event_id": self.last_contradiction_event_id,
            "last_updated_event_id": self.last_updated_event_id,
            "last_updated_timestamp": self.updated_at.isoformat(),
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Belief":
        return cls(
            id=data["id"],
            claim=data.get("claim", ""),
            confidence=data.get("confidence", 0.5),
            status=BeliefStatus(data.get("status", BeliefStatus.ACTIVE.value)),
            tags=data.get("tags", []),
            source=BeliefSource(data.get("source", BeliefSource.USER.value)),
            source_event_id=data.get("source_event_id"),
            source_memory_id=data.get("source_memory_id"),
            sources=data.get("sources", []),
            normalized_key=data.get("normalized_key", ""),
            belief_type=BeliefType(data.get("belief_type", BeliefType.UNKNOWN.value)),
            support_count=data.get("support_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            last_support_event_id=data.get("last_support_event_id"),
            last_contradiction_event_id=data.get("last_contradiction_event_id"),
            last_updated_event_id=data.get("last_updated_event_id"),
            priority=data.get("priority", 1.0),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
