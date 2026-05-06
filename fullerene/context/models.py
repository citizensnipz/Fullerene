"""Models for Fullerene Context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


STATIC_RECENT_EPISODIC_V0 = "static_recent_episodic_v0"
DYNAMIC_ACTIVE_FACETS_V1 = "dynamic_active_facets_v1"
PRESSURE_RELEVANCE_V2 = "pressure_relevance_v2"


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


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


class ContextItemType(str, Enum):
    EVENT = "event"
    WORKING_MEMORY = "working_memory"
    CONVERSATION_CONTINUITY = "conversation_continuity"
    ATTENTION = "attention"
    MEMORY = "memory"
    GOAL = "goal"
    BELIEF = "belief"
    POLICY = "policy"
    SIGNAL = "signal"
    SYSTEM = "system"


@dataclass(slots=True)
class ReferenceAnchor:
    anchor_id: str
    surface_form: str
    referent_text: str
    referent_source_turn_id: str | None = None
    referent_source_role: str | None = None
    confidence: float = 0.0
    reason: str | None = None
    current_message_fragment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.anchor_id = str(self.anchor_id or uuid4().hex)
        self.surface_form = str(self.surface_form or "").strip().lower()
        self.referent_text = str(self.referent_text or "").strip()
        self.referent_source_turn_id = (
            str(self.referent_source_turn_id).strip()
            if self.referent_source_turn_id is not None
            else None
        )
        self.referent_source_role = (
            str(self.referent_source_role).strip().lower()
            if self.referent_source_role is not None
            else None
        )
        try:
            conf = float(self.confidence)
        except (TypeError, ValueError):
            conf = 0.0
        self.confidence = max(0.0, min(conf, 1.0))
        self.reason = str(self.reason).strip() if self.reason is not None else None
        if self.current_message_fragment is not None:
            self.current_message_fragment = str(self.current_message_fragment).strip()
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "surface_form": self.surface_form,
            "referent_text": self.referent_text,
            "referent_source_turn_id": self.referent_source_turn_id,
            "referent_source_role": self.referent_source_role,
            "confidence": self.confidence,
            "reason": self.reason,
            "current_message_fragment": self.current_message_fragment,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceAnchor":
        return cls(
            anchor_id=str(data.get("anchor_id") or uuid4().hex),
            surface_form=str(data.get("surface_form") or ""),
            referent_text=str(data.get("referent_text") or ""),
            referent_source_turn_id=data.get("referent_source_turn_id"),
            referent_source_role=data.get("referent_source_role"),
            confidence=float(data.get("confidence") or 0.0),
            reason=data.get("reason"),
            current_message_fragment=data.get("current_message_fragment"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ConversationContinuity:
    current_topic_hint: str | None = None
    topic_terms: list[str] = field(default_factory=list)
    reference_anchors: list[ReferenceAnchor] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    continuity_confidence: float = 0.0
    working_memory_turn_count: int = 0
    source: str = "working_memory"

    def __post_init__(self) -> None:
        self.current_topic_hint = (
            str(self.current_topic_hint).strip()
            if self.current_topic_hint is not None and str(self.current_topic_hint).strip()
            else None
        )
        self.topic_terms = [
            str(term).strip().lower()
            for term in self.topic_terms
            if str(term).strip()
        ]
        self.reference_anchors = list(self.reference_anchors or [])
        self.unresolved_references = [
            str(token).strip().lower()
            for token in self.unresolved_references
            if str(token).strip()
        ]
        try:
            conf = float(self.continuity_confidence)
        except (TypeError, ValueError):
            conf = 0.0
        self.continuity_confidence = max(0.0, min(conf, 1.0))
        self.working_memory_turn_count = max(int(self.working_memory_turn_count or 0), 0)
        self.source = str(self.source or "working_memory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_topic_hint": self.current_topic_hint,
            "topic_terms": list(self.topic_terms),
            "reference_anchors": [anchor.to_dict() for anchor in self.reference_anchors],
            "unresolved_references": list(self.unresolved_references),
            "continuity_confidence": self.continuity_confidence,
            "working_memory_turn_count": self.working_memory_turn_count,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContinuity":
        raw_anchors = data.get("reference_anchors", [])
        anchors: list[ReferenceAnchor] = []
        if isinstance(raw_anchors, list):
            for raw in raw_anchors:
                if isinstance(raw, dict):
                    anchors.append(ReferenceAnchor.from_dict(raw))
        return cls(
            current_topic_hint=data.get("current_topic_hint"),
            topic_terms=data.get("topic_terms", []),
            reference_anchors=anchors,
            unresolved_references=data.get("unresolved_references", []),
            continuity_confidence=float(data.get("continuity_confidence") or 0.0),
            working_memory_turn_count=int(data.get("working_memory_turn_count") or 0),
            source=str(data.get("source") or "working_memory"),
        )


@dataclass(slots=True)
class ContextAssemblyConfig:
    max_goals: int = 3
    max_items_total: int = 16
    max_working_memory_turns: int = 8
    max_long_term_memories: int = 5
    max_memories: int = 5
    max_beliefs: int = 5
    max_lpb_items: int = 3
    max_attention_items: int = 2
    max_policy_items: int = 2
    max_working_turns: int = 8
    min_relevance_score: float = 0.15
    min_pressure_score: float = 0.20
    salience_threshold: float = 0.0
    include_working_memory: bool = True
    include_lpb: bool = True
    include_attention: bool = True
    include_policy: bool = True
    include_world_model: bool = True
    include_goals: bool = True
    include_recent_signals: bool = True
    include_policy_summary: bool = True
    include_signal_summaries: bool = True
    strategy: str = DYNAMIC_ACTIVE_FACETS_V1

    def __post_init__(self) -> None:
        raw_strategy = str(self.strategy or DYNAMIC_ACTIVE_FACETS_V1).strip().lower()
        if raw_strategy in {"dynamic", DYNAMIC_ACTIVE_FACETS_V1}:
            self.strategy = DYNAMIC_ACTIVE_FACETS_V1
        elif raw_strategy in {"static", STATIC_RECENT_EPISODIC_V0}:
            self.strategy = STATIC_RECENT_EPISODIC_V0
        elif raw_strategy == PRESSURE_RELEVANCE_V2:
            self.strategy = PRESSURE_RELEVANCE_V2
        else:
            self.strategy = DYNAMIC_ACTIVE_FACETS_V1
        self.max_items_total = max(int(self.max_items_total), 1)
        self.max_working_memory_turns = max(int(self.max_working_memory_turns), 0)
        self.max_long_term_memories = max(int(self.max_long_term_memories), 0)
        self.max_goals = max(int(self.max_goals), 0)
        self.max_beliefs = max(int(self.max_beliefs), 0)
        self.max_lpb_items = max(int(self.max_lpb_items), 0)
        self.max_attention_items = max(int(self.max_attention_items), 0)
        self.max_policy_items = max(int(self.max_policy_items), 0)
        self.min_relevance_score = max(0.0, min(float(self.min_relevance_score), 1.0))
        self.min_pressure_score = max(0.0, min(float(self.min_pressure_score), 1.0))
        self.include_working_memory = bool(self.include_working_memory)
        self.include_lpb = bool(self.include_lpb)
        self.include_attention = bool(self.include_attention)
        self.include_policy = bool(self.include_policy)
        self.include_world_model = bool(self.include_world_model)
        self.include_goals = bool(self.include_goals)
        self.include_recent_signals = bool(self.include_recent_signals)
        self.max_memories = max(int(self.max_memories), 0)
        if self.max_memories != self.max_long_term_memories:
            self.max_long_term_memories = self.max_memories
        self.max_memories = self.max_long_term_memories
        self.max_working_turns = max(int(self.max_working_turns), 0)
        if self.max_working_turns != self.max_working_memory_turns:
            self.max_working_turns = self.max_working_memory_turns
        self.salience_threshold = max(0.0, min(float(self.salience_threshold), 1.0))
        self.include_policy_summary = bool(self.include_policy_summary)
        self.include_signal_summaries = bool(self.include_signal_summaries)
        self.include_policy_summary = self.include_policy
        self.include_signal_summaries = self.include_recent_signals

    @property
    def max_items(self) -> int:
        if self.strategy == PRESSURE_RELEVANCE_V2:
            return self.max_items_total
        policy_items = 1 if self.include_policy_summary else 0
        signal_items = 5 if self.include_signal_summaries else 0
        total = 1 + self.max_goals + self.max_memories + self.max_beliefs + policy_items + signal_items
        return max(total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_goals": self.max_goals,
            "max_items_total": self.max_items_total,
            "max_working_memory_turns": self.max_working_memory_turns,
            "max_long_term_memories": self.max_long_term_memories,
            "max_memories": self.max_memories,
            "max_beliefs": self.max_beliefs,
            "max_lpb_items": self.max_lpb_items,
            "max_attention_items": self.max_attention_items,
            "max_policy_items": self.max_policy_items,
            "min_relevance_score": self.min_relevance_score,
            "min_pressure_score": self.min_pressure_score,
            "include_working_memory": self.include_working_memory,
            "include_lpb": self.include_lpb,
            "include_attention": self.include_attention,
            "include_policy": self.include_policy,
            "include_world_model": self.include_world_model,
            "include_goals": self.include_goals,
            "include_recent_signals": self.include_recent_signals,
            "max_working_turns": self.max_working_turns,
            "salience_threshold": self.salience_threshold,
            "include_policy_summary": self.include_policy_summary,
            "include_signal_summaries": self.include_signal_summaries,
            "strategy": self.strategy,
            "max_items": self.max_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextAssemblyConfig":
        return cls(
            max_goals=data.get("max_goals", 3),
            max_items_total=data.get("max_items_total", 16),
            max_working_memory_turns=data.get(
                "max_working_memory_turns",
                data.get("max_working_turns", 8),
            ),
            max_long_term_memories=data.get(
                "max_long_term_memories",
                data.get("max_memories", 5),
            ),
            max_beliefs=data.get("max_beliefs", 5),
            max_lpb_items=data.get("max_lpb_items", 3),
            max_attention_items=data.get("max_attention_items", 2),
            max_policy_items=data.get("max_policy_items", 2),
            min_relevance_score=data.get("min_relevance_score", 0.15),
            min_pressure_score=data.get("min_pressure_score", 0.20),
            include_working_memory=data.get("include_working_memory", True),
            include_lpb=data.get("include_lpb", True),
            include_attention=data.get("include_attention", True),
            include_policy=data.get(
                "include_policy",
                data.get("include_policy_summary", True),
            ),
            include_world_model=data.get("include_world_model", True),
            include_goals=data.get("include_goals", True),
            include_recent_signals=data.get(
                "include_recent_signals",
                data.get("include_signal_summaries", True),
            ),
            max_memories=data.get("max_memories", 5),
            max_working_turns=data.get("max_working_turns", 8),
            salience_threshold=data.get("salience_threshold", 0.0),
            include_policy_summary=data.get("include_policy_summary", True),
            include_signal_summaries=data.get("include_signal_summaries", True),
            strategy=data.get("strategy", DYNAMIC_ACTIVE_FACETS_V1),
        )


@dataclass(slots=True)
class ContextItem:
    id: str = field(default_factory=lambda: uuid4().hex)
    item_type: ContextItemType = ContextItemType.MEMORY
    content: str = ""
    source_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_type": self.item_type.value,
            "content": self.content,
            "source_id": self.source_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextItem":
        return cls(
            id=data["id"],
            item_type=ContextItemType(data["item_type"]),
            content=data.get("content", ""),
            source_id=data.get("source_id"),
            created_at=_parse_datetime(data.get("created_at")),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ContextWindow:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    items: list[ContextItem] = field(default_factory=list)
    max_items: int = 5
    strategy: str = STATIC_RECENT_EPISODIC_V0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_items = max(int(self.max_items), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
            "max_items": self.max_items,
            "strategy": self.strategy,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextWindow":
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            items=[ContextItem.from_dict(item) for item in data.get("items", [])],
            max_items=data.get("max_items", 5),
            strategy=data.get("strategy", STATIC_RECENT_EPISODIC_V0),
            metadata=data.get("metadata", {}),
        )
