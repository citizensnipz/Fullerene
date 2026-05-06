"""Memory v3 community / cluster models (thematic concern areas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fullerene.memory.models import utcnow


def _json_dict(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if str(x).strip()]


@dataclass(slots=True)
class MemoryCommunity:
    """Persisted memory cluster (community) used for bounded graph + pressure."""

    community_id: str
    label: str = ""
    member_memory_ids: list[str] = field(default_factory=list)
    member_count: int = 0
    top_tags: list[str] = field(default_factory=list)
    top_domains: list[str] = field(default_factory=list)
    top_roles: list[str] = field(default_factory=list)
    representative_memory_ids: list[str] = field(default_factory=list)
    centroid_embedding_id: str | None = None
    centroid_vector_hash: str | None = None
    activation_score: float = 0.0
    pressure_score: float = 0.0
    unresolved_score: float = 0.0
    contradiction_count: int = 0
    refinement_count: int = 0
    activation_streak: int = 0
    inactive_streak: int = 0
    last_activated_at: datetime | None = None
    last_activated_event_id: str | None = None
    last_pressure_update_at: datetime | None = None
    last_resolution_event_id: str | None = None
    resolved_recently: bool = False
    activation_reasons: list[str] = field(default_factory=list)
    pressure_reasons: list[str] = field(default_factory=list)
    community_detection_strategy: str = "deterministic_connected_components_v0"
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "label": self.label,
            "member_memory_ids": list(self.member_memory_ids),
            "member_count": int(self.member_count),
            "top_tags": list(self.top_tags),
            "top_domains": list(self.top_domains),
            "top_roles": list(self.top_roles),
            "representative_memory_ids": list(self.representative_memory_ids),
            "centroid_embedding_id": self.centroid_embedding_id,
            "centroid_vector_hash": self.centroid_vector_hash,
            "activation_score": float(self.activation_score),
            "pressure_score": float(self.pressure_score),
            "unresolved_score": float(self.unresolved_score),
            "contradiction_count": int(self.contradiction_count),
            "refinement_count": int(self.refinement_count),
            "activation_streak": int(self.activation_streak),
            "inactive_streak": int(self.inactive_streak),
            "last_activated_at": (
                self.last_activated_at.isoformat() if self.last_activated_at else None
            ),
            "last_activated_event_id": self.last_activated_event_id,
            "last_pressure_update_at": (
                self.last_pressure_update_at.isoformat()
                if self.last_pressure_update_at
                else None
            ),
            "last_resolution_event_id": self.last_resolution_event_id,
            "resolved_recently": bool(self.resolved_recently),
            "activation_reasons": list(self.activation_reasons),
            "pressure_reasons": list(self.pressure_reasons),
            "community_detection_strategy": self.community_detection_strategy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": _json_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryCommunity:
        def _dt(key: str) -> datetime | None:
            raw = data.get(key)
            if not raw:
                return None
            return datetime.fromisoformat(str(raw))

        return cls(
            community_id=str(data["community_id"]),
            label=str(data.get("label") or ""),
            member_memory_ids=_str_list(data.get("member_memory_ids")),
            member_count=int(data.get("member_count") or 0),
            top_tags=_str_list(data.get("top_tags")),
            top_domains=_str_list(data.get("top_domains")),
            top_roles=_str_list(data.get("top_roles")),
            representative_memory_ids=_str_list(data.get("representative_memory_ids")),
            centroid_embedding_id=(
                str(data["centroid_embedding_id"])
                if data.get("centroid_embedding_id")
                else None
            ),
            centroid_vector_hash=(
                str(data["centroid_vector_hash"])
                if data.get("centroid_vector_hash")
                else None
            ),
            activation_score=float(data.get("activation_score") or 0.0),
            pressure_score=float(data.get("pressure_score") or 0.0),
            unresolved_score=float(data.get("unresolved_score") or 0.0),
            contradiction_count=int(data.get("contradiction_count") or 0),
            refinement_count=int(data.get("refinement_count") or 0),
            activation_streak=int(data.get("activation_streak") or 0),
            inactive_streak=int(data.get("inactive_streak") or 0),
            last_activated_at=_dt("last_activated_at"),
            last_activated_event_id=data.get("last_activated_event_id"),
            last_pressure_update_at=_dt("last_pressure_update_at"),
            last_resolution_event_id=data.get("last_resolution_event_id"),
            resolved_recently=bool(data.get("resolved_recently")),
            activation_reasons=_str_list(data.get("activation_reasons")),
            pressure_reasons=_str_list(data.get("pressure_reasons")),
            community_detection_strategy=str(
                data.get("community_detection_strategy")
                or "deterministic_connected_components_v0"
            ),
            created_at=_dt("created_at") or utcnow(),
            updated_at=_dt("updated_at") or utcnow(),
            metadata=_json_dict(data.get("metadata")),
        )
