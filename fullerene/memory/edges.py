"""Memory edge models for Fullerene Memory v2.

Edges are computed at *write time* against a bounded candidate set so the
graph stays inspectable and avoids the cost of full graph traversal at
retrieval time. Memory v2 only writes edges; it does not yet use them for
retrieval. Memory v3 will build clustering, community detection, and richer
graph-aware retrieval on top of the same rows.

Edge types:

- ``same_goal`` - both memories carry the same ``goal_id`` in metadata.
- ``tag_overlap`` - tag intersection above a small threshold.
- ``temporal_proximity`` - memories created within a short configurable
  window of each other.
- ``keyword_similarity`` - token-overlap above a threshold.
- ``semantic_similarity`` - cosine similarity above a threshold (only when
  both records have stored embeddings).
- ``same_domain`` - both records share a non-empty domain bucket.
- ``role_related`` - role pairs that v2 considers complementary, e.g.
  ``question`` with ``preference`` in the same domain, or ``task`` with
  ``feedback`` / ``outcome``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from fullerene.memory.models import _parse_datetime, _serialize_value, utcnow


class MemoryEdgeType(str, Enum):
    SAME_GOAL = "same_goal"
    TAG_OVERLAP = "tag_overlap"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    KEYWORD_SIMILARITY = "keyword_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SAME_DOMAIN = "same_domain"
    ROLE_RELATED = "role_related"


@dataclass(slots=True)
class MemoryEdge:
    source_memory_id: str
    target_memory_id: str
    edge_type: MemoryEdgeType
    weight: float = 0.0
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_memory_id == self.target_memory_id:
            raise ValueError(
                "MemoryEdge cannot link a memory to itself"
            )
        weight = float(self.weight)
        if weight < 0.0:
            weight = 0.0
        if weight > 1.0:
            weight = 1.0
        self.weight = weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_memory_id": self.source_memory_id,
            "target_memory_id": self.target_memory_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        return cls(
            id=data["id"],
            source_memory_id=data["source_memory_id"],
            target_memory_id=data["target_memory_id"],
            edge_type=MemoryEdgeType(data["edge_type"]),
            weight=float(data.get("weight", 0.0)),
            created_at=_parse_datetime(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
