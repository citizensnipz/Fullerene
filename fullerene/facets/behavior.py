"""Compatibility orchestration wrapper for behavior facet."""

from __future__ import annotations

from fullerene.behavior.orchestrator import (
    HIGH_AMBIGUITY_THRESHOLD,
    LOW_AMBIGUITY_THRESHOLD,
    BehaviorFacet,
)

__all__ = [
    "BehaviorFacet",
    "HIGH_AMBIGUITY_THRESHOLD",
    "LOW_AMBIGUITY_THRESHOLD",
]
