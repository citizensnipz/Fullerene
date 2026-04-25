"""Nexus runtime exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DecisionAction",
    "Event",
    "EventType",
    "FacetResult",
    "Nexus",
    "NexusDecision",
    "NexusRecord",
    "NexusRuntime",
    "NexusState",
]


def __getattr__(name: str):
    if name in {
        "DecisionAction",
        "Event",
        "EventType",
        "FacetResult",
        "NexusDecision",
        "NexusRecord",
        "NexusState",
    }:
        return getattr(import_module("fullerene.nexus.models"), name)
    if name in {"Nexus", "NexusRuntime"}:
        return getattr(import_module("fullerene.nexus.runtime"), name)
    raise AttributeError(f"module 'fullerene.nexus' has no attribute {name!r}")
