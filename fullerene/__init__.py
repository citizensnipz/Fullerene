"""Public Fullerene runtime API."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BehaviorFacet",
    "DecisionAction",
    "EchoFacet",
    "Event",
    "EventType",
    "Facet",
    "FacetResult",
    "FileStateStore",
    "InMemoryStateStore",
    "MemoryFacet",
    "MemoryRecord",
    "MemoryStore",
    "MemoryType",
    "Nexus",
    "NexusDecision",
    "NexusRecord",
    "NexusRuntime",
    "NexusState",
    "SQLiteMemoryStore",
    "StateStore",
]


def __getattr__(name: str):
    if name in {"Facet"}:
        return getattr(import_module("fullerene.facets.base"), name)
    if name in {"BehaviorFacet", "EchoFacet", "MemoryFacet"}:
        return getattr(import_module("fullerene.facets"), name)
    if name in {"MemoryRecord", "MemoryStore", "MemoryType", "SQLiteMemoryStore"}:
        return getattr(import_module("fullerene.memory"), name)
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
    if name in {"FileStateStore", "InMemoryStateStore", "StateStore"}:
        return getattr(import_module("fullerene.state.store"), name)
    raise AttributeError(f"module 'fullerene' has no attribute {name!r}")
