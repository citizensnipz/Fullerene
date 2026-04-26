"""Public Fullerene runtime API."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BehaviorFacet",
    "Belief",
    "BeliefSource",
    "BeliefStatus",
    "DecisionAction",
    "EchoFacet",
    "Event",
    "EventType",
    "Facet",
    "FacetResult",
    "FileStateStore",
    "Goal",
    "GoalSource",
    "GoalStatus",
    "GoalsFacet",
    "GoalStore",
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
    "SQLiteGoalStore",
    "SQLiteMemoryStore",
    "SQLiteWorldModelStore",
    "StateStore",
    "WorldModelFacet",
    "WorldModelStore",
]


def __getattr__(name: str):
    if name in {"Facet"}:
        return getattr(import_module("fullerene.facets.base"), name)
    if name in {
        "BehaviorFacet",
        "EchoFacet",
        "GoalsFacet",
        "MemoryFacet",
        "WorldModelFacet",
    }:
        return getattr(import_module("fullerene.facets"), name)
    if name in {"Goal", "GoalSource", "GoalStatus", "GoalStore", "SQLiteGoalStore"}:
        return getattr(import_module("fullerene.goals"), name)
    if name in {"MemoryRecord", "MemoryStore", "MemoryType", "SQLiteMemoryStore"}:
        return getattr(import_module("fullerene.memory"), name)
    if name in {
        "Belief",
        "BeliefSource",
        "BeliefStatus",
        "SQLiteWorldModelStore",
        "WorldModelStore",
    }:
        return getattr(import_module("fullerene.world_model"), name)
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
