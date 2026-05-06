from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullerene.nexus.models import Event, NexusState


@dataclass(slots=True)
class AdapterSignals:
    memory: dict[str, Any] | None
    goals: dict[str, Any] | None
    world_model: dict[str, Any] | None
    policy: dict[str, Any] | None
    planner: dict[str, Any] | None
    context: dict[str, Any]
    attention: dict[str, Any] | None
    lpb: dict[str, Any] | None


def extract_adapter_signals(event: Event, state: NexusState) -> AdapterSignals:
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    return AdapterSignals(
        memory=_first_dict(metadata, ("memory", "memory_facet", "stored_memory", "relevant_memory"))
        or _state_dict(state, "memory"),
        goals=_first_dict(metadata, ("goals", "goal_signal", "goals_facet")) or _state_dict(state, "goals"),
        world_model=_first_dict(metadata, ("world_model", "world_signal", "world_model_facet"))
        or _state_dict(state, "world_model"),
        policy=_first_dict(metadata, ("policy",)) or _state_dict(state, "policy"),
        planner=_state_dict(state, "planner"),
        context=_first_dict(metadata, ("context",))
        or _context_window_dict(metadata.get("context_window"))
        or (_state_dict(state, "context") or {}),
        attention=_state_dict(state, "attention"),
        lpb=_state_dict(state, "lpb"),
    )


def _state_dict(state: NexusState, key: str) -> dict[str, Any] | None:
    payload = state.facet_state.get(key)
    return payload if isinstance(payload, dict) else None


def _first_dict(metadata: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any] | None:
    for key in keys:
        payload = metadata.get(key)
        if isinstance(payload, dict):
            return payload
    return None


def _context_window_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return {"last_context_window": raw}
    return None
