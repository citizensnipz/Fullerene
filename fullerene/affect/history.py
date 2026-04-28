"""Small bounded history helpers for Fullerene Affect v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fullerene.affect.models import AffectState


@dataclass(slots=True)
class AffectHistoryBuffer:
    max_size: int = 20
    states: list[AffectState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_size = max(int(self.max_size), 1)
        self.states = list(self.states or [])
        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size :]

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        max_size: int = 20,
    ) -> "AffectHistoryBuffer":
        states: list[AffectState] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                states.append(AffectState.from_dict(item))
        return cls(max_size=max_size, states=states)

    def append(self, state: AffectState) -> None:
        self.states.append(state)
        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size :]

    def to_dict(self) -> list[dict[str, Any]]:
        return [state.to_dict() for state in self.states]
