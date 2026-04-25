"""Facet protocol for Nexus-compatible components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fullerene.nexus.models import Event, FacetResult, NexusState


@runtime_checkable
class Facet(Protocol):
    """Minimal facet contract for the Nexus runtime."""

    name: str

    def process(self, event: Event, state: NexusState) -> FacetResult:
        """Return a structured observation for the current event and state."""
