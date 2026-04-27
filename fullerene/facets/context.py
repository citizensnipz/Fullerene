"""Static working-context facet for Fullerene Context v0."""

from __future__ import annotations

from pathlib import Path

from fullerene.context import (
    STATIC_RECENT_EPISODIC_V0,
    ContextWindow,
    StaticContextAssembler,
)
from fullerene.memory import MemoryStore, SQLiteMemoryStore
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState


class ContextFacet:
    """Assemble a small, deterministic working context from recent memory."""

    name = "context"

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        window_size: int = 5,
    ) -> None:
        self.store = store
        self.window_size = max(int(window_size), 1)
        self.assembler = StaticContextAssembler(store, max_items=self.window_size)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        window_size: int = 5,
    ) -> "ContextFacet":
        return cls(SQLiteMemoryStore(path), window_size=window_size)

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del event, state

        if self.store is None:
            window = ContextWindow(max_items=self.window_size)
            summary = (
                "Context facet returned an empty static context window because no "
                "memory store is configured."
            )
        else:
            window = self.assembler.assemble()
            summary = (
                f"Context facet assembled {len(window.items)} recent episodic "
                f"memories using {STATIC_RECENT_EPISODIC_V0}."
            )

        source_types = self._source_types(window)
        return FacetResult(
            facet_name=self.name,
            summary=summary,
            proposed_decision=DecisionAction.WAIT,
            state_updates={
                "last_context_window_id": window.id,
                "last_context_item_ids": [item.id for item in window.items],
                "last_context_item_count": len(window.items),
                "last_context_strategy": window.strategy,
            },
            metadata={
                "context_window": window.to_dict(),
                "item_count": len(window.items),
                "strategy": window.strategy,
                "max_items": window.max_items,
                "source_types": source_types,
            },
        )

    @staticmethod
    def _source_types(window: ContextWindow) -> list[str]:
        raw_source_types = window.metadata.get("source_types", [])
        if not isinstance(raw_source_types, list):
            return []
        return [str(source_type) for source_type in raw_source_types]
