"""Deterministic working-context facet for Fullerene Context v0 and v1."""

from __future__ import annotations

from pathlib import Path

from fullerene.context import (
    DYNAMIC_ACTIVE_FACETS_V1,
    STATIC_RECENT_EPISODIC_V0,
    ContextAssemblyConfig,
    ContextWindow,
    DynamicContextAssembler,
    StaticContextAssembler,
)
from fullerene.goals import GoalStore
from fullerene.memory import MemoryStore, SQLiteMemoryStore
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.policy import PolicyStore
from fullerene.world_model import WorldModelStore

CONTEXT_STRATEGIES = {
    "dynamic": DYNAMIC_ACTIVE_FACETS_V1,
    DYNAMIC_ACTIVE_FACETS_V1: DYNAMIC_ACTIVE_FACETS_V1,
    "static": STATIC_RECENT_EPISODIC_V0,
    STATIC_RECENT_EPISODIC_V0: STATIC_RECENT_EPISODIC_V0,
}


class ContextFacet:
    """Assemble a small, deterministic working context from recent memory."""

    name = "context"

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        policy_store: PolicyStore | None = None,
        window_size: int = 5,
        strategy: str | None = None,
        config: ContextAssemblyConfig | None = None,
    ) -> None:
        self.store = store
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.policy_store = policy_store
        self.window_size = max(int(window_size), 1)
        self.config = config or ContextAssemblyConfig(max_memories=self.window_size)
        self.strategy = self._resolve_strategy(strategy)
        self.static_assembler = StaticContextAssembler(
            store,
            max_items=self.window_size,
        )
        self.dynamic_assembler = DynamicContextAssembler(
            memory_store=store,
            goal_store=goal_store,
            world_model_store=world_model_store,
            policy_store=policy_store,
            config=self.config,
        )

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        window_size: int = 5,
        strategy: str | None = None,
        config: ContextAssemblyConfig | None = None,
    ) -> "ContextFacet":
        return cls(
            SQLiteMemoryStore(path),
            window_size=window_size,
            strategy=strategy,
            config=config,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        window = self._assemble_window(event, state)
        source_types = self._source_types(window)
        included_goal_ids = self._ids_from_metadata(window, key="included_goal_ids")
        deduped_goal_ids = self._ids_from_metadata(window, key="deduped_goal_ids")
        included_memory_ids = self._ids_from_metadata(window, key="included_memory_ids")
        included_belief_ids = self._ids_from_metadata(window, key="included_belief_ids")
        normalized_goal_keys = self._ids_from_metadata(window, key="normalized_goal_keys")
        proposed_decision = (
            DecisionAction.RECORD
            if self._has_meaningful_items(window)
            else DecisionAction.WAIT
        )
        summary = self._summary(window)
        return FacetResult(
            facet_name=self.name,
            summary=summary,
            proposed_decision=proposed_decision,
            state_updates={
                "last_context_window": window.to_dict(),
                "last_context_window_id": window.id,
                "last_context_item_ids": [item.id for item in window.items],
                "last_context_item_count": len(window.items),
                "last_context_strategy": window.strategy,
                "last_context_source_types": source_types,
                "last_included_goal_ids": included_goal_ids,
                "last_deduped_goal_ids": deduped_goal_ids,
                "last_included_memory_ids": included_memory_ids,
                "last_included_belief_ids": included_belief_ids,
                "last_normalized_goal_keys": normalized_goal_keys,
            },
            metadata={
                "context_window": window.to_dict(),
                "item_count": len(window.items),
                "strategy": window.strategy,
                "max_items": window.max_items,
                "source_types": source_types,
                "included_goal_ids": included_goal_ids,
                "deduped_goal_count": window.metadata.get("deduped_goal_count", 0),
                "deduped_goal_ids": deduped_goal_ids,
                "normalized_goal_keys": normalized_goal_keys,
                "included_memory_ids": included_memory_ids,
                "included_belief_ids": included_belief_ids,
                "salience_threshold": window.metadata.get("salience_threshold", 0.0),
                "limits": window.metadata.get("limits", {}),
                "reasons": window.metadata.get("reasons", []),
            },
        )

    def _assemble_window(self, event: Event, state: NexusState) -> ContextWindow:
        if self.strategy == STATIC_RECENT_EPISODIC_V0:
            if self.store is None:
                return ContextWindow(max_items=self.window_size)
            return self.static_assembler.assemble()
        return self.dynamic_assembler.assemble(event=event, state=state)

    def _summary(self, window: ContextWindow) -> str:
        if window.strategy == STATIC_RECENT_EPISODIC_V0:
            if not window.items:
                return (
                    "Context facet returned an empty static context window because no "
                    "recent episodic memories were available."
                )
            return (
                f"Context facet assembled {len(window.items)} recent episodic "
                f"memories using {STATIC_RECENT_EPISODIC_V0}."
            )
        return (
            f"Context facet assembled {len(window.items)} working-context item(s) "
            f"using {window.strategy}."
        )

    @staticmethod
    def _resolve_strategy(raw_strategy: str | None) -> str:
        if raw_strategy is None:
            return DYNAMIC_ACTIVE_FACETS_V1
        cleaned = str(raw_strategy).strip().lower()
        return CONTEXT_STRATEGIES.get(cleaned, DYNAMIC_ACTIVE_FACETS_V1)

    @staticmethod
    def _ids_from_metadata(window: ContextWindow, *, key: str) -> list[str]:
        raw_ids = window.metadata.get(key, [])
        if not isinstance(raw_ids, list):
            return []
        return [str(item_id) for item_id in raw_ids]

    @staticmethod
    def _source_types(window: ContextWindow) -> list[str]:
        raw_source_types = window.metadata.get("source_types", [])
        if not isinstance(raw_source_types, list):
            return []
        return [str(source_type) for source_type in raw_source_types]

    @staticmethod
    def _has_meaningful_items(window: ContextWindow) -> bool:
        for item in window.items:
            if item.content.strip():
                return True
            if item.metadata:
                return True
        return False
