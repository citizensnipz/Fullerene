"""Deterministic working-context facet for Fullerene Context v0 and v1."""

from __future__ import annotations

from pathlib import Path

from fullerene.context import (
    DYNAMIC_ACTIVE_FACETS_V1,
    PRESSURE_RELEVANCE_V2,
    SELF_EDITING_V3,
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
    "pressure_relevance_v2": PRESSURE_RELEVANCE_V2,
    PRESSURE_RELEVANCE_V2: PRESSURE_RELEVANCE_V2,
    "self_editing_v3": SELF_EDITING_V3,
    SELF_EDITING_V3: SELF_EDITING_V3,
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
        self.config.strategy = self.strategy
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
        included_context_types = sorted(set(source_types))
        included_goal_ids = self._ids_from_metadata(window, key="included_goal_ids")
        deduped_goal_ids = self._ids_from_metadata(window, key="deduped_goal_ids")
        included_memory_ids = self._ids_from_metadata(window, key="included_memory_ids")
        included_belief_ids = self._ids_from_metadata(window, key="included_belief_ids")
        normalized_goal_keys = self._ids_from_metadata(window, key="normalized_goal_keys")
        included_memory_roles = self._ids_from_metadata(
            window, key="included_memory_roles"
        )
        included_memory_domains = self._ids_from_metadata(
            window, key="included_memory_domains"
        )
        retrieval_strategy = window.metadata.get("retrieval_strategy")
        query_intent = window.metadata.get("query_intent")
        event_domain = window.metadata.get("event_domain")
        memory_score_breakdowns = window.metadata.get("memory_score_breakdowns", [])
        wm_v2_cluster_ids = window.metadata.get("included_wm_v2_contradiction_cluster_ids")
        if not isinstance(wm_v2_cluster_ids, list):
            wm_v2_cluster_ids = []
        wm_v2_cluster_ids = [str(cid) for cid in wm_v2_cluster_ids if str(cid).strip()]
        belief_consistency_prior_wm = bool(
            window.metadata.get("belief_consistency_prior_wm", False)
        )
        reference_anchors = window.metadata.get("reference_anchors", [])
        unresolved_references = window.metadata.get("unresolved_references", [])
        continuity_confidence = float(window.metadata.get("continuity_confidence", 0.0) or 0.0)
        current_topic_hint = window.metadata.get("current_topic_hint")
        topic_terms = window.metadata.get("topic_terms", [])
        reference_anchor_count = int(window.metadata.get("reference_anchor_count", len(reference_anchors)) or 0)
        item_count = len(window.items)
        max_items = window.max_items
        load_ratio = round(item_count / max(float(max_items), 1.0), 3)
        context_load = {
            "item_count": item_count,
            "max_items": max_items,
            "load_ratio": load_ratio,
            "overloaded": load_ratio >= 0.85,
        }
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
                "last_context_item_count": item_count,
                "last_context_max_items": max_items,
                "last_context_load_ratio": load_ratio,
                "last_context_overloaded": context_load["overloaded"],
                "last_context_pressure": window.metadata.get("context_pressure", 0.0),
                "last_context_strategy": window.strategy,
                "last_context_source_types": source_types,
                "last_included_context_types": included_context_types,
                "last_included_goal_ids": included_goal_ids,
                "last_deduped_goal_ids": deduped_goal_ids,
                "last_included_memory_ids": included_memory_ids,
                "last_included_belief_ids": included_belief_ids,
                "last_normalized_goal_keys": normalized_goal_keys,
                "last_included_memory_roles": included_memory_roles,
                "last_included_memory_domains": included_memory_domains,
                "last_retrieval_strategy": retrieval_strategy,
                "last_query_intent": query_intent,
                "last_event_domain": event_domain,
                "last_reference_anchors": reference_anchors,
                "last_unresolved_references": unresolved_references,
                "last_continuity_confidence": continuity_confidence,
                "last_current_topic_hint": current_topic_hint,
                "last_topic_terms": topic_terms,
                "last_reference_anchor_count": reference_anchor_count,
                "last_included_wm_v2_contradiction_cluster_ids": wm_v2_cluster_ids,
                "last_included_belief_consistency_wm_v2": belief_consistency_prior_wm,
                "last_context_item_lifecycle": window.metadata.get("context_item_lifecycle", []),
                "last_context_learning_events": window.metadata.get("learning_events", []),
                "last_context_lpb_signals": window.metadata.get("lpb_signals", []),
            },
            metadata={
                "context_window": window.to_dict(),
                "item_count": item_count,
                "strategy": window.strategy,
                "max_items": max_items,
                "context_load": context_load,
                "source_types": source_types,
                "included_context_types": included_context_types,
                "included_goal_ids": included_goal_ids,
                "deduped_goal_count": window.metadata.get("deduped_goal_count", 0),
                "deduped_goal_ids": deduped_goal_ids,
                "normalized_goal_keys": normalized_goal_keys,
                "included_memory_ids": included_memory_ids,
                "included_belief_ids": included_belief_ids,
                "included_memory_roles": included_memory_roles,
                "included_memory_domains": included_memory_domains,
                "retrieval_strategy": retrieval_strategy,
                "query_intent": query_intent,
                "event_domain": event_domain,
                "memory_score_breakdowns": memory_score_breakdowns,
                "reference_anchors": reference_anchors,
                "reference_anchor_count": reference_anchor_count,
                "unresolved_references": unresolved_references,
                "continuity_confidence": continuity_confidence,
                "current_topic_hint": current_topic_hint,
                "topic_terms": topic_terms,
                "included_wm_v2_contradiction_cluster_ids": wm_v2_cluster_ids,
                "belief_consistency_prior_wm": belief_consistency_prior_wm,
                "context_pressure": window.metadata.get("context_pressure", 0.0),
                "context_pressure_components": window.metadata.get("context_pressure_components", {}),
                "context_pressure_reason": window.metadata.get("context_pressure_reason"),
                "context_overloaded": window.metadata.get("context_overloaded", context_load["overloaded"]),
                "consolidation_recommended": window.metadata.get("consolidation_recommended", False),
                "predictive_context_item_count": len(window.metadata.get("predictive_item_ids", [])),
                "protected_context_item_count": len(window.metadata.get("protected_context_item_ids", [])),
                "pruned_context_item_count": len(window.metadata.get("pruned_context_item_ids", [])),
                "stale_context_item_count": window.metadata.get("stale_context_item_count", 0),
                "active_memory_community_count": len(
                    [
                        item
                        for item in window.items
                        if item.item_type.value == "memory_community"
                    ]
                ),
                "active_belief_cluster_count": len(wm_v2_cluster_ids),
                "consolidated_context_items": window.metadata.get("consolidated_context_items", []),
                "predictive_context_items": window.metadata.get("predictive_context_items", []),
                "learning_events": window.metadata.get("learning_events", []),
                "lpb_signals": window.metadata.get("lpb_signals", []),
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
