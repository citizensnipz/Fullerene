"""Context assembly for Fullerene Context v0 and v1."""

from __future__ import annotations

from typing import Any, Sequence

from fullerene.attention import AttentionBroadcast
from fullerene.context.models import (
    PRESSURE_RELEVANCE_V2,
    STATIC_RECENT_EPISODIC_V0,
    ContextAssemblyConfig,
    ContextItem,
    ContextItemType,
    ContextWindow,
)
from fullerene.goals import GoalStore
from fullerene.goals.normalization import GoalDeduplicationResult, dedupe_active_goals
from fullerene.memory import (
    MemoryRecord,
    MemoryStore,
    MemoryType,
    classify_query_intent,
    extract_event_tags,
    infer_domain,
    tokenize,
)
from fullerene.nexus.models import Event, FacetResult, NexusState
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicyStore,
    PolicyTargetType,
)
from fullerene.world_model import Belief, WorldModelStore

EXTERNAL_POLICY_TARGETS = (
    PolicyTargetType.FILE_WRITE,
    PolicyTargetType.FILE_DELETE,
    PolicyTargetType.SHELL,
    PolicyTargetType.NETWORK,
    PolicyTargetType.MESSAGE,
    PolicyTargetType.GIT,
    PolicyTargetType.TOOL,
)

LPB_INCLUDE_TYPES = {
    "contradiction",
    "policy_block",
    "verifier_failure",
    "unresolved_query",
    "goal_block",
    "context_overload",
    "planner_conflict",
}


class StaticContextAssembler:
    """Build a small, deterministic context window from recent episodic memory."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        max_items: int = 5,
    ) -> None:
        self.store = store
        self.max_items = max(int(max_items), 1)

    def assemble(
        self,
        recent_records: Sequence[MemoryRecord] | None = None,
    ) -> ContextWindow:
        records = self._load_recent_records(recent_records)
        items = [self._memory_to_context_item(record) for record in records]
        included_memory_ids = [item.id for item in items]
        source_types = [ContextItemType.MEMORY.value] if items else []

        return ContextWindow(
            items=items,
            max_items=self.max_items,
            strategy=STATIC_RECENT_EPISODIC_V0,
            metadata={
                "source_types": source_types,
                "source_memory_type": MemoryType.EPISODIC.value,
                "item_count": len(items),
                "included_goal_ids": [],
                "included_memory_ids": included_memory_ids,
                "included_belief_ids": [],
                "salience_threshold": 0.0,
                "limits": {
                    "max_goals": 0,
                    "max_memories": self.max_items,
                    "max_beliefs": 0,
                },
                "reasons": (
                    ["static_recent_episodic_v0"]
                    if items
                    else ["no_recent_episodic_memories"]
                ),
            },
        )

    def _load_recent_records(
        self,
        recent_records: Sequence[MemoryRecord] | None,
    ) -> list[MemoryRecord]:
        if recent_records is not None:
            episodic_records = [
                record
                for record in recent_records
                if record.memory_type == MemoryType.EPISODIC
            ]
            return list(episodic_records[: self.max_items])
        if self.store is None:
            return []
        return self.store.list_recent(
            limit=self.max_items,
            memory_type=MemoryType.EPISODIC,
        )

    @staticmethod
    def _memory_to_context_item(record: MemoryRecord) -> ContextItem:
        return ContextItem(
            id=record.id,
            item_type=ContextItemType.MEMORY,
            content=record.content,
            source_id=record.source_event_id,
            created_at=record.created_at,
            metadata={
                "memory_type": record.memory_type.value,
                "salience": record.salience,
                "confidence": record.confidence,
                "tags": list(record.tags),
                "memory_metadata": dict(record.metadata),
                "role": record.role,
                "domain": record.domain,
            },
        )


class DynamicContextAssembler:
    """Build a bounded working packet from the current event and active state."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        policy_store: PolicyStore | None = None,
        config: ContextAssemblyConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.policy_store = policy_store
        self.config = config or ContextAssemblyConfig()

    def assemble(
        self,
        *,
        event: Event,
        state: NexusState | None = None,
        facet_results: Sequence[FacetResult] | None = None,
    ) -> ContextWindow:
        if self.config.strategy == PRESSURE_RELEVANCE_V2:
            return self._assemble_pressure_relevance_v2(
                event=event,
                state=state,
                facet_results=facet_results,
            )
        working_state = state or NexusState()
        items: list[ContextItem] = []
        reasons: list[str] = ["included_current_event"]

        event_item = self._event_item(event)
        items.append(event_item)

        attention_item = self._attention_context_item(
            state=working_state,
            facet_results=facet_results,
            existing_items=items,
        )
        if attention_item is not None:
            items.append(attention_item)
            reasons.append("included_attention_broadcast")
        else:
            reasons.append("attention_broadcast_unavailable_or_duplicate")

        working_memory_items, working_memory_meta = self._working_memory_items(event)
        items.extend(working_memory_items)
        reasons.append(f"included_working_memory_turns={len(working_memory_items)}")

        goal_items, goal_deduplication = self._goal_items()
        items.extend(goal_items)
        reasons.append(f"included_goals={len(goal_items)}")
        if goal_deduplication.deduped_goal_count > 0:
            reasons.append(f"deduped_goals={goal_deduplication.deduped_goal_count}")

        memory_assembly = self._memory_items(event)
        relevant_memory_items = memory_assembly["relevant_items"]
        recent_memory_items = memory_assembly["recent_items"]
        items.extend(relevant_memory_items)
        items.extend(recent_memory_items)
        reasons.append(f"included_relevant_memories={len(relevant_memory_items)}")
        reasons.append(f"included_recent_memories={len(recent_memory_items)}")

        belief_items = self._belief_items(event)
        items.extend(belief_items)
        reasons.append(f"included_beliefs={len(belief_items)}")

        policy_item = self._policy_item()
        if policy_item is not None:
            items.append(policy_item)
            reasons.append("included_policy_summary")
        elif self.config.include_policy_summary:
            reasons.append("policy_summary_unavailable")

        signal_items = self._signal_items(
            state=working_state,
            facet_results=facet_results,
        )
        items.extend(signal_items)
        reasons.append(f"included_signal_summaries={len(signal_items)}")

        memory_items = [*relevant_memory_items, *recent_memory_items]
        included_memory_roles = sorted(
            {
                str(item.metadata.get("role"))
                for item in memory_items
                if isinstance(item.metadata, dict) and item.metadata.get("role")
            }
        )
        included_memory_domains = sorted(
            {
                str(item.metadata.get("domain"))
                for item in memory_items
                if isinstance(item.metadata, dict) and item.metadata.get("domain")
            }
        )
        memory_score_breakdowns = [
            {
                "memory_id": item.id,
                "role": item.metadata.get("role") if isinstance(item.metadata, dict) else None,
                "domain": item.metadata.get("domain") if isinstance(item.metadata, dict) else None,
                "context_source": item.metadata.get("context_source")
                if isinstance(item.metadata, dict)
                else None,
                "hybrid_score": item.metadata.get("hybrid_score")
                if isinstance(item.metadata, dict)
                else None,
                "score_breakdown": item.metadata.get("score_breakdown")
                if isinstance(item.metadata, dict)
                else None,
            }
            for item in memory_items
        ]

        metadata = {
            "source_types": self._source_types(items),
            "item_count": len(items),
            "included_attention_item_ids": (
                [attention_item.metadata["attention_item_id"]]
                if attention_item is not None
                else []
            ),
            "working_memory_session_id": working_memory_meta["session_id"],
            "working_memory_turn_count": len(working_memory_items),
            "included_working_memory_turns": [item.id for item in working_memory_items],
            "included_goal_ids": [item.id for item in goal_items],
            "deduped_goal_count": goal_deduplication.deduped_goal_count,
            "deduped_goal_ids": list(goal_deduplication.deduped_goal_ids),
            "normalized_goal_keys": list(goal_deduplication.normalized_goal_keys),
            "included_memory_ids": [item.id for item in memory_items],
            "included_belief_ids": [item.id for item in belief_items],
            "salience_threshold": self.config.salience_threshold,
            "limits": {
                "max_goals": self.config.max_goals,
                "max_memories": self.config.max_memories,
                "max_beliefs": self.config.max_beliefs,
                "max_working_turns": self.config.max_working_turns,
            },
            "config": self.config.to_dict(),
            "reasons": reasons,
            "retrieval_strategy": memory_assembly["retrieval_strategy"],
            "query_intent": memory_assembly["query_intent"],
            "event_domain": memory_assembly["event_domain"],
            "included_memory_roles": included_memory_roles,
            "included_memory_domains": included_memory_domains,
            "memory_score_breakdowns": memory_score_breakdowns,
            "context_load": {
                "item_count": len(items),
                "max_items": self.config.max_items,
                "load_ratio": round(
                    len(items) / max(float(self.config.max_items), 1.0),
                    3,
                ),
                "overloaded": (
                    len(items) / max(float(self.config.max_items), 1.0)
                )
                >= 0.85,
            },
        }
        return ContextWindow(
            items=items,
            max_items=self.config.max_items,
            strategy=self.config.strategy,
            metadata=metadata,
        )

    def _assemble_pressure_relevance_v2(
        self,
        *,
        event: Event,
        state: NexusState | None = None,
        facet_results: Sequence[FacetResult] | None = None,
    ) -> ContextWindow:
        working_state = state or NexusState()
        included: list[ContextItem] = []
        included_ids: set[str] = set()
        protected_ids: set[str] = set()
        reasons: list[str] = []
        score_breakdowns: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []

        event_item = self._event_item(event)
        event_item.metadata["include_reason"] = "current_event_always_included"
        event_item.metadata["protected_inclusion"] = True
        event_item.metadata["final_score"] = 1.0
        included.append(event_item)
        included_ids.add(event_item.id)
        protected_ids.add(event_item.id)
        reasons.append("included_current_event")

        if self.config.include_working_memory:
            wm_items, wm_meta = self._working_memory_items(event)
            for item in wm_items[: self.config.max_working_memory_turns]:
                item.metadata["working_memory_protected"] = True
                item.metadata["include_reason"] = "recent_working_memory_protected"
                item.metadata["protected_inclusion"] = True
                item.metadata["final_score"] = 1.0
                if item.id in included_ids:
                    continue
                included.append(item)
                included_ids.add(item.id)
                protected_ids.add(item.id)
            reasons.append(f"included_working_memory_turns={len(wm_items)}")
        else:
            wm_meta = {"session_id": None}

        candidates: list[dict[str, Any]] = []
        candidates.extend(self._build_lpb_candidates(working_state, facet_results))
        candidates.extend(self._build_attention_candidates(working_state, facet_results, included))
        candidates.extend(self._build_goal_candidates(event))
        candidates.extend(self._build_belief_candidates(event))
        candidates.extend(self._build_memory_candidates_v2(event))
        candidates.extend(self._build_policy_candidates(working_state, facet_results))
        if self.config.include_recent_signals:
            candidates.extend(self._build_signal_candidates(working_state, facet_results))

        by_id: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate["item"].id)
            if candidate_id in by_id:
                excluded.append({"id": candidate_id, "reason": "duplicate"})
                continue
            by_id[candidate_id] = candidate

        ranked = sorted(
            by_id.values(),
            key=lambda row: (
                float(row["score"]["final_score"]),
                float(row["score"]["priority_score"]),
                float(row["score"]["recency_score"]),
                row["source_rank"],
            ),
            reverse=True,
        )
        budget = max(self.config.max_items_total, 1)
        for row in ranked:
            item = row["item"]
            if item.id in included_ids:
                excluded.append({"id": item.id, "reason": "duplicate"})
                continue
            if len(included) >= budget:
                excluded.append({"id": item.id, "reason": "budget_evicted"})
                continue
            score = row["score"]
            pressure_bypass = score["pressure_score"] >= self.config.min_pressure_score and row["source_type"] in {"lpb", "policy", "signal"}
            if not pressure_bypass and score["final_score"] < self.config.min_relevance_score:
                excluded.append({"id": item.id, "reason": "below_relevance_cutoff"})
                continue
            if row["source_type"] == "lpb" and score["pressure_score"] < self.config.min_pressure_score:
                excluded.append({"id": item.id, "reason": "low_pressure"})
                continue
            item.metadata["final_score"] = score["final_score"]
            item.metadata["score_breakdown"] = score
            item.metadata["include_reason"] = row["include_reason"]
            item.metadata["protected_inclusion"] = False
            included.append(item)
            included_ids.add(item.id)
            score_breakdowns.append(
                {
                    "item_id": item.id,
                    "source_type": row["source_type"],
                    "score": score,
                }
            )

        prior_ids = self._prior_context_item_ids(working_state)
        stale_ids = [item_id for item_id in prior_ids if item_id not in included_ids]
        for stale_id in stale_ids:
            excluded.append({"id": stale_id, "reason": "stale"})

        metadata = {
            "context_strategy": PRESSURE_RELEVANCE_V2,
            "source_types": self._source_types(included),
            "included_context_types": self._source_types(included),
            "included_item_ids": [item.id for item in included],
            "included_working_memory_turns": [item.id for item in included if item.item_type == ContextItemType.WORKING_MEMORY],
            "working_memory_session_id": wm_meta.get("session_id"),
            "working_memory_turn_count": len([item for item in included if item.item_type == ContextItemType.WORKING_MEMORY]),
            "included_lpb_entry_ids": [item.id for item in included if item.item_type == ContextItemType.SIGNAL and str(item.id).startswith("lpb:")],
            "included_attention_ids": [item.id for item in included if item.item_type == ContextItemType.ATTENTION],
            "included_memory_ids": [item.id for item in included if item.item_type == ContextItemType.MEMORY],
            "included_goal_ids": [item.id for item in included if item.item_type == ContextItemType.GOAL],
            "included_belief_ids": [item.id for item in included if item.item_type == ContextItemType.BELIEF],
            "included_policy_ids": [item.id for item in included if item.item_type == ContextItemType.POLICY],
            "excluded_context_items": excluded[:64],
            "item_score_breakdowns": score_breakdowns,
            "context_budget": budget,
            "budget_used": len(included),
            "cutoff_settings": {
                "min_relevance_score": self.config.min_relevance_score,
                "min_pressure_score": self.config.min_pressure_score,
            },
            "stale_context_item_count": len(stale_ids),
            "evicted_context_item_count": len([row for row in excluded if row.get("reason") == "budget_evicted"]),
            "reasons": reasons,
            "context_load": {
                "item_count": len(included),
                "max_items": budget,
                "load_ratio": round(len(included) / max(float(budget), 1.0), 3),
                "overloaded": len(included) / max(float(budget), 1.0) >= 0.85,
            },
            "config": self.config.to_dict(),
        }
        return ContextWindow(
            items=included,
            max_items=budget,
            strategy=PRESSURE_RELEVANCE_V2,
            metadata=metadata,
        )

    def _prior_context_item_ids(self, state: NexusState) -> list[str]:
        context_state = state.facet_state.get("context")
        if not isinstance(context_state, dict):
            return []
        raw_ids = context_state.get("last_context_item_ids")
        if not isinstance(raw_ids, list):
            return []
        return [str(item_id) for item_id in raw_ids]

    def _score_candidate(
        self,
        *,
        relevance_score: float,
        pressure_score: float = 0.0,
        salience_score: float = 0.0,
        recency_score: float = 0.0,
        confidence_score: float = 0.0,
        priority_score: float = 0.0,
    ) -> dict[str, float]:
        def c01(value: Any) -> float:
            try:
                return max(0.0, min(float(value), 1.0))
            except (TypeError, ValueError):
                return 0.0

        rel = c01(relevance_score)
        prs = c01(pressure_score)
        sal = c01(salience_score)
        rec = c01(recency_score)
        con = c01(confidence_score)
        pri = c01(priority_score)
        final = c01((rel * 0.35) + (prs * 0.25) + (sal * 0.15) + (rec * 0.10) + (con * 0.05) + (pri * 0.10))
        return {
            "relevance_score": rel,
            "pressure_score": prs,
            "salience_score": sal,
            "recency_score": rec,
            "confidence_score": con,
            "priority_score": pri,
            "final_score": final,
        }

    def _build_lpb_candidates(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> list[dict[str, Any]]:
        if not self.config.include_lpb:
            return []
        signals = state.facet_state.get("signals")
        latent = signals.get("latent_pressure") if isinstance(signals, dict) else None
        result = latent.get("last_result") if isinstance(latent, dict) else None
        top_entries = result.get("top_entries", []) if isinstance(result, dict) else []
        rows: list[dict[str, Any]] = []
        for entry in top_entries[: self.config.max_lpb_items]:
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("entry_type") or "")
            intensity = float(entry.get("intensity") or 0.0)
            if entry_type not in LPB_INCLUDE_TYPES:
                continue
            score = self._score_candidate(
                relevance_score=0.4,
                pressure_score=intensity,
                salience_score=min(float(entry.get("retrigger_count") or 0) * 0.2, 1.0),
                recency_score=1.0,
                confidence_score=0.8,
                priority_score=0.9,
            )
            item = ContextItem(
                id=f"lpb:{entry.get('id')}",
                item_type=ContextItemType.SIGNAL,
                content=str(entry.get("description") or entry_type),
                source_id=str(entry.get("id") or ""),
                metadata={
                    "entry_type": entry_type,
                    "intensity": intensity,
                    "retrigger_count": entry.get("retrigger_count", 0),
                    "status": entry.get("status"),
                    "source": entry.get("source"),
                    "reason": "high_latent_pressure" if intensity >= self.config.min_pressure_score else "low_pressure",
                },
            )
            rows.append(
                {
                    "item": item,
                    "score": score,
                    "source_rank": 7,
                    "source_type": "lpb",
                    "include_reason": "high_latent_pressure" if intensity >= self.config.min_pressure_score else "lpb_ignition",
                }
            )
        return rows

    def _build_attention_candidates(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
        existing_items: Sequence[ContextItem],
    ) -> list[dict[str, Any]]:
        if not self.config.include_attention:
            return []
        attention_item = self._attention_context_item(
            state=state,
            facet_results=facet_results,
            existing_items=existing_items,
        )
        if attention_item is None:
            return []
        score = self._score_candidate(
            relevance_score=0.65,
            pressure_score=float(attention_item.metadata.get("attention_pressure_contribution", 0.0)),
            salience_score=float(attention_item.metadata.get("attention_score", 0.0)),
            recency_score=1.0,
            confidence_score=0.8,
            priority_score=0.75,
        )
        return [
            {
                "item": attention_item,
                "score": score,
                "source_rank": 6,
                "source_type": "attention",
                "include_reason": "attention_broadcast",
            }
        ]

    def _build_goal_candidates(self, event: Event) -> list[dict[str, Any]]:
        if not self.config.include_goals:
            return []
        items, _dedupe = self._goal_items()
        out: list[dict[str, Any]] = []
        for item in items[: self.config.max_goals]:
            priority = float(item.metadata.get("priority", 0.0))
            reinforcement = float(item.metadata.get("reinforcement_score", 0.0))
            pressure = float(item.metadata.get("goal_pressure_contribution", 0.0))
            relevance = min(1.0, (priority * 0.75) + (reinforcement * 0.25))
            score = self._score_candidate(
                relevance_score=relevance,
                pressure_score=pressure,
                salience_score=max(priority, reinforcement),
                recency_score=0.7,
                confidence_score=0.9,
                priority_score=priority,
            )
            out.append({"item": item, "score": score, "source_rank": 5, "source_type": "goal", "include_reason": "active_goal"})
        return out

    def _build_belief_candidates(self, event: Event) -> list[dict[str, Any]]:
        if not self.config.include_world_model:
            return []
        out: list[dict[str, Any]] = []
        for item in self._belief_items(event)[: self.config.max_beliefs]:
            conf = float(item.metadata.get("confidence", 0.0))
            status = str(item.metadata.get("status") or "")
            score = self._score_candidate(
                relevance_score=conf if status != "contradicted" else 0.7,
                pressure_score=0.5 if status == "contradicted" else 0.0,
                salience_score=0.6 if status == "contradicted" else conf,
                recency_score=0.7,
                confidence_score=conf,
                priority_score=0.55,
            )
            out.append({"item": item, "score": score, "source_rank": 4, "source_type": "belief", "include_reason": "relevant_belief"})
        return out

    def _build_memory_candidates_v2(self, event: Event) -> list[dict[str, Any]]:
        assembly = self._memory_items(event)
        out: list[dict[str, Any]] = []
        for item in [*assembly["relevant_items"], *assembly["recent_items"]][: self.config.max_long_term_memories]:
            breakdown = item.metadata.get("score_breakdown") if isinstance(item.metadata, dict) else {}
            relevance = float((breakdown or {}).get("total", item.metadata.get("hybrid_score", 0.0) if isinstance(item.metadata, dict) else 0.0))
            score = self._score_candidate(
                relevance_score=relevance,
                salience_score=float(item.metadata.get("salience", 0.0) if isinstance(item.metadata, dict) else 0.0),
                recency_score=0.6 if item.metadata.get("context_source") == "recent" else 0.4,
                confidence_score=float(item.metadata.get("confidence", 0.0) if isinstance(item.metadata, dict) else 0.0),
                priority_score=0.45,
            )
            item.metadata["cluster_id"] = None
            item.metadata["cluster_activation_score"] = 0.0
            out.append({"item": item, "score": score, "source_rank": 3, "source_type": "memory", "include_reason": "relevant_long_term_memory"})
        return out

    def _build_policy_candidates(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> list[dict[str, Any]]:
        if not self.config.include_policy:
            return []
        policy_item = self._policy_item()
        if policy_item is None:
            return []
        signal_map = self._facet_bucket("nexus", state, facet_results) or {}
        policy_pressure = 1.0 if bool(signal_map.get("policy_blocks_act") or signal_map.get("policy_requires_approval")) else 0.0
        score = self._score_candidate(
            relevance_score=0.7,
            pressure_score=policy_pressure,
            salience_score=0.7,
            recency_score=0.9,
            confidence_score=0.95,
            priority_score=0.85,
        )
        return [{"item": policy_item, "score": score, "source_rank": 8, "source_type": "policy", "include_reason": "policy_constraint"}]

    def _build_signal_candidates(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._signal_items(state=state, facet_results=facet_results):
            score = self._score_candidate(
                relevance_score=0.3,
                salience_score=0.4,
                recency_score=0.8,
                confidence_score=0.7,
                priority_score=0.2,
            )
            out.append({"item": item, "score": score, "source_rank": 2, "source_type": "signal", "include_reason": "recent_system_signal"})
        return out

    def _working_memory_items(self, event: Event) -> tuple[list[ContextItem], dict[str, Any]]:
        if self.memory_store is None or self.config.max_working_turns == 0:
            return [], {"session_id": None}
        session_id = str(event.metadata.get("session_id") or "").strip()
        if not session_id or not hasattr(self.memory_store, "list_working_turns"):
            return [], {"session_id": None}
        records = self.memory_store.list_working_turns(
            session_id=session_id,
            limit=self.config.max_working_turns,
        )
        items = [
            ContextItem(
                id=record.id,
                item_type=ContextItemType.WORKING_MEMORY,
                content=record.content,
                source_id=record.source_event_id,
                created_at=record.created_at,
                metadata={
                    "context_source": "recent_working_memory",
                    "dialogue_role": record.metadata.get("dialogue_role"),
                    "turn_index": record.metadata.get("turn_index"),
                    "session_id": session_id,
                    "reason": "recent_working_memory",
                },
            )
            for record in records
        ]
        return items, {"session_id": session_id}

    def _event_item(self, event: Event) -> ContextItem:
        return ContextItem(
            id=event.event_id,
            item_type=ContextItemType.EVENT,
            content=event.content,
            source_id=event.event_id,
            created_at=event.timestamp,
            metadata={
                "context_source": "current_event",
                "event_type": event.event_type.value,
                "event_metadata": dict(event.metadata),
            },
        )

    def _attention_context_item(
        self,
        *,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
        existing_items: Sequence[ContextItem],
    ) -> ContextItem | None:
        attention_state = self._facet_bucket("attention", state, facet_results)
        if not attention_state:
            return None
        raw_broadcast = attention_state.get("last_attention_broadcast")
        if not isinstance(raw_broadcast, dict):
            return None
        try:
            broadcast = AttentionBroadcast.from_dict(raw_broadcast)
        except (KeyError, TypeError, ValueError):
            return None
        if self._is_duplicate_attention_source_item(existing_items, broadcast):
            return None
        return ContextItem(
            id=f"attention:{broadcast.id}",
            item_type=ContextItemType.ATTENTION,
            content=broadcast.content,
            source_id=broadcast.item_id,
            created_at=broadcast.created_at,
            metadata={
                "context_source": "attention_broadcast",
                "attention_broadcast_id": broadcast.id,
                "attention_item_id": broadcast.item_id,
                "attention_source": broadcast.source.value,
                "attention_score": broadcast.score,
                "attention_mode": broadcast.mode.value,
                "attention_source_id": broadcast.source_id,
                "attention_recipients": list(broadcast.recipients),
                "attention_conflict_ids": list(broadcast.conflict_ids),
                "attention_repeated_count": broadcast.repeated_count,
                "attention_pressure_contribution": broadcast.pressure_contribution,
            },
        )

    def _goal_items(self) -> tuple[list[ContextItem], GoalDeduplicationResult]:
        if self.goal_store is None or self.config.max_goals == 0:
            return [], dedupe_active_goals([], limit=0)
        goal_fetch_limit = max(self.config.max_goals * 5, self.config.max_goals, 10)
        goals = self.goal_store.list_active_goals(limit=goal_fetch_limit)
        deduped_goals = dedupe_active_goals(goals, limit=self.config.max_goals)
        return (
            [
                ContextItem(
                    id=goal.id,
                    item_type=ContextItemType.GOAL,
                    content=goal.description,
                    source_id=goal.id,
                    created_at=goal.updated_at,
                    metadata={
                        "context_source": "active_goal",
                        "priority": goal.priority,
                        "status": goal.status.value,
                        "tags": list(goal.tags),
                        "source": goal.source.value,
                        "reinforcement_score": goal.reinforcement_score,
                        "activation_count": goal.activation_count,
                        "last_activated_at": goal.last_activated_at.isoformat()
                        if goal.last_activated_at
                        else None,
                        "last_activated_event_id": goal.last_activated_event_id,
                        "completion_score": goal.completion_score,
                        "blocked_reason": goal.blocked_reason,
                        "stale_score": goal.stale_score,
                        "goal_score_breakdown": {
                            "priority_component": round(goal.priority * 0.45, 4),
                            "relevance_component": 0.0,
                            "reinforcement_component": round(
                                goal.reinforcement_score * 0.15, 4
                            ),
                            "recency_component": 0.0,
                            "final_score": round(
                                min(
                                    1.0,
                                    (goal.priority * 0.45)
                                    + (goal.reinforcement_score * 0.15),
                                ),
                                4,
                            ),
                        },
                        "goal_pressure_contribution": min(
                            1.0,
                            (goal.priority * (1.0 - goal.completion_score) * 0.5)
                            + (goal.reinforcement_score * 0.25)
                            + (0.25 if goal.blocked_reason else 0.0),
                        ),
                        "goal_metadata": dict(goal.metadata),
                    },
                )
                for goal in deduped_goals.goals
            ],
            deduped_goals,
        )

    def _memory_items(self, event: Event) -> dict[str, Any]:
        if self.memory_store is None or self.config.max_memories == 0:
            return {
                "relevant_items": [],
                "recent_items": [],
                "retrieval_strategy": "memory_disabled",
                "query_intent": classify_query_intent(event.content).value,
                "event_domain": None,
            }

        intent_value = classify_query_intent(event.content).value
        event_domain = infer_domain(event.content, extract_event_tags(event))
        breakdowns_by_id: dict[str, dict[str, Any]] = {}
        retrieval_strategy = "deterministic_v1_fallback"
        relevant_records: list[MemoryRecord] = []

        store = self.memory_store
        if hasattr(store, "hybrid_retrieve_relevant"):
            try:
                ranked_pairs = store.hybrid_retrieve_relevant(
                    event,
                    limit=self.config.max_memories,
                    domain_hint=event_domain,
                )
            except Exception:  # noqa: BLE001 - hybrid retrieval is opt-in
                ranked_pairs = []
            else:
                retrieval_strategy = "hybrid_v2_deterministic"
                for record, breakdown in ranked_pairs:
                    relevant_records.append(record)
                    breakdowns_by_id[record.id] = breakdown

        if not relevant_records:
            relevant_records = list(
                store.retrieve_relevant(event, limit=self.config.max_memories)
            )

        relevant_records = self._filter_memory_records(relevant_records)
        recent_records = self._filter_memory_records(
            store.list_recent(
                limit=self.config.max_memories,
                memory_type=MemoryType.EPISODIC,
            )
        )

        remaining = self.config.max_memories
        deduped_relevant: list[MemoryRecord] = []
        seen_memory_ids: set[str] = set()
        for record in relevant_records:
            if record.id in seen_memory_ids:
                continue
            deduped_relevant.append(record)
            seen_memory_ids.add(record.id)
            remaining -= 1
            if remaining <= 0:
                break

        deduped_recent: list[MemoryRecord] = []
        if remaining > 0:
            for record in recent_records:
                if record.id in seen_memory_ids:
                    continue
                deduped_recent.append(record)
                seen_memory_ids.add(record.id)
                remaining -= 1
                if remaining <= 0:
                    break

        relevant_items = [
            self._memory_to_context_item(
                record,
                context_source="relevant",
                breakdown=breakdowns_by_id.get(record.id),
            )
            for record in deduped_relevant
        ]
        recent_items = [
            self._memory_to_context_item(
                record,
                context_source="recent",
                breakdown=breakdowns_by_id.get(record.id),
            )
            for record in deduped_recent
        ]
        return {
            "relevant_items": relevant_items,
            "recent_items": recent_items,
            "retrieval_strategy": retrieval_strategy,
            "query_intent": intent_value,
            "event_domain": event_domain,
        }

    def _filter_memory_records(
        self,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        filtered: list[MemoryRecord] = []
        for record in records:
            if record.memory_type != MemoryType.EPISODIC:
                continue
            if record.salience < self.config.salience_threshold:
                continue
            filtered.append(record)
        return filtered

    def _belief_items(self, event: Event) -> list[ContextItem]:
        if self.world_model_store is None or self.config.max_beliefs == 0:
            return []
        beliefs = self.world_model_store.list_beliefs(
            limit=max(self.config.max_beliefs * 2, self.config.max_beliefs)
        )
        ranked_beliefs = sorted(
            beliefs,
            key=lambda belief: self._belief_sort_key(event, belief),
            reverse=True,
        )[: self.config.max_beliefs]
        return [
            ContextItem(
                id=belief.id,
                item_type=ContextItemType.BELIEF,
                content=belief.claim,
                source_id=belief.id,
                created_at=belief.updated_at,
                metadata={
                    "context_source": "active_belief",
                    "confidence": belief.confidence,
                    "status": belief.status.value,
                    "contradiction_count": getattr(belief, "contradiction_count", 0),
                    "support_count": getattr(belief, "support_count", 0),
                    "tags": list(belief.tags),
                    "source": belief.source.value,
                    "belief_metadata": dict(belief.metadata),
                },
            )
            for belief in ranked_beliefs
        ]

    def _belief_sort_key(
        self,
        event: Event,
        belief: Belief,
    ) -> tuple[float, float, float, str]:
        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        belief_tags = set(belief.tags)
        belief_keywords = tokenize(belief.claim)
        shared_tags = event_tags & belief_tags
        shared_keywords = event_keywords & belief_keywords
        tag_overlap = len(shared_tags) / len(belief_tags) if belief_tags else 0.0
        keyword_overlap = (
            len(shared_keywords) / len(belief_keywords) if belief_keywords else 0.0
        )
        relevance_score = round(tag_overlap + keyword_overlap + belief.confidence, 3)
        return (
            1.0 if shared_tags or shared_keywords else 0.0,
            relevance_score,
            belief.confidence,
            belief.id,
        )

    def _policy_item(self) -> ContextItem | None:
        if not self.config.include_policy_summary or self.policy_store is None:
            return None
        enabled_policies = self._list_enabled_policies(limit=20)
        enabled_policy_count = self._count_enabled_policies(enabled_policies)
        major_constraints = self._major_policy_constraints(enabled_policies)
        default_constraint = "External side effects require approval by default."
        if default_constraint not in major_constraints:
            major_constraints.insert(0, default_constraint)
        content = (
            f"Enabled policies: {enabled_policy_count}. "
            f"Constraints: {'; '.join(major_constraints[:4])}"
        )
        return ContextItem(
            id="policy-summary",
            item_type=ContextItemType.POLICY,
            content=content,
            source_id="policy-store",
            metadata={
                "context_source": "policy_summary",
                "enabled_policy_count": enabled_policy_count,
                "major_constraints": major_constraints[:4],
            },
        )

    def _signal_items(
        self,
        *,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> list[ContextItem]:
        if not self.config.include_signal_summaries:
            return []
        items: list[ContextItem] = []
        planner_item = self._planner_signal_item(state, facet_results)
        if planner_item is not None:
            items.append(planner_item)
        executor_item = self._executor_signal_item(state, facet_results)
        if executor_item is not None:
            items.append(executor_item)
        attention_item = self._attention_signal_item(state, facet_results)
        if attention_item is not None:
            items.append(attention_item)
        affect_item = self._affect_signal_item(state, facet_results)
        if affect_item is not None:
            items.append(affect_item)
        learning_item = self._learning_signal_item(state, facet_results)
        if learning_item is not None:
            items.append(learning_item)
        return items

    def _planner_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        planner_state = self._facet_bucket("planner", state, facet_results)
        if not planner_state:
            return None
        last_plan = planner_state.get("last_plan")
        if not isinstance(last_plan, dict):
            return None
        raw_steps = last_plan.get("steps")
        step_count = len(raw_steps) if isinstance(raw_steps, list) else 0
        confidence = self._numeric_value(planner_state.get("last_plan_confidence"))
        trigger_reason = (
            self._coerce_string(planner_state.get("last_trigger_reason")) or "none"
        )
        content = (
            f"Planner: {step_count} step(s), confidence {confidence:.2f}, "
            f"trigger {trigger_reason}."
        )
        return ContextItem(
            id="signal-planner",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="planner",
            metadata={"signal_type": "planner"},
        )

    def _executor_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        executor_state = self._facet_bucket("executor", state, facet_results)
        if not executor_state:
            return None
        execution_result = executor_state.get("last_execution_result")
        if not isinstance(execution_result, dict):
            return None
        overall_status = (
            self._coerce_string(execution_result.get("overall_status")) or "unknown"
        )
        dry_run = bool(execution_result.get("dry_run", True))
        content = f"Executor: {overall_status}, {'dry-run' if dry_run else 'live'}."
        return ContextItem(
            id="signal-executor",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="executor",
            metadata={"signal_type": "executor"},
        )

    def _attention_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        attention_state = self._facet_bucket("attention", state, facet_results)
        if not attention_state:
            return None
        dominant_source = self._coerce_string(attention_state.get("last_dominant_source"))
        raw_focus_ids = attention_state.get("last_focus_item_ids")
        focus_count = len(raw_focus_ids) if isinstance(raw_focus_ids, list) else 0
        raw_broadcast = attention_state.get("last_attention_broadcast")
        broadcast_mode = None
        broadcast_source = None
        if isinstance(raw_broadcast, dict):
            broadcast_mode = self._coerce_string(raw_broadcast.get("mode"))
            broadcast_source = self._coerce_string(raw_broadcast.get("source"))
        if dominant_source is None and focus_count == 0 and broadcast_mode is None:
            return None
        content = (
            f"Attention: dominant source {dominant_source or broadcast_source or 'none'}, "
            f"broadcast mode {broadcast_mode or 'none'}, {focus_count} focus item(s)."
        )
        return ContextItem(
            id="signal-attention",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="attention",
            metadata={"signal_type": "attention"},
        )

    def _affect_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        affect_state = self._facet_bucket("affect", state, facet_results)
        if not affect_state:
            return None
        snapshot = affect_state.get("last_affect_state")
        if not isinstance(snapshot, dict):
            return None
        valence = self._numeric_value(snapshot.get("valence"), allow_negative=True)
        arousal = self._numeric_value(snapshot.get("arousal"))
        dominance = self._numeric_value(snapshot.get("dominance"))
        novelty = self._numeric_value(snapshot.get("novelty"))
        content = (
            f"Affect: V={valence:.2f}, A={arousal:.2f}, "
            f"D={dominance:.2f}, N={novelty:.2f}."
        )
        return ContextItem(
            id="signal-affect",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="affect",
            metadata={"signal_type": "affect"},
        )

    def _learning_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        learning_state = self._facet_bucket("learning", state, facet_results)
        if not learning_state:
            return None
        signal_count = int(learning_state.get("last_signal_count", 0) or 0)
        proposal_count = int(learning_state.get("last_proposal_count", 0) or 0)
        adjustment_count = int(learning_state.get("last_adjustment_count", 0) or 0)
        if signal_count == 0 and proposal_count == 0 and adjustment_count == 0:
            return None
        content = (
            f"Learning: {signal_count} signal(s), {adjustment_count} adjustment(s), "
            f"{proposal_count} proposal(s)."
        )
        return ContextItem(
            id="signal-learning",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="learning",
            metadata={"signal_type": "learning"},
        )

    def _facet_bucket(
        self,
        facet_name: str,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> dict[str, Any] | None:
        if facet_results is not None:
            for result in reversed(facet_results):
                if result.facet_name != facet_name:
                    continue
                if result.state_updates:
                    return dict(result.state_updates)
                if result.metadata:
                    return dict(result.metadata)
        facet_state = state.facet_state.get(facet_name)
        if isinstance(facet_state, dict):
            return facet_state
        return None

    def _list_enabled_policies(self, limit: int) -> list[PolicyRule]:
        if hasattr(self.policy_store, "list_policies"):
            raw_policies = getattr(self.policy_store, "list_policies")(
                limit=limit,
                enabled_only=True,
            )
            if isinstance(raw_policies, list):
                return [
                    rule for rule in raw_policies if isinstance(rule, PolicyRule)
                ]
        if hasattr(self.policy_store, "list_enabled_policies"):
            raw_policies = getattr(self.policy_store, "list_enabled_policies")()
            if isinstance(raw_policies, list):
                return [
                    rule
                    for rule in raw_policies[:limit]
                    if isinstance(rule, PolicyRule)
                ]
        return []

    def _count_enabled_policies(self, fallback_policies: list[PolicyRule]) -> int:
        if hasattr(self.policy_store, "count_enabled_policies"):
            raw_count = getattr(self.policy_store, "count_enabled_policies")()
            if isinstance(raw_count, int) and raw_count >= 0:
                return raw_count
        if hasattr(self.policy_store, "list_enabled_policies"):
            raw_policies = getattr(self.policy_store, "list_enabled_policies")()
            if isinstance(raw_policies, list):
                return len(raw_policies)
        return len(fallback_policies)

    @staticmethod
    def _major_policy_constraints(policies: Sequence[PolicyRule]) -> list[str]:
        constraints: list[str] = []
        for target_type in EXTERNAL_POLICY_TARGETS:
            matching_rules = [
                rule
                for rule in policies
                if rule.target_type == target_type
                and rule.rule_type
                in {
                    PolicyRuleType.DENY,
                    PolicyRuleType.REQUIRE_APPROVAL,
                    PolicyRuleType.ALLOW,
                }
            ]
            if not matching_rules:
                continue
            top_rule = matching_rules[0]
            if top_rule.rule_type == PolicyRuleType.DENY:
                constraints.append(f"{target_type.value} denied by {top_rule.name}")
            elif top_rule.rule_type == PolicyRuleType.REQUIRE_APPROVAL:
                constraints.append(
                    f"{target_type.value} requires approval via {top_rule.name}"
                )
            else:
                constraints.append(f"{target_type.value} allowed by {top_rule.name}")
        return constraints

    @staticmethod
    def _source_types(items: Sequence[ContextItem]) -> list[str]:
        source_types: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = item.item_type.value
            if value in seen:
                continue
            seen.add(value)
            source_types.append(value)
        return source_types

    @staticmethod
    def _is_duplicate_attention_source_item(
        existing_items: Sequence[ContextItem],
        broadcast: AttentionBroadcast,
    ) -> bool:
        candidate_ids = {
            broadcast.item_id,
            str(broadcast.source_id or ""),
        }
        candidate_ids.discard("")
        for item in existing_items:
            if item.id in candidate_ids:
                return True
            if item.source_id in candidate_ids:
                return True
        return False

    @staticmethod
    def _memory_to_context_item(
        record: MemoryRecord,
        *,
        context_source: str,
        breakdown: dict[str, Any] | None = None,
    ) -> ContextItem:
        item = StaticContextAssembler._memory_to_context_item(record)
        item.metadata["context_source"] = context_source
        item.metadata["role"] = record.role
        item.metadata["domain"] = record.domain
        if breakdown is not None:
            item.metadata["hybrid_score"] = round(float(breakdown.get("total", 0.0)), 4)
            item.metadata["score_breakdown"] = breakdown
        return item

    @staticmethod
    def _coerce_string(raw_value: Any) -> str | None:
        if not isinstance(raw_value, str):
            return None
        cleaned = " ".join(raw_value.split())
        return cleaned or None

    @staticmethod
    def _numeric_value(
        raw_value: Any,
        *,
        allow_negative: bool = False,
    ) -> float:
        if not isinstance(raw_value, (int, float)):
            return 0.0
        value = float(raw_value)
        if allow_negative:
            return max(-1.0, min(value, 1.0))
        return max(0.0, min(value, 1.0))
