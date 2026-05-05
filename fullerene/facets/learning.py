"""Stateless Learning v1 feedback facet (deterministic cross-facet routing)."""

from __future__ import annotations

from fullerene.goals import GoalStore
from fullerene.learning import build_learning_result
from fullerene.memory import MemoryStore
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.world_model.store import WorldModelStore


class LearningFacet:
    """Observe outcomes and emit traceable adjustments without owning state."""

    name = "learning"

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.goal_store = goal_store
        self.world_model_store = world_model_store

    def process(self, event: Event, state: NexusState) -> FacetResult:
        learning_result = build_learning_result(
            event,
            state,
            memory_store=self.memory_store,
            goal_store=self.goal_store,
            world_model_store=self.world_model_store,
        )
        meta = learning_result.metadata if isinstance(learning_result.metadata, dict) else {}
        has_learning_output = bool(
            learning_result.signals
            or learning_result.adjustments
            or meta.get("cross_facet_routes")
        )
        proposed_decision = (
            DecisionAction.RECORD if has_learning_output else DecisionAction.WAIT
        )
        skipped = [
            record
            for record in learning_result.adjustments
            if record.status.value == "skipped"
        ]
        return FacetResult(
            facet_name=self.name,
            summary=(
                f"Learning facet classified {len(learning_result.signals)} signal(s) and "
                f"produced {len(learning_result.adjustments)} adjustment record(s)."
            )
            if has_learning_output
            else "Learning facet found no feedback signals for this event.",
            proposed_decision=proposed_decision,
            state_updates={
                "last_learning_result": learning_result.to_dict(),
                "last_signal_count": len(learning_result.signals),
                "last_adjustment_count": len(learning_result.adjustments),
                "last_proposal_count": len(learning_result.proposals),
                "last_applied_count": len(learning_result.applied),
                "last_skipped_count": len(skipped),
                "last_learning_version": meta.get("learning_version", "v1"),
                "last_consumed_learning_events": meta.get("consumed_learning_events", []),
                "last_adjustment_records": meta.get("adjustment_records", []),
                "last_proposed_adjustments": meta.get("proposed_adjustments", []),
                "last_applied_adjustments": meta.get("applied_adjustments", []),
                "last_cross_facet_routes": meta.get("cross_facet_routes", []),
                "last_signal_sources": meta.get("signal_sources", []),
                "last_skipped_adjustments": meta.get("skipped_adjustments", []),
            },
            metadata={
                "learning_result": learning_result.to_dict(),
                "signals": [signal.to_dict() for signal in learning_result.signals],
                "adjustments": [
                    record.to_dict() for record in learning_result.adjustments
                ],
                "proposals": [
                    record.to_dict() for record in learning_result.proposals
                ],
                "applied": [record.to_dict() for record in learning_result.applied],
                "skipped": [record.to_dict() for record in skipped],
                "reasons": meta.get("reasons", []),
            },
        )
