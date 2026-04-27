"""Stateless Learning v0 feedback facet."""

from __future__ import annotations

from fullerene.goals import GoalStore
from fullerene.learning import build_learning_result
from fullerene.memory import MemoryStore
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState


class LearningFacet:
    """Observe outcomes and emit traceable adjustments without owning state."""

    name = "learning"

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        goal_store: GoalStore | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.goal_store = goal_store

    def process(self, event: Event, state: NexusState) -> FacetResult:
        learning_result = build_learning_result(
            event,
            state,
            memory_store=self.memory_store,
            goal_store=self.goal_store,
        )
        has_learning_output = bool(
            learning_result.signals or learning_result.adjustments
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
                "reasons": [
                    reason
                    for signal in learning_result.signals
                    for reason in signal.reasons
                ],
            },
        )
