"""Deterministic Affect v0 observation facet."""

from __future__ import annotations

from fullerene.affect import (
    AffectHistoryBuffer,
    AffectResult,
    DeterministicAffectDeriver,
)
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState


class AffectFacet:
    """Compute and record internal affect state without influencing other facets."""

    name = "affect"

    def __init__(
        self,
        *,
        history_size: int = 20,
        deriver: DeterministicAffectDeriver | None = None,
    ) -> None:
        self.history_size = max(int(history_size), 1)
        self.deriver = deriver or DeterministicAffectDeriver()

    def process(self, event: Event, state: NexusState) -> FacetResult:
        prior_history = self._load_history(state)
        derived = self.deriver.derive(event, state, history=prior_history.states)
        prior_history.append(derived.current_state)
        affect_result = AffectResult(
            current_state=derived.current_state,
            history=prior_history.states,
            strategy=derived.strategy,
            reasons=derived.reasons,
            metadata={
                **derived.metadata,
                "history_count": len(prior_history.states),
                "history_max_size": self.history_size,
            },
        )
        affect_state_payload = affect_result.current_state.to_dict()
        history_payload = prior_history.to_dict()

        return FacetResult(
            facet_name=self.name,
            summary=(
                "Affect facet recorded internal state "
                f"V={affect_result.current_state.valence:.3f}, "
                f"A={affect_result.current_state.arousal:.3f}, "
                f"D={affect_result.current_state.dominance:.3f}, "
                f"N={affect_result.current_state.novelty:.3f}."
            ),
            proposed_decision=DecisionAction.RECORD,
            state_updates={
                "last_affect_state": affect_state_payload,
                "last_affect_result": affect_result.to_dict(),
                "last_affect_state_id": affect_result.current_state.id,
                "last_strategy": affect_result.strategy,
                "history": history_payload,
                "history_count": len(history_payload),
                "history_max_size": self.history_size,
            },
            metadata={
                "affect_result": affect_result.to_dict(),
                "affect_state": affect_state_payload,
                "history_count": len(history_payload),
                "strategy": affect_result.strategy,
                "reasons": list(affect_result.reasons),
                "components": dict(affect_result.current_state.components),
            },
        )

    def _load_history(self, state: NexusState) -> AffectHistoryBuffer:
        facet_state = state.facet_state.get(self.name)
        if not isinstance(facet_state, dict):
            return AffectHistoryBuffer(max_size=self.history_size)
        return AffectHistoryBuffer.from_payload(
            facet_state.get("history"),
            max_size=self.history_size,
        )
