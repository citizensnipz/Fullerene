"""Minimal Nexus runtime loop."""

from __future__ import annotations

from typing import Iterable

from fullerene.facets.base import Facet
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusState,
)
from fullerene.state.store import InMemoryStateStore, StateStore

# Higher score wins when multiple facets explicitly propose a decision.
# ACT > ASK > RECORD > WAIT.
DECISION_PRIORITY = {
    DecisionAction.WAIT: 0,
    DecisionAction.RECORD: 1,
    DecisionAction.ASK: 2,
    DecisionAction.ACT: 3,
}


class Nexus:
    """Central interpreter/integrator loop for Fullerene v0."""

    def __init__(
        self,
        facets: Iterable[Facet] | None = None,
        store: StateStore | None = None,
        initial_state: NexusState | None = None,
    ) -> None:
        self._store = store or InMemoryStateStore()
        self._facets: list[Facet] = list(facets or [])
        self.state = initial_state or self._store.load_state() or NexusState()

    @property
    def facets(self) -> tuple[Facet, ...]:
        return tuple(self._facets)

    def register_facet(self, facet: Facet) -> None:
        self._facets.append(facet)

    def process_event(self, event: Event) -> NexusRecord:
        facet_results = [self._run_facet(facet, event) for facet in self._facets]
        decision = self._integrate(event, facet_results)
        self.state.apply(event, facet_results, decision)

        record = NexusRecord(
            event=event,
            facet_results=facet_results,
            decision=decision,
        )
        self._store.save_state(self.state)
        self._store.append_record(record)
        return record

    def _run_facet(self, facet: Facet, event: Event) -> FacetResult:
        try:
            return facet.process(event, self.state)
        except Exception as exc:
            facet_name = self._facet_name(facet)
            error_message = str(exc) or "Facet raised without an error message."
            return FacetResult(
                facet_name=facet_name,
                summary=(
                    f"Facet '{facet_name}' failed while processing the event: "
                    f"{error_message}"
                ),
                proposed_decision=DecisionAction.RECORD,
                metadata={
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message,
                },
            )

    def _facet_name(self, facet: Facet) -> str:
        raw_name = getattr(facet, "name", "") or facet.__class__.__name__
        return str(raw_name)

    def _integrate(
        self,
        event: Event,
        facet_results: list[FacetResult],
    ) -> NexusDecision:
        explicit_results = [
            result for result in facet_results if result.proposed_decision is not None
        ]
        if explicit_results:
            selected_action = max(
                (result.proposed_decision for result in explicit_results),
                key=lambda action: DECISION_PRIORITY[action],
            )
            source_facets = [
                result.facet_name
                for result in explicit_results
                if result.proposed_decision == selected_action
            ]
            reason = (
                f"Selected {selected_action.value.upper()} from facet proposals: "
                f"{', '.join(source_facets)}."
            )
            return NexusDecision(
                action=selected_action,
                reason=reason,
                source_facets=source_facets,
            )

        if event.event_type == EventType.USER_MESSAGE:
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD for a user message event.",
            )

        if any(result.state_updates for result in facet_results):
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD because facets produced state updates.",
            )

        return NexusDecision(
            action=DecisionAction.WAIT,
            reason="Defaulted to WAIT because no facet proposed or updated anything.",
        )


class NexusRuntime(Nexus):
    """Explicit runtime alias for callers that prefer a runtime-oriented name."""
