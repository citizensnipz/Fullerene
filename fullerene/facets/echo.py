"""Tiny example facet for exercising the Nexus loop."""

from __future__ import annotations

from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusState,
)


class EchoFacet:
    """Reflects user input into facet state without causing side effects."""

    name = "echo"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        if event.event_type == EventType.USER_MESSAGE:
            if not event.content.strip():
                return FacetResult(
                    facet_name=self.name,
                    summary=(
                        "Observed an empty user message and left echo state unchanged."
                    ),
                )
            seen_messages = (
                int(state.facet_state.get(self.name, {}).get("seen_messages", 0)) + 1
            )
            return FacetResult(
                facet_name=self.name,
                summary="Observed a user message and captured it for the runtime state.",
                proposed_decision=DecisionAction.RECORD,
                state_updates={
                    "last_user_message": event.content,
                    "seen_messages": seen_messages,
                },
            )

        if event.event_type == EventType.SYSTEM_TICK:
            return FacetResult(
                facet_name=self.name,
                summary="Observed a system tick and chose to stay idle.",
            )

        return FacetResult(
            facet_name=self.name,
            summary="Observed a system note and left a breadcrumb in facet state.",
            state_updates={"last_system_note": event.content},
        )
