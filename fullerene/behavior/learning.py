from __future__ import annotations

from .models import BehaviorSignals


def learning_signals(signals: BehaviorSignals) -> list[str]:
    out: list[str] = []
    if signals.text.conversational_intent == "source_request" and not signals.text.grounding_available:
        out.append("source_request_unresolved")
    if signals.text.conversational_intent in {"challenge", "contradiction_report"} and not signals.text.grounding_available:
        out.append("challenge_unresolved")
    if signals.text.repeated_dissatisfaction:
        out.append("repeated_dissatisfaction")
    if signals.text.conversational_intent == "clarification_supplied":
        out.append("clarification_supplied")
    if signals.text.conversational_intent == "follow_up" and signals.text.continuity_confidence >= 0.6:
        out.append("follow_up_resolved_by_working_memory")
    if signals.text.grounding_confidence < 0.4:
        out.append("low_grounding_confidence")
    if signals.belief_contradiction:
        out.append("contradiction_pressure")
    return out
