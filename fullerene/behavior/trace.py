from __future__ import annotations

from datetime import datetime, timezone

from fullerene.nexus.models import DecisionAction, Event

from .models import BehaviorSignals


def build_decision_trace(
    *,
    event: Event,
    signals: BehaviorSignals,
    raw_candidate_scores: dict[str, float],
    adjusted_candidate_scores: dict[str, float],
    selected_decision: DecisionAction,
    confidence: float,
    reasons: list[str],
    interrupt_recommended: bool,
    interrupt_reason: str | None,
) -> dict[str, object]:
    event_summary = " ".join(event.content.split())[:120]
    return {
        "event": {"id": event.event_id, "type": event.event_type.value, "content_summary": event_summary},
        "pressure_score": signals.pressure,
        "latent_pressure": signals.latent_pressure,
        "memory_relevance_score": signals.relevant_memory_strength,
        "goal_relevance_score": signals.goal_relevance,
        "world_model_belief_confidence": signals.belief_confidence,
        "contradiction_flag": signals.belief_contradiction,
        "policy_result": signals.policy_result,
        "context_load_ratio": signals.context_load_ratio,
        "conversational_intent": signals.text.conversational_intent,
        "conversational_intent_score": signals.text.conversational_intent_score,
        "grounding_need": signals.text.grounding_need,
        "grounding_available": signals.text.grounding_available,
        "grounding_confidence": signals.text.grounding_confidence,
        "ambiguity_kind": signals.text.ambiguity_kind,
        "ambiguity_score": signals.text.ambiguity_score,
        "continuity_confidence": signals.text.continuity_confidence,
        "self_consistency_confidence": signals.text.self_consistency_confidence,
        "challenge_confidence_penalty": signals.text.challenge_confidence_penalty,
        "raw_candidate_scores": raw_candidate_scores,
        "adjusted_candidate_scores": adjusted_candidate_scores,
        "final_decision": selected_decision.value,
        "confidence": confidence,
        "reasons": list(reasons),
        "interrupt_recommended": interrupt_recommended,
        "interrupt_reason": interrupt_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
