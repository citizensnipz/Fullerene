from __future__ import annotations

from .models import DECISION_BASE_SCORES, BehaviorSignals


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))


def _world_confidence_boost(*, world_alignment_score: float, world_alignment_confidence: float) -> float:
    if world_alignment_score <= 0.0 or world_alignment_confidence <= 0.0:
        return 0.0
    normalized_alignment = _clamp_unit(world_alignment_score / 3.0)
    boost = 0.06 + (0.08 * _clamp_unit(world_alignment_confidence)) + (0.04 * normalized_alignment)
    return round(min(boost, 0.18), 3)


def confidence_breakdown(action_key: str, signals: BehaviorSignals) -> dict[str, float]:
    base = DECISION_BASE_SCORES[action_key]
    breakdown: dict[str, float] = {
        "base": base,
        "pressure_contribution": round(_clamp_unit(signals.pressure) * 0.25, 3),
        "goal_relevance_contribution": round(_clamp_unit(signals.goal_signal_strength) * 0.30, 3),
        "memory_retrieval_contribution": round(_clamp_unit(signals.relevant_memory_strength) * 0.30, 3),
        "ambiguity_penalty": round(_clamp_unit(signals.text.ambiguity_score) * -0.35, 3),
    }
    world_boost = _world_confidence_boost(
        world_alignment_score=signals.world_alignment_score,
        world_alignment_confidence=signals.world_alignment_confidence,
    )
    if world_boost > 0.0:
        breakdown["world_alignment_signal"] = world_boost
    breakdown["grounding_confidence"] = signals.text.grounding_confidence
    breakdown["continuity_confidence"] = signals.text.continuity_confidence
    breakdown["self_consistency_confidence"] = signals.text.self_consistency_confidence
    breakdown["challenge_confidence_penalty"] = -signals.text.challenge_confidence_penalty
    total = (
        breakdown["base"]
        + breakdown["pressure_contribution"]
        + breakdown["goal_relevance_contribution"]
        + breakdown["memory_retrieval_contribution"]
        + breakdown["ambiguity_penalty"]
        + breakdown.get("world_alignment_signal", 0.0)
        + (signals.text.grounding_confidence * 0.15)
        + (signals.text.continuity_confidence * 0.1)
        + (signals.text.self_consistency_confidence * 0.1)
        - signals.text.challenge_confidence_penalty
    )
    breakdown["total"] = round(_clamp_unit(total), 3)
    return breakdown


def contribution_reasons(signals: BehaviorSignals, breakdown: dict[str, float]) -> list[str]:
    return [
        f"pressure contribution: {signals.pressure:.3f} -> {breakdown['pressure_contribution']:.3f}",
        f"goal relevance contribution: {signals.goal_relevance:.3f} -> {breakdown['goal_relevance_contribution']:.3f}",
        f"memory contribution: {signals.relevant_memory_strength:.3f} -> {breakdown['memory_retrieval_contribution']:.3f}",
        f"ambiguity contribution: {signals.text.ambiguity_score:.3f} -> {breakdown['ambiguity_penalty']:.3f}",
        (
            "final confidence breakdown: "
            f"base={breakdown['base']:.3f}, pressure={breakdown['pressure_contribution']:.3f}, "
            f"goal={breakdown['goal_relevance_contribution']:.3f}, memory={breakdown['memory_retrieval_contribution']:.3f}, "
            f"ambiguity={breakdown['ambiguity_penalty']:.3f}, total={breakdown['total']:.3f}"
        ),
    ]


def final_confidence_reasons(signals: BehaviorSignals, breakdown: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if not signals.text.grounding_available:
        reasons.append("grounding unavailable lowers confidence")
    if signals.belief_contradiction:
        reasons.append("belief contradiction lowers confidence")
    if signals.text.repeated_dissatisfaction:
        reasons.append("repeated dissatisfaction lowers confidence")
    if signals.text.continuity_confidence >= 0.6:
        reasons.append("working memory continuity supports confidence")
    reasons.append(f"final_total={breakdown.get('total', 0.0):.3f}")
    return reasons
