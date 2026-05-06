from __future__ import annotations

from typing import Any

from fullerene.nexus.models import DecisionAction, Event, EventType

from .models import (
    CONTEXT_SUFFICIENCY_THRESHOLD,
    CONTRADICTION_ACT_PENALTY,
    DECISION_BASE_SCORES,
    DECISION_PRIORITY,
    GOAL_RELEVANCE_THRESHOLD,
    HIGH_AMBIGUITY_THRESHOLD,
    LOW_AMBIGUITY_THRESHOLD,
    LOW_BELIEF_CONFIDENCE_THRESHOLD,
    BehaviorSignals,
)


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))


def _is_unclear(signals: BehaviorSignals) -> bool:
    return (
        signals.requires_response
        or signals.text.response_needed
        or signals.uncertainty
        or signals.question_like
        or (signals.explicit_action and not signals.low_risk)
    )


def select_decision(
    event: Event,
    signals: BehaviorSignals,
) -> tuple[DecisionAction, list[str], dict[str, float]]:
    reasons: list[str] = []
    score_breakdown: dict[DecisionAction, dict[str, float]] = {
        action: {"base": base_score} for action, base_score in DECISION_BASE_SCORES.items()
    }

    def add(action: DecisionAction, reason: str, value: float) -> None:
        if value == 0.0:
            return
        score_breakdown[action][reason] = score_breakdown[action].get(reason, 0.0) + value

    if event.event_type in {EventType.SYSTEM_TICK, EventType.INTERNAL}:
        reasons.append("internal_system_event_wait")
        add(DecisionAction.WAIT, "internal_system_event_wait", 0.8)
        _apply_pressure_biases(score_breakdown, reasons, signals)
        return _finalize(DecisionAction.WAIT, reasons, score_breakdown)

    if not signals.meaningful_content and not signals.has_metadata_signal:
        reasons.append("empty_content_wait")
        add(DecisionAction.WAIT, "empty_content_wait", 0.8)
        return _finalize(DecisionAction.WAIT, reasons, score_breakdown)

    if signals.explicit_action:
        if signals.low_risk and signals.text.ambiguity_score < HIGH_AMBIGUITY_THRESHOLD:
            reasons.append("explicit_action_low_risk")
            add(DecisionAction.ACT, "explicit_action_low_risk", 0.75)
            selected = DecisionAction.ACT
        else:
            reasons.append("explicit_action_without_low_risk")
            add(DecisionAction.ASK, "explicit_action_without_low_risk", 0.75)
            selected = DecisionAction.ASK
        _apply_pressure_biases(score_breakdown, reasons, signals)
        _apply_goal_biases(score_breakdown, reasons, signals)
        _apply_memory_biases(score_breakdown, reasons, signals)
        return _finalize(selected, reasons, score_breakdown)

    if not signals.text.response_needed:
        if signals.meaningful_content:
            if event.event_type == EventType.USER_MESSAGE:
                reasons.append("user_message_default_record")
            if signals.high_priority:
                reasons.append("high_priority_tags")
            reasons.append("non_question_statement_record")
            add(DecisionAction.RECORD, "non_question_statement_record", 0.75)
            selected = DecisionAction.RECORD
        else:
            reasons.append("no_response_needed_wait")
            add(DecisionAction.WAIT, "no_response_needed_wait", 0.55)
            selected = DecisionAction.WAIT
        _apply_pressure_biases(score_breakdown, reasons, signals)
        _apply_goal_biases(score_breakdown, reasons, signals)
        _apply_memory_biases(score_breakdown, reasons, signals)
        _apply_low_signal_bias(score_breakdown, reasons, signals)
        return _finalize(selected, reasons, score_breakdown)

    grounded = (
        signals.has_relevant_memory
        or signals.has_goal
        or signals.text.deterministic_response_available
        or signals.context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD
    )
    low_ambiguity = signals.text.ambiguity_score <= LOW_AMBIGUITY_THRESHOLD
    high_ambiguity = signals.text.ambiguity_score >= HIGH_AMBIGUITY_THRESHOLD

    if signals.text.query_intent in {"recommendation", "planning"} and signals.has_preference_memory:
        reasons.append("preference_memory_signal")
        add(DecisionAction.ACT, "preference_memory_signal", 0.8)
        grounded = True
        low_ambiguity = True

    if signals.has_goal and signals.goal_signal_strength >= GOAL_RELEVANCE_THRESHOLD:
        reasons.append("goal_signal")
        add(DecisionAction.ACT, "goal_signal", 0.65)
        grounded = True

    if signals.has_relevant_memory:
        reasons.append("relevant_memory_signal")
        add(DecisionAction.ACT, "relevant_memory_signal", 0.55)

    if signals.requires_response:
        reasons.append("requires_response_metadata")
        add(DecisionAction.ASK, "requires_response_metadata", 0.25)

    intent_needs_response = signals.text.query_intent in {
        "recommendation",
        "planning",
        "factual",
        "memory_summary",
    }
    if grounded and low_ambiguity and intent_needs_response:
        reasons.append("grounded_low_ambiguity_act")
        add(DecisionAction.ACT, "grounded_low_ambiguity_act", 0.75)
        selected = DecisionAction.ACT
    elif high_ambiguity:
        reasons.append("high_ambiguity_insufficient_context")
        add(DecisionAction.ASK, "high_ambiguity_insufficient_context", 0.45)
        selected = DecisionAction.ASK
    elif grounded and intent_needs_response:
        reasons.append("grounded_response_context_act")
        add(DecisionAction.ACT, "grounded_response_context_act", 0.45)
        selected = DecisionAction.ACT
    else:
        reasons.append("response_needed_but_unclear_ask")
        add(DecisionAction.ASK, "response_needed_but_unclear_ask", 0.55)
        selected = DecisionAction.ASK

    _apply_pressure_biases(score_breakdown, reasons, signals)
    _apply_goal_biases(score_breakdown, reasons, signals)
    _apply_memory_biases(score_breakdown, reasons, signals)
    return _finalize(selected, reasons, score_breakdown)


def _finalize(
    selected_decision: DecisionAction,
    reasons: list[str],
    score_breakdown: dict[DecisionAction, dict[str, float]],
) -> tuple[DecisionAction, list[str], dict[str, float]]:
    decision_scores = {
        action: round(_clamp_unit(sum(breakdown.values())), 3)
        for action, breakdown in score_breakdown.items()
    }
    reasons.append(f"selected_policy_rule:{selected_decision.value}")
    return selected_decision, reasons, {a.value: decision_scores[a] for a in DECISION_BASE_SCORES}


def _apply_pressure_biases(
    score_breakdown: dict[DecisionAction, dict[str, float]],
    reasons: list[str],
    signals: BehaviorSignals,
) -> None:
    pressure = signals.pressure
    if pressure <= 0.0:
        reasons.append("pressure contribution: 0.000 no pressure bias applied")
        return
    actionable = signals.explicit_action and signals.low_risk
    unclear = _is_unclear(signals)
    score_breakdown[DecisionAction.ACT]["pressure_act_bias"] = pressure * (0.3 if actionable else 0.15)
    score_breakdown[DecisionAction.ASK]["pressure_ask_bias"] = pressure * (0.25 if unclear else 0.1)
    score_breakdown[DecisionAction.WAIT]["pressure_wait_penalty"] = pressure * -0.2
    reasons.extend(
        ["high pressure increased ACT score", "high pressure increased ASK score", "pressure reduced WAIT score"]
    )


def _apply_goal_biases(
    score_breakdown: dict[DecisionAction, dict[str, float]],
    reasons: list[str],
    signals: BehaviorSignals,
) -> None:
    goal_relevance = signals.goal_relevance
    if goal_relevance <= 0.0:
        reasons.append("goal relevance contribution: 0.000 no goal bias applied")
        return
    actionable = signals.explicit_action and signals.low_risk
    unclear = _is_unclear(signals)
    if actionable:
        score_breakdown[DecisionAction.ACT]["goal_relevance_act_bias"] = goal_relevance * 0.35
        reasons.append("goal priority boosted ACT score")
    if signals.text.deterministic_response_available:
        score_breakdown[DecisionAction.ACT]["goal_response_act_bias"] = goal_relevance * 0.25
        reasons.append("goal context supported deterministic ACT response")
    if unclear and not signals.text.deterministic_response_available:
        score_breakdown[DecisionAction.ASK]["goal_relevance_ask_bias"] = goal_relevance * 0.25
        reasons.append("goal priority boosted ASK score")
    reasons.append("goal priority boosted decision confidence")


def _apply_memory_biases(
    score_breakdown: dict[DecisionAction, dict[str, float]],
    reasons: list[str],
    signals: BehaviorSignals,
) -> None:
    retrieval_strength = signals.retrieval_strength
    if retrieval_strength <= 0.0:
        reasons.append("memory contribution: 0.000 no retrieval bias applied")
        return
    if signals.explicit_action and signals.low_risk:
        score_breakdown[DecisionAction.ACT]["memory_retrieval_act_bias"] = retrieval_strength * 0.1
    if _is_unclear(signals):
        score_breakdown[DecisionAction.ASK]["memory_retrieval_ask_bias"] = retrieval_strength * 0.1
    reasons.append("memory retrieval strength increased decision confidence")


def _apply_low_signal_bias(
    score_breakdown: dict[DecisionAction, dict[str, float]],
    reasons: list[str],
    signals: BehaviorSignals,
) -> None:
    low_signal = (
        signals.pressure <= 0.0
        and signals.goal_relevance <= 0.0
        and signals.retrieval_strength <= 0.0
        and signals.salience < 0.6
        and not signals.explicit_action
        and not _is_unclear(signals)
    )
    if not low_signal:
        return
    if signals.meaningful_content:
        score_breakdown[DecisionAction.RECORD]["low_signal_record_bias"] = 0.1
        reasons.append("low signal environment favored RECORD")
    else:
        score_breakdown[DecisionAction.WAIT]["low_signal_wait_bias"] = 0.15
        reasons.append("low signal environment favored WAIT")


def apply_v2_candidate_adjustments(
    decision_scores: dict[str, float],
    signals: BehaviorSignals,
    reasons: list[str],
) -> dict[str, float]:
    adjusted = {key: _clamp_unit(value) for key, value in decision_scores.items()}
    if signals.latent_pressure > 0.0:
        adjusted["ask"] = _clamp_unit(
            adjusted["ask"] + (signals.latent_pressure * (0.2 + (0.1 * signals.goal_relevance)))
        )
        adjusted["act"] = _clamp_unit(
            adjusted["act"] + (signals.latent_pressure * (0.1 + (0.15 * signals.goal_relevance)))
        )
        adjusted["record"] = _clamp_unit(
            adjusted["record"] + (signals.latent_pressure * (0.12 * (1.0 - signals.goal_relevance)))
        )
        reasons.append("latent_pressure influenced ask/act/record scoring")

    if 0.0 < signals.belief_confidence < LOW_BELIEF_CONFIDENCE_THRESHOLD:
        adjusted["act"] = _clamp_unit(adjusted["act"] - 0.35)
        adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
        reasons.append("low belief confidence suppressed ACT and boosted ASK")
    if signals.belief_contradiction:
        adjusted["act"] = _clamp_unit(adjusted["act"] - CONTRADICTION_ACT_PENALTY)
        adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.25)
        reasons.append("belief contradiction biased decision toward ASK")
    if signals.context_overloaded:
        adjusted["act"] = _clamp_unit(adjusted["act"] - (0.25 * (1.0 - (signals.pressure * signals.goal_relevance))))
        adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.15)
        adjusted["record"] = _clamp_unit(adjusted["record"] + 0.1)
        reasons.append("context_overload biased away from ACT")
    return {key: round(value, 3) for key, value in adjusted.items()}


def apply_v21_candidate_adjustments(
    decision_scores: dict[str, float],
    signals: BehaviorSignals,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {key: _clamp_unit(value) for key, value in decision_scores.items()}
    reasons: list[str] = []
    if signals.text.conversational_intent == "source_request":
        if signals.text.grounding_available:
            adjusted["act"] = _clamp_unit(adjusted["act"] + 0.25)
            reasons.append("source_request with grounding boosted ACT")
        else:
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.25)
            adjusted["act"] = _clamp_unit(adjusted["act"] - 0.2)
            reasons.append("source_request without grounding biased ASK")
    if signals.text.conversational_intent in {"challenge", "contradiction_report", "correction"}:
        adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
        adjusted["act"] = _clamp_unit(adjusted["act"] - (0.15 if signals.text.grounding_available else 0.25))
        reasons.append("challenge_or_contradiction bias toward ASK")
    if signals.text.conversational_intent == "clarification_supplied":
        adjusted["act"] = _clamp_unit(adjusted["act"] + 0.2)
        adjusted["ask"] = _clamp_unit(adjusted["ask"] - 0.1)
        reasons.append("clarification_supplied reduced ASK bias")
    if signals.text.conversational_intent == "follow_up":
        if signals.text.has_resolved_reference:
            adjusted["act"] = _clamp_unit(adjusted["act"] + 0.2)
            adjusted["ask"] = _clamp_unit(adjusted["ask"] - 0.1)
            reasons.append("context_continuity_supported_follow_up")
            reasons.append("resolved_reference_lowered_ambiguity")
        else:
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
            adjusted["act"] = _clamp_unit(adjusted["act"] - 0.1)
            reasons.append("unresolved_reference_requires_targeted_clarification")
    if signals.text.repeated_dissatisfaction:
        adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.1)
        adjusted["act"] = _clamp_unit(adjusted["act"] - 0.15)
        reasons.append("repeated dissatisfaction lowered ACT confidence path")
    return ({k: round(v, 3) for k, v in adjusted.items()}, reasons)


def select_highest_scored_decision(decision_scores: dict[str, float]) -> DecisionAction:
    ordering = {"wait": DecisionAction.WAIT, "record": DecisionAction.RECORD, "ask": DecisionAction.ASK, "act": DecisionAction.ACT}
    selected_key = max(ordering, key=lambda key: (decision_scores.get(key, 0.0), DECISION_PRIORITY[ordering[key]]))
    return ordering[selected_key]


def apply_policy_downgrade(
    selected_decision: DecisionAction,
    decision_scores: dict[str, float],
    signals: BehaviorSignals,
) -> tuple[DecisionAction, list[str]]:
    reasons: list[str] = []
    if selected_decision != DecisionAction.ACT or not signals.policy_blocks_act:
        return selected_decision, reasons
    reasons.append(f"policy_result:{signals.policy_result}")
    if signals.policy_requires_approval:
        reasons.append("policy_downgrade:act_to_ask_requires_approval")
        return DecisionAction.ASK, reasons
    if decision_scores.get("ask", 0.0) >= decision_scores.get("record", 0.0):
        reasons.append("policy_downgrade:act_to_ask_clarification_path")
        return DecisionAction.ASK, reasons
    if signals.meaningful_content:
        reasons.append("policy_downgrade:act_to_record_useful_not_actionable")
        return DecisionAction.RECORD, reasons
    reasons.append("policy_downgrade:act_to_wait_no_safe_path")
    return DecisionAction.WAIT, reasons


def interrupt_recommendation(signals: BehaviorSignals) -> tuple[bool, str | None]:
    if signals.pressure >= 0.85:
        return True, "pressure_spike"
    if signals.goal_relevance >= 0.85 and signals.pressure >= 0.6:
        return True, "high_goal_relevance_under_pressure"
    if signals.latent_pressure >= 0.75:
        return True, "latent_pressure_high"
    return False, None
