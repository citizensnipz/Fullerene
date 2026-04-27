"""Deterministic behavior facet for Fullerene Behavior v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullerene.memory import compute_salience, infer_tags, merge_tags, normalize_tags
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusState,
)

HIGH_PRIORITY_TAGS = frozenset(
    {"hard-rule-candidate", "correction", "urgent", "authority"}
)
RESPONSE_PHRASES = (
    "what should i do",
    "should i",
    "can you",
    "help me",
    "how do i",
)
DECISION_BASE_CONFIDENCE = {
    DecisionAction.WAIT: 0.88,
    DecisionAction.RECORD: 0.68,
    DecisionAction.ASK: 0.78,
    DecisionAction.ACT: 0.84,
}
REASON_CONFIDENCE_BOOSTS = {
    "empty_content_wait": 0.08,
    "user_message_default_record": 0.06,
    "system_note_default_record": 0.04,
    "high_priority_tags": 0.04,
    "high_priority_record": 0.02,
    "question_phrase_response_needed": 0.06,
    "requires_response_metadata": 0.1,
    "uncertainty_metadata": 0.08,
    "explicit_action_without_low_risk": 0.08,
    "explicit_action_low_risk": 0.1,
    "system_tick_or_idle_wait": 0.04,
}


@dataclass(slots=True)
class _BehaviorSignals:
    tags: list[str]
    salience: float
    salience_source: str
    meaningful_content: bool
    has_metadata_signal: bool
    question_like: bool
    requires_response: bool
    explicit_action: bool
    low_risk: bool
    uncertainty: bool
    high_priority: bool
    memory_signal_available: bool
    goal_signal_available: bool
    goal_alignment_score: float
    goal_alignment_priority: float
    aligned_goal_ids: list[str]
    world_signal_available: bool
    world_alignment_score: float
    world_alignment_confidence: float
    aligned_belief_ids: list[str]


class BehaviorFacet:
    """Choose an inspectable Nexus decision from deterministic rules only."""

    name = "behavior"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        signals = self._collect_signals(event, state)
        selected_decision, reasons = self._select_decision(event, signals)
        confidence_breakdown = self._score_confidence(
            selected_decision,
            reasons,
            salience=signals.salience,
            goal_alignment_score=signals.goal_alignment_score,
            goal_alignment_priority=signals.goal_alignment_priority,
            world_alignment_score=signals.world_alignment_score,
            world_alignment_confidence=signals.world_alignment_confidence,
        )
        confidence = confidence_breakdown["total"]
        priority_level = "high" if signals.high_priority else "normal"

        return FacetResult(
            facet_name=self.name,
            summary=(
                f"Behavior facet selected {selected_decision.value.upper()} "
                f"from deterministic rules: {', '.join(reasons)}."
            ),
            proposed_decision=selected_decision,
            state_updates={
                "last_event_id": event.event_id,
                "last_selected_decision": selected_decision.value,
                "last_confidence": confidence,
                "last_salience": signals.salience,
                "last_tags_considered": list(signals.tags),
                "last_reasons": list(reasons),
                "last_priority_level": priority_level,
                "last_goal_alignment_score": signals.goal_alignment_score,
                "last_aligned_goal_ids": list(signals.aligned_goal_ids),
                "last_world_alignment_score": signals.world_alignment_score,
                "last_aligned_belief_ids": list(signals.aligned_belief_ids),
            },
            metadata={
                "selected_decision": selected_decision.value,
                "confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "salience": signals.salience,
                "salience_source": signals.salience_source,
                "tags_considered": list(signals.tags),
                "reasons": list(reasons),
                "high_priority": signals.high_priority,
                "priority_level": priority_level,
                "response_needed": selected_decision == DecisionAction.ASK,
                "memory_signal_available": signals.memory_signal_available,
                "goal_signal_available": signals.goal_signal_available,
                "goal_alignment_score": signals.goal_alignment_score,
                "goal_alignment_priority": signals.goal_alignment_priority,
                "aligned_goal_ids": list(signals.aligned_goal_ids),
                "world_signal_available": signals.world_signal_available,
                "world_alignment_score": signals.world_alignment_score,
                "world_alignment_confidence": signals.world_alignment_confidence,
                "aligned_belief_ids": list(signals.aligned_belief_ids),
            },
        )

    def _collect_signals(self, event: Event, state: NexusState) -> _BehaviorSignals:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        metadata_tags = self._normalize_tag_group(metadata.get("tags"))
        memory_context = self._extract_memory_context(metadata)
        memory_tags = self._normalize_tag_group(
            memory_context.get("tags") if memory_context else []
        )
        tags = merge_tags(metadata_tags, memory_tags, infer_tags(event.content))
        salience, salience_source = self._resolve_salience(
            event,
            metadata,
            memory_context,
            tags,
        )
        goal_context = self._extract_goal_context(metadata, state)
        aligned_goals = self._extract_relevant_goals(goal_context)
        goal_alignment_score = self._resolve_goal_alignment_score(
            goal_context,
            aligned_goals,
        )
        goal_alignment_priority = self._resolve_goal_alignment_priority(aligned_goals)
        world_context = self._extract_world_context(metadata, state)
        aligned_beliefs = self._extract_relevant_beliefs(world_context)
        world_alignment_score = self._resolve_world_alignment_score(
            world_context,
            aligned_beliefs,
        )
        world_alignment_confidence = self._resolve_world_alignment_confidence(
            aligned_beliefs
        )

        return _BehaviorSignals(
            tags=tags,
            salience=salience,
            salience_source=salience_source,
            meaningful_content=bool(event.content.strip()),
            has_metadata_signal=bool(
                metadata_tags
                or memory_tags
                or self._metadata_flag(metadata, "requires_response")
                or self._metadata_flag(metadata, "explicit_action")
                or self._metadata_flag(metadata, "low_risk")
                or self._metadata_flag(metadata, "uncertainty")
                or "salience" in metadata
            ),
            question_like=self._contains_response_phrase(event.content),
            requires_response=self._metadata_flag(metadata, "requires_response"),
            explicit_action=self._metadata_flag(metadata, "explicit_action"),
            low_risk=self._metadata_flag(metadata, "low_risk"),
            uncertainty=self._metadata_flag(metadata, "uncertainty"),
            high_priority=bool(HIGH_PRIORITY_TAGS & set(tags)),
            memory_signal_available=bool(memory_context)
            or isinstance(state.facet_state.get("memory"), dict),
            goal_signal_available=goal_context is not None,
            goal_alignment_score=goal_alignment_score,
            goal_alignment_priority=goal_alignment_priority,
            aligned_goal_ids=[
                str(goal.get("id"))
                for goal in aligned_goals
                if isinstance(goal.get("id"), str)
            ],
            world_signal_available=world_context is not None,
            world_alignment_score=world_alignment_score,
            world_alignment_confidence=world_alignment_confidence,
            aligned_belief_ids=[
                str(belief.get("id"))
                for belief in aligned_beliefs
                if isinstance(belief.get("id"), str)
            ],
        )

    @staticmethod
    def _normalize_tag_group(raw_tags: Any) -> list[str]:
        if isinstance(raw_tags, (list, tuple, set, frozenset)):
            return normalize_tags(raw_tags)
        return []

    @staticmethod
    def _extract_memory_context(metadata: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("memory", "memory_facet", "stored_memory", "relevant_memory"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                return candidate
        return None

    @staticmethod
    def _extract_goal_context(
        metadata: dict[str, Any],
        state: NexusState,
    ) -> dict[str, Any] | None:
        for key in ("goals", "goal_signal", "goals_facet"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                return candidate

        state_goals = state.facet_state.get("goals")
        return state_goals if isinstance(state_goals, dict) else None

    @staticmethod
    def _extract_world_context(
        metadata: dict[str, Any],
        state: NexusState,
    ) -> dict[str, Any] | None:
        for key in ("world_model", "world_signal", "world_model_facet"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                return candidate

        state_world_model = state.facet_state.get("world_model")
        return state_world_model if isinstance(state_world_model, dict) else None

    @staticmethod
    def _extract_relevant_goals(
        goal_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if goal_context is None:
            return []

        for key in ("last_relevant_goals", "relevant_goals"):
            candidate = goal_context.get(key)
            if isinstance(candidate, list):
                return [goal for goal in candidate if isinstance(goal, dict)]
        return []

    @staticmethod
    def _extract_relevant_beliefs(
        world_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if world_context is None:
            return []

        for key in ("last_relevant_beliefs", "relevant_beliefs"):
            candidate = world_context.get(key)
            if isinstance(candidate, list):
                return [belief for belief in candidate if isinstance(belief, dict)]
        return []

    @staticmethod
    def _resolve_goal_alignment_score(
        goal_context: dict[str, Any] | None,
        aligned_goals: list[dict[str, Any]],
    ) -> float:
        if goal_context is not None:
            raw_score = BehaviorFacet._numeric_score(
                goal_context.get("last_relevance_score")
            )
            if raw_score is not None:
                return raw_score

        best_score = 0.0
        for goal in aligned_goals:
            goal_score = BehaviorFacet._numeric_score(goal.get("score"))
            if goal_score is not None:
                best_score = max(best_score, goal_score)
        return best_score

    @staticmethod
    def _resolve_goal_alignment_priority(
        aligned_goals: list[dict[str, Any]],
    ) -> float:
        best_priority = 0.0
        for goal in aligned_goals:
            goal_priority = BehaviorFacet._numeric_unit_value(goal.get("priority"))
            if goal_priority is not None:
                best_priority = max(best_priority, goal_priority)
        return best_priority

    @staticmethod
    def _resolve_world_alignment_score(
        world_context: dict[str, Any] | None,
        aligned_beliefs: list[dict[str, Any]],
    ) -> float:
        if world_context is not None:
            raw_score = BehaviorFacet._numeric_score(
                world_context.get("last_relevance_score")
            )
            if raw_score is not None:
                return raw_score

        best_score = 0.0
        for belief in aligned_beliefs:
            belief_score = BehaviorFacet._numeric_score(belief.get("score"))
            if belief_score is not None:
                best_score = max(best_score, belief_score)
        return best_score

    @staticmethod
    def _resolve_world_alignment_confidence(
        aligned_beliefs: list[dict[str, Any]],
    ) -> float:
        best_confidence = 0.0
        for belief in aligned_beliefs:
            belief_confidence = BehaviorFacet._numeric_unit_value(
                belief.get("confidence")
            )
            if belief_confidence is not None:
                best_confidence = max(best_confidence, belief_confidence)
        return best_confidence

    @staticmethod
    def _resolve_salience(
        event: Event,
        metadata: dict[str, Any],
        memory_context: dict[str, Any] | None,
        tags: list[str],
    ) -> tuple[float, str]:
        metadata_salience = BehaviorFacet._numeric_unit_value(metadata.get("salience"))
        if metadata_salience is not None:
            return metadata_salience, "event_metadata"

        if memory_context is not None:
            memory_salience = BehaviorFacet._numeric_unit_value(
                memory_context.get("salience")
            )
            if memory_salience is not None:
                return memory_salience, "memory_metadata"

        computed_salience = compute_salience(
            content=event.content,
            tags=tags,
            is_user_message=event.event_type == EventType.USER_MESSAGE,
        )
        return computed_salience, "computed"

    @staticmethod
    def _numeric_unit_value(raw_value: Any) -> float | None:
        if not isinstance(raw_value, (int, float)):
            return None
        return round(_clamp_unit(float(raw_value)), 2)

    @staticmethod
    def _numeric_score(raw_value: Any) -> float | None:
        if not isinstance(raw_value, (int, float)):
            return None
        return round(max(float(raw_value), 0.0), 3)

    @staticmethod
    def _metadata_flag(metadata: dict[str, Any], key: str) -> bool:
        raw_value = metadata.get(key)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return bool(raw_value)
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _contains_response_phrase(content: str) -> bool:
        normalized = content.casefold()
        return any(phrase in normalized for phrase in RESPONSE_PHRASES)

    @staticmethod
    def _select_decision(
        event: Event,
        signals: _BehaviorSignals,
    ) -> tuple[DecisionAction, list[str]]:
        reasons: list[str] = []
        if signals.high_priority:
            reasons.append("high_priority_tags")

        if not signals.meaningful_content and not signals.has_metadata_signal:
            reasons.append("empty_content_wait")
            return DecisionAction.WAIT, reasons

        if signals.explicit_action:
            if signals.low_risk:
                reasons.append("explicit_action_low_risk")
                return DecisionAction.ACT, reasons
            reasons.append("explicit_action_without_low_risk")
            return DecisionAction.ASK, reasons

        if signals.requires_response:
            reasons.append("requires_response_metadata")
            return DecisionAction.ASK, reasons

        if signals.uncertainty:
            reasons.append("uncertainty_metadata")
            return DecisionAction.ASK, reasons

        if signals.question_like:
            reasons.append("question_phrase_response_needed")
            return DecisionAction.ASK, reasons

        if event.event_type == EventType.USER_MESSAGE:
            reasons.append("user_message_default_record")
            return DecisionAction.RECORD, reasons

        if signals.high_priority:
            reasons.append("high_priority_record")
            return DecisionAction.RECORD, reasons

        if signals.meaningful_content and event.event_type == EventType.SYSTEM_NOTE:
            reasons.append("system_note_default_record")
            return DecisionAction.RECORD, reasons

        reasons.append("system_tick_or_idle_wait")
        return DecisionAction.WAIT, reasons

    @staticmethod
    def _score_confidence(
        action: DecisionAction,
        reasons: list[str],
        *,
        salience: float,
        goal_alignment_score: float,
        goal_alignment_priority: float,
        world_alignment_score: float,
        world_alignment_confidence: float,
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {"base": DECISION_BASE_CONFIDENCE[action]}
        for reason in reasons:
            boost = REASON_CONFIDENCE_BOOSTS.get(reason)
            if boost is not None:
                breakdown[reason] = boost

        if salience >= 0.8:
            breakdown["salience_signal"] = 0.05
        elif salience >= 0.6:
            breakdown["salience_signal"] = 0.03

        goal_boost = _goal_confidence_boost(
            goal_alignment_score=goal_alignment_score,
            goal_alignment_priority=goal_alignment_priority,
        )
        if goal_boost > 0.0:
            breakdown["goal_alignment_signal"] = goal_boost

        world_boost = _world_confidence_boost(
            world_alignment_score=world_alignment_score,
            world_alignment_confidence=world_alignment_confidence,
        )
        if world_boost > 0.0:
            breakdown["world_alignment_signal"] = world_boost

        breakdown["total"] = round(_clamp_unit(sum(breakdown.values())), 2)
        return breakdown


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))


def _goal_confidence_boost(
    *,
    goal_alignment_score: float,
    goal_alignment_priority: float,
) -> float:
    if goal_alignment_score <= 0.0 or goal_alignment_priority <= 0.0:
        return 0.0

    normalized_alignment = _clamp_unit(goal_alignment_score / 2.0)
    boost = (
        0.02
        + (0.04 * _clamp_unit(goal_alignment_priority))
        + (0.02 * normalized_alignment)
    )
    return round(min(boost, 0.08), 2)


def _world_confidence_boost(
    *,
    world_alignment_score: float,
    world_alignment_confidence: float,
) -> float:
    if world_alignment_score <= 0.0 or world_alignment_confidence <= 0.0:
        return 0.0

    normalized_alignment = _clamp_unit(world_alignment_score / 3.0)
    boost = (
        0.02
        + (0.04 * _clamp_unit(world_alignment_confidence))
        + (0.02 * normalized_alignment)
    )
    return round(min(boost, 0.08), 2)
