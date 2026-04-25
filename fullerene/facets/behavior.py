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
                "last_selected_decision": selected_decision.value,
                "last_confidence": confidence,
                "last_salience": signals.salience,
                "last_tags_considered": list(signals.tags),
                "last_reasons": list(reasons),
                "last_priority_level": priority_level,
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

        breakdown["total"] = round(_clamp_unit(sum(breakdown.values())), 2)
        return breakdown


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))
