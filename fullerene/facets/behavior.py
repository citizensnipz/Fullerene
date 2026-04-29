"""Deterministic behavior facet for Fullerene Behavior v1."""

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
    "what are you doing right now",
    "what are you doing",
    "what should i do",
    "what should i focus on",
    "should i",
    "what next",
    "next steps",
    "can you",
    "could you",
    "how do i",
    "help me",
    "tell me",
    "explain",
)
STATUS_RESPONSE_PHRASES = (
    "what are you doing right now",
    "what are you doing",
)
NEXT_STEPS_RESPONSE_PHRASES = (
    "what should i do",
    "what should i focus on",
    "what next",
    "next steps",
)
DECISION_BASE_SCORES = {
    DecisionAction.WAIT: 0.1,
    DecisionAction.RECORD: 0.4,
    DecisionAction.ASK: 0.3,
    DecisionAction.ACT: 0.2,
}
DECISION_PRIORITY = {
    DecisionAction.WAIT: 0,
    DecisionAction.RECORD: 1,
    DecisionAction.ASK: 2,
    DecisionAction.ACT: 3,
}
LOW_RETRIEVAL_THRESHOLD = 0.2
HIGH_GOAL_RELEVANCE_THRESHOLD = 0.7


@dataclass(slots=True)
class _BehaviorSignals:
    tags: list[str]
    salience: float
    salience_source: str
    meaningful_content: bool
    has_metadata_signal: bool
    question_like: bool
    requires_response: bool
    response_needed: bool
    response_reason: str | None
    response_template: str | None
    deterministic_response_available: bool
    explicit_action: bool
    low_risk: bool
    uncertainty: bool
    high_priority: bool
    pressure: float
    goal_relevance: float
    retrieval_strength: float
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
        selected_decision, reasons, decision_scores = self._select_decision(
            event,
            signals,
        )
        confidence_breakdown = self._score_confidence(
            selected_decision,
            salience=signals.salience,
            pressure=signals.pressure,
            goal_relevance=signals.goal_relevance,
            retrieval_strength=signals.retrieval_strength,
            goal_alignment_score=signals.goal_alignment_score,
            goal_alignment_priority=signals.goal_alignment_priority,
            world_alignment_score=signals.world_alignment_score,
            world_alignment_confidence=signals.world_alignment_confidence,
        )
        confidence = confidence_breakdown["total"]
        reasons.extend(self._contribution_reasons(signals, confidence_breakdown))
        priority_level = "high" if signals.high_priority else "normal"

        response_metadata = self._response_metadata(selected_decision, signals)

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
                "last_pressure": signals.pressure,
                "last_goal_relevance": signals.goal_relevance,
                "last_retrieval_strength": signals.retrieval_strength,
                "last_tags_considered": list(signals.tags),
                "last_reasons": list(reasons),
                "last_decision_scores": dict(decision_scores),
                "last_priority_level": priority_level,
                "last_goal_alignment_score": signals.goal_alignment_score,
                "last_aligned_goal_ids": list(signals.aligned_goal_ids),
                "last_world_alignment_score": signals.world_alignment_score,
                "last_aligned_belief_ids": list(signals.aligned_belief_ids),
                "last_response_needed": response_metadata["response_needed"],
                "last_response_reason": response_metadata.get("response_reason"),
                "last_response_template": response_metadata.get("response_template"),
            },
            metadata={
                "selected_decision": selected_decision.value,
                "confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "decision_scores": dict(decision_scores),
                "salience": signals.salience,
                "salience_source": signals.salience_source,
                "pressure": signals.pressure,
                "goal_relevance": signals.goal_relevance,
                "retrieval_strength": signals.retrieval_strength,
                "tags_considered": list(signals.tags),
                "reasons": list(reasons),
                "high_priority": signals.high_priority,
                "priority_level": priority_level,
                **response_metadata,
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
        memory_context = self._extract_memory_context(metadata, state)
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
        goal_relevance = self._resolve_goal_relevance(goal_context, aligned_goals)
        world_context = self._extract_world_context(metadata, state)
        aligned_beliefs = self._extract_relevant_beliefs(world_context)
        world_alignment_score = self._resolve_world_alignment_score(
            world_context,
            aligned_beliefs,
        )
        world_alignment_confidence = self._resolve_world_alignment_confidence(
            aligned_beliefs
        )
        pressure = self._numeric_unit_value(metadata.get("pressure")) or 0.0
        retrieval_strength = self._resolve_retrieval_strength(
            metadata=metadata,
            state=state,
            memory_context=memory_context,
        )
        direct_response_needed = self._contains_response_phrase(event.content)
        requires_response = self._metadata_flag(metadata, "requires_response")
        response_template = self._resolve_response_template(
            event.content,
            state=state,
            goal_context=goal_context,
        )
        deterministic_response_available = response_template in {
            "status_report",
            "next_steps_available",
        }
        response_reason = None
        if direct_response_needed:
            response_reason = "direct_question"
        elif requires_response:
            response_reason = "requires_response_metadata"

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
                or "pressure" in metadata
            ),
            question_like=direct_response_needed,
            requires_response=requires_response,
            response_needed=direct_response_needed or requires_response,
            response_reason=response_reason,
            response_template=response_template,
            deterministic_response_available=deterministic_response_available,
            explicit_action=self._metadata_flag(metadata, "explicit_action"),
            low_risk=self._metadata_flag(metadata, "low_risk"),
            uncertainty=self._metadata_flag(metadata, "uncertainty"),
            high_priority=bool(HIGH_PRIORITY_TAGS & set(tags)),
            pressure=pressure,
            goal_relevance=goal_relevance,
            retrieval_strength=retrieval_strength,
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
    def _extract_memory_context(
        metadata: dict[str, Any],
        state: NexusState,
    ) -> dict[str, Any] | None:
        for key in ("memory", "memory_facet", "stored_memory", "relevant_memory"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                return candidate

        state_memory = state.facet_state.get("memory")
        return state_memory if isinstance(state_memory, dict) else None

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
    def _resolve_goal_relevance(
        goal_context: dict[str, Any] | None,
        aligned_goals: list[dict[str, Any]],
    ) -> float:
        best_priority = BehaviorFacet._resolve_goal_alignment_priority(aligned_goals)
        if goal_context is None:
            return best_priority

        for key in ("active_goals", "last_active_goals", "goals"):
            candidate = goal_context.get(key)
            if not isinstance(candidate, list):
                continue
            for goal in candidate:
                if not isinstance(goal, dict):
                    continue
                goal_priority = BehaviorFacet._numeric_unit_value(
                    goal.get("priority")
                )
                if goal_priority is not None:
                    best_priority = max(best_priority, goal_priority)
        return best_priority

    @staticmethod
    def _resolve_retrieval_strength(
        *,
        metadata: dict[str, Any],
        state: NexusState,
        memory_context: dict[str, Any] | None,
    ) -> float:
        explicit_strength = BehaviorFacet._numeric_unit_value(
            metadata.get("retrieval_strength")
        )
        if explicit_strength is not None:
            return explicit_strength

        candidates: list[float] = []
        if memory_context is not None:
            memory_strength = BehaviorFacet._numeric_unit_value(
                memory_context.get("retrieval_strength")
            )
            if memory_strength is not None:
                candidates.append(memory_strength)
            candidates.extend(
                [
                    _normalized_count(memory_context.get("relevant_memories"), 3),
                    _normalized_count(memory_context.get("last_relevant_memory_ids"), 3),
                    _normalized_count(memory_context.get("working_memories"), 5),
                    _normalized_count(memory_context.get("last_working_memory_ids"), 5),
                ]
            )

        context_state = state.facet_state.get("context")
        if isinstance(context_state, dict):
            candidates.append(
                _normalized_count(context_state.get("last_context_item_ids"), 5)
            )
            context_item_count = BehaviorFacet._numeric_score(
                context_state.get("last_context_item_count")
            )
            if context_item_count is not None:
                candidates.append(_clamp_unit(context_item_count / 5.0))

        attention_state = state.facet_state.get("attention")
        if isinstance(attention_state, dict):
            candidates.append(
                BehaviorFacet._attention_memory_strength(attention_state)
            )

        return round(max(candidates, default=0.0), 3)

    @staticmethod
    def _attention_memory_strength(attention_state: dict[str, Any]) -> float:
        raw_result = attention_state.get("last_attention_result")
        if not isinstance(raw_result, dict):
            return 0.0
        focus_items = raw_result.get("focus_items")
        if not isinstance(focus_items, list):
            return 0.0

        memory_scores: list[float] = []
        for item in focus_items:
            if not isinstance(item, dict) or item.get("source") != "memory":
                continue
            score = BehaviorFacet._numeric_unit_value(item.get("score"))
            if score is not None:
                memory_scores.append(score)
        return max(memory_scores, default=0.0)

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
        stripped = content.strip()
        if stripped.endswith("?"):
            return True
        normalized = _normalize_content(content)
        return any(phrase in normalized for phrase in RESPONSE_PHRASES)

    @staticmethod
    def _resolve_response_template(
        content: str,
        *,
        state: NexusState,
        goal_context: dict[str, Any] | None,
    ) -> str | None:
        normalized = _normalize_content(content)
        if any(phrase in normalized for phrase in STATUS_RESPONSE_PHRASES):
            return "status_report"
        if any(phrase in normalized for phrase in NEXT_STEPS_RESPONSE_PHRASES):
            if BehaviorFacet._has_next_steps_context(state, goal_context):
                return "next_steps_available"
            return "clarification_needed"
        if BehaviorFacet._contains_response_phrase(content):
            return "clarification_needed"
        return None

    @staticmethod
    def _has_next_steps_context(
        state: NexusState,
        goal_context: dict[str, Any] | None,
    ) -> bool:
        planner_state = state.facet_state.get("planner")
        if isinstance(planner_state, dict):
            last_plan = planner_state.get("last_plan")
            if isinstance(last_plan, dict) and isinstance(last_plan.get("steps"), list):
                return bool(last_plan["steps"])
        if goal_context is not None:
            for key in (
                "last_relevant_goals",
                "relevant_goals",
                "last_active_goals",
                "active_goals",
                "goals",
            ):
                if isinstance(goal_context.get(key), list) and goal_context[key]:
                    return True
        return False

    @staticmethod
    def _response_metadata(
        selected_decision: DecisionAction,
        signals: _BehaviorSignals,
    ) -> dict[str, Any]:
        response_needed = signals.response_needed
        response_reason = signals.response_reason
        response_template = signals.response_template

        if selected_decision == DecisionAction.ASK:
            response_needed = True
            response_reason = response_reason or "clarification_needed"
            response_template = "clarification_needed"
        elif selected_decision == DecisionAction.ACT and response_needed:
            response_template = response_template or "clarification_needed"
        else:
            return {
                "response_needed": response_needed,
                "response_reason": response_reason,
                "response_template": response_template,
            }

        metadata: dict[str, Any] = {
            "response_needed": response_needed,
            "response_reason": response_reason,
            "response_template": response_template,
        }
        if selected_decision in {DecisionAction.ASK, DecisionAction.ACT}:
            metadata.update(
                {
                    "output_type": "text",
                    "tool": "text",
                }
            )
        return metadata

    @staticmethod
    def _select_decision(
        event: Event,
        signals: _BehaviorSignals,
    ) -> tuple[DecisionAction, list[str], dict[str, float]]:
        reasons: list[str] = []
        score_breakdown: dict[DecisionAction, dict[str, float]] = {
            action: {"base": base_score}
            for action, base_score in DECISION_BASE_SCORES.items()
        }

        def add(action: DecisionAction, reason: str, value: float) -> None:
            if value == 0.0:
                return
            score_breakdown[action][reason] = (
                score_breakdown[action].get(reason, 0.0) + value
            )

        if signals.high_priority:
            reasons.append("high_priority_tags")
            add(DecisionAction.RECORD, "high_priority_record_bias", 0.15)
            add(DecisionAction.ASK, "high_priority_ask_bias", 0.05)

        if not signals.meaningful_content and not signals.has_metadata_signal:
            reasons.append("empty_content_wait")
            add(DecisionAction.WAIT, "empty_content_wait", 0.55)

        if signals.explicit_action:
            if signals.low_risk:
                reasons.append("explicit_action_low_risk")
                add(DecisionAction.ACT, "explicit_action_low_risk", 0.45)
            else:
                reasons.append("explicit_action_without_low_risk")
                add(DecisionAction.ASK, "explicit_action_without_low_risk", 0.45)

        if signals.requires_response:
            reasons.append("requires_response_metadata")
            add(DecisionAction.ASK, "requires_response_metadata", 0.4)

        if signals.response_needed:
            if signals.deterministic_response_available:
                reasons.append("deterministic_text_response_available")
                add(DecisionAction.ACT, "deterministic_text_response_available", 0.65)
            else:
                reasons.append("response_needed_low_context")
                add(DecisionAction.ASK, "response_needed_low_context", 0.25)

        if signals.uncertainty:
            reasons.append("uncertainty_metadata")
            add(DecisionAction.ASK, "uncertainty_metadata", 0.35)

        if signals.question_like:
            reasons.append("question_phrase_response_needed")
            add(DecisionAction.ASK, "question_phrase_response_needed", 0.3)

        if event.event_type == EventType.USER_MESSAGE:
            reasons.append("user_message_default_record")
            add(DecisionAction.RECORD, "user_message_default_record", 0.15)

        if signals.high_priority:
            reasons.append("high_priority_record")
            add(DecisionAction.RECORD, "high_priority_record", 0.05)

        if signals.meaningful_content and event.event_type == EventType.SYSTEM_NOTE:
            reasons.append("system_note_default_record")
            add(DecisionAction.RECORD, "system_note_default_record", 0.15)

        if event.event_type == EventType.SYSTEM_TICK:
            reasons.append("system_tick_or_idle_wait")
            add(DecisionAction.WAIT, "system_tick_or_idle_wait", 0.3)

        BehaviorFacet._apply_pressure_biases(score_breakdown, reasons, signals)
        BehaviorFacet._apply_goal_biases(score_breakdown, reasons, signals)
        BehaviorFacet._apply_memory_biases(score_breakdown, reasons, signals)
        BehaviorFacet._apply_low_signal_bias(score_breakdown, reasons, signals)

        decision_scores = {
            action: round(_clamp_unit(sum(breakdown.values())), 3)
            for action, breakdown in score_breakdown.items()
        }
        selected_decision = max(
            DECISION_BASE_SCORES,
            key=lambda action: (decision_scores[action], DECISION_PRIORITY[action]),
        )
        reasons.append(f"selected_highest_weighted_score:{selected_decision.value}")
        return (
            selected_decision,
            reasons,
            {action.value: decision_scores[action] for action in DECISION_BASE_SCORES},
        )

    @staticmethod
    def _apply_pressure_biases(
        score_breakdown: dict[DecisionAction, dict[str, float]],
        reasons: list[str],
        signals: _BehaviorSignals,
    ) -> None:
        pressure = signals.pressure
        if pressure <= 0.0:
            reasons.append("pressure contribution: 0.000 no pressure bias applied")
            return

        actionable = signals.explicit_action and signals.low_risk
        unclear = BehaviorFacet._is_unclear(signals)
        score_breakdown[DecisionAction.ACT]["pressure_act_bias"] = pressure * (
            0.3 if actionable else 0.15
        )
        score_breakdown[DecisionAction.ASK]["pressure_ask_bias"] = pressure * (
            0.25 if unclear else 0.1
        )
        score_breakdown[DecisionAction.WAIT]["pressure_wait_penalty"] = pressure * -0.2
        reasons.append("high pressure increased ACT score")
        reasons.append("high pressure increased ASK score")
        reasons.append("pressure reduced WAIT score")

    @staticmethod
    def _apply_goal_biases(
        score_breakdown: dict[DecisionAction, dict[str, float]],
        reasons: list[str],
        signals: _BehaviorSignals,
    ) -> None:
        goal_relevance = signals.goal_relevance
        if goal_relevance <= 0.0:
            reasons.append("goal relevance contribution: 0.000 no goal bias applied")
            return

        actionable = signals.explicit_action and signals.low_risk
        unclear = BehaviorFacet._is_unclear(signals)
        if actionable:
            score_breakdown[DecisionAction.ACT]["goal_relevance_act_bias"] = (
                goal_relevance * 0.35
            )
            reasons.append("goal priority boosted ACT score")
        if signals.deterministic_response_available:
            score_breakdown[DecisionAction.ACT]["goal_response_act_bias"] = (
                goal_relevance * 0.25
            )
            reasons.append("goal context supported deterministic ACT response")
        if unclear and not signals.deterministic_response_available:
            score_breakdown[DecisionAction.ASK]["goal_relevance_ask_bias"] = (
                goal_relevance * 0.25
            )
            reasons.append("goal priority boosted ASK score")
        if (
            goal_relevance >= HIGH_GOAL_RELEVANCE_THRESHOLD
            and signals.retrieval_strength < LOW_RETRIEVAL_THRESHOLD
            and not signals.deterministic_response_available
        ):
            score_breakdown[DecisionAction.ASK]["goal_relevant_low_context_ask_bias"] = (
                goal_relevance * (1.0 - signals.retrieval_strength) * 0.35
            )
            reasons.append("goal relevant but insufficient context")
            reasons.append("low retrieval caused ASK preference")
        if goal_relevance < LOW_RETRIEVAL_THRESHOLD:
            score_breakdown[DecisionAction.RECORD]["low_goal_relevance_record_bias"] = 0.08
            score_breakdown[DecisionAction.WAIT]["low_goal_relevance_wait_bias"] = 0.04
            reasons.append("low goal relevance favored RECORD or WAIT")
        reasons.append("goal priority boosted decision confidence")

    @staticmethod
    def _apply_memory_biases(
        score_breakdown: dict[DecisionAction, dict[str, float]],
        reasons: list[str],
        signals: _BehaviorSignals,
    ) -> None:
        retrieval_strength = signals.retrieval_strength
        if retrieval_strength <= 0.0:
            reasons.append("memory contribution: 0.000 no retrieval bias applied")
            return

        if signals.explicit_action and signals.low_risk:
            score_breakdown[DecisionAction.ACT]["memory_retrieval_act_bias"] = (
                retrieval_strength * 0.1
            )
        if BehaviorFacet._is_unclear(signals):
            score_breakdown[DecisionAction.ASK]["memory_retrieval_ask_bias"] = (
                retrieval_strength * 0.1
            )
        reasons.append("memory retrieval strength increased decision confidence")

    @staticmethod
    def _apply_low_signal_bias(
        score_breakdown: dict[DecisionAction, dict[str, float]],
        reasons: list[str],
        signals: _BehaviorSignals,
    ) -> None:
        low_signal = (
            signals.pressure <= 0.0
            and signals.goal_relevance <= 0.0
            and signals.retrieval_strength <= 0.0
            and signals.salience < 0.6
            and not signals.explicit_action
            and not BehaviorFacet._is_unclear(signals)
        )
        if not low_signal:
            return
        if signals.meaningful_content:
            score_breakdown[DecisionAction.RECORD]["low_signal_record_bias"] = 0.1
            reasons.append("low signal environment favored RECORD")
        else:
            score_breakdown[DecisionAction.WAIT]["low_signal_wait_bias"] = 0.15
            reasons.append("low signal environment favored WAIT")

    @staticmethod
    def _score_confidence(
        action: DecisionAction,
        *,
        salience: float,
        pressure: float,
        goal_relevance: float,
        retrieval_strength: float,
        goal_alignment_score: float,
        goal_alignment_priority: float,
        world_alignment_score: float,
        world_alignment_confidence: float,
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {
            "base": DECISION_BASE_SCORES[action],
            "pressure_contribution": round(_clamp_unit(pressure) * 0.25, 3),
            "goal_relevance_contribution": round(
                _clamp_unit(goal_relevance) * 0.30,
                3,
            ),
            "memory_retrieval_contribution": round(
                _clamp_unit(retrieval_strength) * 0.20,
                3,
            ),
            "salience_contribution": round(_clamp_unit(salience) * 0.25, 3),
        }
        if goal_alignment_score > 0.0 and goal_alignment_priority > 0.0:
            breakdown["goal_alignment_signal"] = breakdown[
                "goal_relevance_contribution"
            ]
        world_boost = _world_confidence_boost(
            world_alignment_score=world_alignment_score,
            world_alignment_confidence=world_alignment_confidence,
        )
        if world_boost > 0.0:
            breakdown["world_alignment_signal"] = world_boost

        total_keys = (
            "base",
            "pressure_contribution",
            "goal_relevance_contribution",
            "memory_retrieval_contribution",
            "salience_contribution",
            "world_alignment_signal",
        )
        breakdown["total"] = round(
            _clamp_unit(sum(breakdown.get(key, 0.0) for key in total_keys)),
            3,
        )
        return breakdown

    @staticmethod
    def _contribution_reasons(
        signals: _BehaviorSignals,
        confidence_breakdown: dict[str, float],
    ) -> list[str]:
        return [
            (
                f"pressure contribution: {signals.pressure:.3f} -> "
                f"{confidence_breakdown['pressure_contribution']:.3f}"
            ),
            (
                f"goal relevance contribution: {signals.goal_relevance:.3f} -> "
                f"{confidence_breakdown['goal_relevance_contribution']:.3f}"
            ),
            (
                f"memory contribution: {signals.retrieval_strength:.3f} -> "
                f"{confidence_breakdown['memory_retrieval_contribution']:.3f}"
            ),
            (
                "final confidence breakdown: "
                f"base={confidence_breakdown['base']:.3f}, "
                f"pressure={confidence_breakdown['pressure_contribution']:.3f}, "
                f"goal={confidence_breakdown['goal_relevance_contribution']:.3f}, "
                f"memory={confidence_breakdown['memory_retrieval_contribution']:.3f}, "
                f"salience={confidence_breakdown['salience_contribution']:.3f}, "
                f"total={confidence_breakdown['total']:.3f}"
            ),
        ]

    @staticmethod
    def _is_unclear(signals: _BehaviorSignals) -> bool:
        return (
            signals.requires_response
            or signals.response_needed
            or signals.uncertainty
            or signals.question_like
            or (signals.explicit_action and not signals.low_risk)
        )


def _clamp_unit(score: float) -> float:
    return max(0.0, min(float(score), 1.0))


def _normalize_content(content: str) -> str:
    return " ".join(content.casefold().split())


def _normalized_count(raw_value: Any, denominator: int) -> float:
    if isinstance(raw_value, list):
        return _clamp_unit(len(raw_value) / max(float(denominator), 1.0))
    if isinstance(raw_value, (int, float)):
        return _clamp_unit(float(raw_value) / max(float(denominator), 1.0))
    return 0.0


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
    return round(min(boost, 0.08), 3)
