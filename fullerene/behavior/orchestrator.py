"""Deterministic behavior facet for Fullerene Behavior v2.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fullerene.memory import (
    compute_salience,
    infer_domain,
    infer_tags,
    merge_tags,
    normalize_tags,
    tokenize,
)
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
    "what is happening",
    "what do you know",
    "what should i do",
    "what book should i",
    "what should i focus on",
    "what should we",
    "recommend",
    "suggest",
    "should i",
    "what next",
    "next steps",
    "plan",
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
    "what is happening",
    "what do you know",
)
MEMORY_SUMMARY_RESPONSE_PHRASES = (
    "what do you know about me",
    "what do you remember about me",
)
FACTUAL_RESPONSE_PHRASES = (
    "what is",
    "what are",
    "who is",
    "who are",
    "what do you know",
    "tell me about",
    "explain",
    "define",
    "describe",
)
RECOMMENDATION_RESPONSE_PHRASES = (
    "what should i",
    "what book should i",
    "what should we",
    "recommend",
    "suggest",
    "this weekend",
)
PLANNING_RESPONSE_PHRASES = (
    "what next",
    "next steps",
    "how should i",
    "how should we",
    "plan",
)
NEXT_STEPS_RESPONSE_PHRASES = (
    "what should i do",
    "what should i focus on",
    "what next",
    "next steps",
)
VAGUE_RESPONSE_PHRASES = (
    "help me",
    "can you",
    "could you",
    "tell me",
    "explain",
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
CONTEXT_SUFFICIENCY_THRESHOLD = 1.0
LOW_AMBIGUITY_THRESHOLD = 0.35
HIGH_AMBIGUITY_THRESHOLD = 0.65
RELEVANT_MEMORY_STRENGTH_THRESHOLD = 0.35
GOAL_RELEVANCE_THRESHOLD = 0.35
QUERY_INTENTS_REQUIRING_RESPONSE = frozenset(
    {"recommendation", "planning", "factual", "memory_summary"}
)
LOW_BELIEF_CONFIDENCE_THRESHOLD = 0.4
CONTRADICTION_ACT_PENALTY = 0.35
CONTEXT_OVERLOAD_RATIO_THRESHOLD = 0.85
FOLLOW_UP_REFERENCE_WORDS = frozenset(
    {
        "that",
        "this",
        "it",
        "there",
        "those",
        "them",
        "one",
        "ones",
        "then",
        "again",
        "where",
        "why",
        "how",
    }
)
CONVERSATIONAL_INTENTS = frozenset(
    {
        "new_question",
        "follow_up",
        "clarification_request",
        "clarification_supplied",
        "challenge",
        "source_request",
        "correction",
        "contradiction_report",
        "repeated_dissatisfaction",
        "action_request",
        "planning_request",
        "memory_update",
        "status_request",
        "unknown",
    }
)
GROUNDING_NEEDS = frozenset(
    {
        "none",
        "memory",
        "working_memory",
        "world_model",
        "policy",
        "executor",
        "verifier",
        "runtime_state",
        "unknown",
    }
)


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
    query_intent: str | None
    ambiguity_score: float
    has_relevant_memory: bool
    has_preference_memory: bool
    relevant_memory_strength: float
    has_goal: bool
    top_goal_priority: float
    goal_signal_strength: float
    domain_match: bool
    event_domain: str | None
    included_memory_roles: list[str]
    included_memory_domains: list[str]
    active_goal_count: int
    relevant_goal_count: int
    relevant_memory_count: int
    relevant_belief_count: int
    context_item_count: int
    planner_available: bool
    context_sufficiency: float
    missing_context: list[str]
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
    belief_confidence: float
    belief_contradiction: bool
    belief_reason: str | None
    policy_result: str
    policy_requires_approval: bool
    policy_blocks_act: bool
    policy_reason: str | None
    context_item_count_signal: int
    context_max_items_signal: int
    context_load_ratio: float
    context_overloaded: bool
    latent_pressure: float
    conversational_intent: str
    conversational_intent_score: float
    conversational_intent_reasons: list[str]
    follow_up_reference_detected: bool
    short_follow_up: bool
    grounding_need: str
    grounding_need_reasons: list[str]
    grounding_available: bool
    grounding_confidence: float
    continuity_confidence: float
    self_consistency_confidence: float
    challenge_confidence_penalty: float
    ambiguity_kind: str
    ambiguity_reasons: list[str]
    repeated_dissatisfaction: bool
    included_working_memory_turns: list[str]
    working_memory_turn_count: int
    included_context_types: list[str]
    included_lpb_entry_ids: list[str]
    included_belief_ids: list[str]
    context_strategy: str | None
    related_context_item_ids: list[str]
    related_memory_ids: list[str]
    related_belief_ids: list[str]


class BehaviorFacet:
    """Choose an inspectable Nexus decision from deterministic rules only."""

    name = "behavior"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        signals = self._collect_signals(event, state)
        selected_decision, reasons, decision_scores = self._select_decision(
            event,
            signals,
        )
        raw_candidate_scores = dict(decision_scores)
        decision_scores = self._apply_v2_candidate_adjustments(decision_scores, signals, reasons)
        decision_scores, adjustment_reasons = self._apply_v21_candidate_adjustments(
            decision_scores,
            signals,
        )
        reasons.extend(adjustment_reasons)
        selected_decision = self._select_highest_scored_decision(decision_scores)
        if (
            selected_decision == DecisionAction.ACT
            and signals.belief_confidence > 0.0
            and signals.belief_confidence < LOW_BELIEF_CONFIDENCE_THRESHOLD
        ):
            selected_decision = DecisionAction.ASK
            reasons.append("belief_guardrail:act_to_ask_low_confidence")
        selected_decision, downgrade_reasons = self._apply_policy_downgrade(
            selected_decision,
            decision_scores,
            signals,
        )
        reasons.extend(downgrade_reasons)
        confidence_breakdown = self._score_confidence(
            selected_decision,
            pressure=signals.pressure,
            memory_signal_strength=signals.relevant_memory_strength,
            goal_signal_strength=signals.goal_signal_strength,
            ambiguity_score=signals.ambiguity_score,
            goal_alignment_score=signals.goal_alignment_score,
            goal_alignment_priority=signals.goal_alignment_priority,
            world_alignment_score=signals.world_alignment_score,
            world_alignment_confidence=signals.world_alignment_confidence,
        )
        confidence_breakdown["grounding_confidence"] = signals.grounding_confidence
        confidence_breakdown["continuity_confidence"] = signals.continuity_confidence
        confidence_breakdown["self_consistency_confidence"] = (
            signals.self_consistency_confidence
        )
        confidence_breakdown["challenge_confidence_penalty"] = (
            -signals.challenge_confidence_penalty
        )
        confidence_breakdown["total"] = round(
            _clamp_unit(
                confidence_breakdown["total"]
                + (signals.grounding_confidence * 0.15)
                + (signals.continuity_confidence * 0.1)
                + (signals.self_consistency_confidence * 0.1)
                - signals.challenge_confidence_penalty
            ),
            3,
        )
        attention_confidence_bias = self._attention_confidence_bias(
            state,
            response_needed=signals.response_needed,
        )
        if attention_confidence_bias > 0.0:
            confidence_breakdown["attention_broadcast_contribution"] = (
                attention_confidence_bias
            )
            confidence_breakdown["total"] = round(
                _clamp_unit(confidence_breakdown["total"] + attention_confidence_bias),
                3,
            )
        confidence = confidence_breakdown["total"]
        if signals.context_overloaded and selected_decision == DecisionAction.ACT:
            confidence = round(_clamp_unit(confidence - 0.1), 3)
            confidence_breakdown["context_overload_penalty"] = -0.1
            confidence_breakdown["total"] = confidence
            reasons.append("context_overload lowered ACT confidence")
        reasons.extend(self._contribution_reasons(signals, confidence_breakdown))
        if attention_confidence_bias > 0.0:
            reasons.append("top_down_attention_broadcast increased response confidence")
        priority_level = "high" if signals.high_priority else "normal"
        interrupt_recommended, interrupt_reason = self._interrupt_recommendation(signals)

        response_metadata = self._response_metadata(selected_decision, signals)
        trace = self._build_decision_trace(
            event=event,
            signals=signals,
            raw_candidate_scores=raw_candidate_scores,
            adjusted_candidate_scores=decision_scores,
            selected_decision=selected_decision,
            confidence=confidence,
            reasons=reasons,
            interrupt_recommended=interrupt_recommended,
            interrupt_reason=interrupt_reason,
        )

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
                "last_ambiguity_score": signals.ambiguity_score,
                "last_has_relevant_memory": signals.has_relevant_memory,
                "last_has_goal": signals.has_goal,
                "last_memory_signal_strength": signals.relevant_memory_strength,
                "last_goal_signal_strength": signals.goal_signal_strength,
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
                "last_query_intent": signals.query_intent,
                "last_context_sufficiency": signals.context_sufficiency,
                "last_missing_context": list(signals.missing_context),
                "last_attention_confidence_bias": attention_confidence_bias,
                "last_policy_result": signals.policy_result,
                "last_belief_confidence": signals.belief_confidence,
                "last_belief_contradiction": signals.belief_contradiction,
                "last_context_load_ratio": signals.context_load_ratio,
                "last_latent_pressure": signals.latent_pressure,
                "last_interrupt_recommended": interrupt_recommended,
                "last_interrupt_reason": interrupt_reason,
                "last_decision_trace": trace,
            },
            metadata={
                "selected_decision": selected_decision.value,
                "confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "confidence_components": confidence_breakdown,
                "decision_scores": dict(decision_scores),
                "salience": signals.salience,
                "salience_source": signals.salience_source,
                "pressure": signals.pressure,
                "goal_relevance": signals.goal_relevance,
                "retrieval_strength": signals.retrieval_strength,
                "tags_considered": list(signals.tags),
                "reasons": list(reasons),
                "decision": selected_decision.value,
                "ambiguity_score": signals.ambiguity_score,
                "ambiguity_kind": signals.ambiguity_kind,
                "ambiguity_reasons": list(signals.ambiguity_reasons),
                "has_relevant_memory": signals.has_relevant_memory,
                "has_preference_memory": signals.has_preference_memory,
                "has_goal": signals.has_goal,
                "memory_signal_strength": signals.relevant_memory_strength,
                "goal_signal_strength": signals.goal_signal_strength,
                "top_goal_priority": signals.top_goal_priority,
                "domain_match": signals.domain_match,
                "event_domain": signals.event_domain,
                "high_priority": signals.high_priority,
                "priority_level": priority_level,
                **response_metadata,
                "query_intent": signals.query_intent,
                "active_goal_count": signals.active_goal_count,
                "relevant_goal_count": signals.relevant_goal_count,
                "relevant_memory_count": signals.relevant_memory_count,
                "relevant_belief_count": signals.relevant_belief_count,
                "context_item_count": signals.context_item_count,
                "planner_available": signals.planner_available,
                "context_sufficiency": signals.context_sufficiency,
                "missing_context": list(signals.missing_context),
                "conversational_intent": signals.conversational_intent,
                "conversational_intent_score": signals.conversational_intent_score,
                "conversational_intent_reasons": list(
                    signals.conversational_intent_reasons
                ),
                "follow_up_reference_detected": signals.follow_up_reference_detected,
                "short_follow_up": signals.short_follow_up,
                "grounding_need": signals.grounding_need,
                "grounding_need_reasons": list(signals.grounding_need_reasons),
                "grounding_available": signals.grounding_available,
                "grounding_confidence": signals.grounding_confidence,
                "continuity_confidence": signals.continuity_confidence,
                "self_consistency_confidence": signals.self_consistency_confidence,
                "challenge_confidence_penalty": signals.challenge_confidence_penalty,
                "final_confidence_reasons": self._final_confidence_reasons(
                    signals,
                    confidence_breakdown,
                ),
                "included_memory_roles": list(signals.included_memory_roles),
                "included_memory_domains": list(signals.included_memory_domains),
                "attention_confidence_bias": attention_confidence_bias,
                "memory_signal_available": signals.memory_signal_available,
                "goal_signal_available": signals.goal_signal_available,
                "goal_alignment_score": signals.goal_alignment_score,
                "goal_alignment_priority": signals.goal_alignment_priority,
                "aligned_goal_ids": list(signals.aligned_goal_ids),
                "world_signal_available": signals.world_signal_available,
                "world_alignment_score": signals.world_alignment_score,
                "world_alignment_confidence": signals.world_alignment_confidence,
                "aligned_belief_ids": list(signals.aligned_belief_ids),
                "belief_confidence": signals.belief_confidence,
                "belief_contradiction": signals.belief_contradiction,
                "belief_reason": signals.belief_reason,
                "policy_result": signals.policy_result,
                "policy_requires_approval": signals.policy_requires_approval,
                "policy_blocks_act": signals.policy_blocks_act,
                "policy_reason": signals.policy_reason,
                "context_load": {
                    "item_count": signals.context_item_count_signal,
                    "max_items": signals.context_max_items_signal,
                    "load_ratio": signals.context_load_ratio,
                    "overloaded": signals.context_overloaded,
                },
                "latent_pressure": signals.latent_pressure,
                "interrupt_recommended": interrupt_recommended,
                "interrupt_reason": interrupt_reason,
                "decision_trace": trace,
                "learning_event": {
                    "event_type": "behavior_decision_trace_v2",
                    "trace": trace,
                    "signals": self._learning_signals(signals),
                    "conversational_intent": signals.conversational_intent,
                    "grounding_need": signals.grounding_need,
                    "grounding_available": signals.grounding_available,
                    "ambiguity_kind": signals.ambiguity_kind,
                    "confidence": confidence,
                    "related_context_item_ids": list(signals.related_context_item_ids),
                    "related_belief_ids": list(signals.related_belief_ids),
                    "related_memory_ids": list(signals.related_memory_ids),
                },
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
        belief_confidence, belief_contradiction, belief_reason = (
            self._resolve_belief_signal(aligned_beliefs)
        )
        policy_result, policy_requires_approval, policy_blocks_act, policy_reason = (
            self._extract_policy_signal(metadata, state)
        )
        planner_context = self._extract_planner_context(state)
        context_signal = self._extract_context_signal(metadata, state)
        (
            context_item_count_signal,
            context_max_items_signal,
            context_load_ratio,
            context_overloaded,
        ) = self._extract_context_load_signal(context_signal, state)
        query_intent = self._resolve_query_intent(
            event.content,
            metadata=metadata,
            context_signal=context_signal,
            memory_context=memory_context,
        )
        active_goal_count = self._active_goal_count(goal_context)
        relevant_goal_count = len(aligned_goals)
        active_goal_count = max(
            active_goal_count,
            self._context_item_type_count(context_signal, "goal"),
        )
        relevant_memory_count = max(
            self._relevant_memory_count(memory_context),
            self._context_relevant_memory_count(context_signal),
        )
        relevant_belief_count = len(aligned_beliefs)
        context_item_count = self._context_item_count(state)
        planner_available = self._planner_available(planner_context)
        included_context_types = self._included_context_types(context_signal)
        included_working_memory_turns = self._included_working_memory_turns(context_signal)
        working_memory_turn_count = self._working_memory_turn_count(context_signal)
        included_lpb_entry_ids = self._included_lpb_entry_ids(context_signal)
        included_belief_ids = self._included_belief_ids(context_signal)
        context_strategy = self._context_strategy(context_signal)
        included_memory_roles = self._included_memory_roles(
            context_signal=context_signal,
            memory_context=memory_context,
        )
        included_memory_domains = self._included_memory_domains(
            context_signal=context_signal,
            memory_context=memory_context,
        )
        event_domain = self._resolve_event_domain(
            event,
            context_signal=context_signal,
            memory_context=memory_context,
        )
        has_preference_memory = "preference" in included_memory_roles
        context_sufficiency = self._context_sufficiency(
            query_intent=query_intent,
            active_goal_count=active_goal_count,
            relevant_goal_count=relevant_goal_count,
            relevant_memory_count=relevant_memory_count,
            relevant_belief_count=relevant_belief_count,
            planner_available=planner_available,
        )
        missing_context = self._missing_context(
            query_intent=query_intent,
            active_goal_count=active_goal_count,
            relevant_goal_count=relevant_goal_count,
            relevant_memory_count=relevant_memory_count,
            relevant_belief_count=relevant_belief_count,
            planner_available=planner_available,
        )
        pressure = self._numeric_unit_value(metadata.get("pressure")) or 0.0
        latent_pressure = self._extract_latent_pressure(metadata, state)
        retrieval_strength = self._resolve_retrieval_strength(
            metadata=metadata,
            state=state,
            memory_context=memory_context,
        )
        relevant_memory_strength = self._resolve_relevant_memory_strength(
            retrieval_strength=retrieval_strength,
            context_signal=context_signal,
            memory_context=memory_context,
            relevant_memory_count=relevant_memory_count,
            has_explicit_strength="retrieval_strength" in metadata,
            has_preference_memory=has_preference_memory,
            query_intent=query_intent,
        )
        has_relevant_memory = (
            relevant_memory_count > 0
            or relevant_belief_count > 0
            or relevant_memory_strength >= RELEVANT_MEMORY_STRENGTH_THRESHOLD
        )
        top_goal_priority = self._resolve_top_goal_priority(
            goal_context=goal_context,
            context_signal=context_signal,
            aligned_goals=aligned_goals,
        )
        goal_signal_strength = self._resolve_goal_signal_strength(
            query_intent=query_intent,
            active_goal_count=active_goal_count,
            relevant_goal_count=relevant_goal_count,
            top_goal_priority=top_goal_priority,
            goal_alignment_score=goal_alignment_score,
            context_signal=context_signal,
        )
        has_goal = active_goal_count > 0 or top_goal_priority > 0.0
        goal_relevance = max(goal_relevance, goal_signal_strength)
        domain_match = self._has_domain_match(
            event_domain=event_domain,
            included_memory_domains=included_memory_domains,
            context_signal=context_signal,
        )
        ambiguity_score = self._compute_ambiguity_score(
            event.content,
            query_intent=query_intent,
            has_relevant_memory=has_relevant_memory,
            relevant_memory_strength=relevant_memory_strength,
            has_preference_memory=has_preference_memory,
            has_goal=has_goal,
            goal_signal_strength=goal_signal_strength,
            relevant_belief_count=relevant_belief_count,
            domain_match=domain_match,
            event_domain=event_domain,
        )
        (
            follow_up_reference_detected,
            short_follow_up,
            conversational_intent,
            conversational_intent_score,
            conversational_intent_reasons,
        ) = self._classify_conversational_intent(
            event,
            query_intent=query_intent,
            working_memory_turn_count=working_memory_turn_count,
            has_recent_assistant_output=working_memory_turn_count > 0,
            has_previous_user_turn=working_memory_turn_count > 1,
            has_context_items=context_item_count > 0,
        )
        grounding_need, grounding_need_reasons = self._classify_grounding_need(
            conversational_intent
        )
        grounding_available, grounding_confidence = self._resolve_grounding(
            grounding_need=grounding_need,
            working_memory_turn_count=working_memory_turn_count,
            relevant_memory_count=relevant_memory_count,
            relevant_belief_count=relevant_belief_count,
            policy_result=policy_result,
            planner_available=planner_available,
            context_item_count=context_item_count,
        )
        ambiguity_kind, ambiguity_reasons = self._classify_ambiguity_kind(
            ambiguity_score=ambiguity_score,
            conversational_intent=conversational_intent,
            follow_up_reference_detected=follow_up_reference_detected,
            short_follow_up=short_follow_up,
            grounding_available=grounding_available,
            working_memory_turn_count=working_memory_turn_count,
        )
        repeated_dissatisfaction = conversational_intent == "repeated_dissatisfaction"
        continuity_confidence = self._continuity_confidence(
            follow_up_reference_detected=follow_up_reference_detected,
            working_memory_turn_count=working_memory_turn_count,
            short_follow_up=short_follow_up,
        )
        self_consistency_confidence = self._self_consistency_confidence(
            belief_confidence=belief_confidence,
            belief_contradiction=belief_contradiction,
        )
        challenge_confidence_penalty = self._challenge_penalty(
            conversational_intent=conversational_intent,
            repeated_dissatisfaction=repeated_dissatisfaction,
            belief_contradiction=belief_contradiction,
            grounding_available=grounding_available,
        )
        direct_response_needed = (
            query_intent in QUERY_INTENTS_REQUIRING_RESPONSE
            or self._contains_response_phrase(event.content)
        )
        requires_response = self._metadata_flag(metadata, "requires_response")
        response_needed = direct_response_needed or requires_response
        if (
            response_needed
            and query_intent == "unknown"
            and context_sufficiency < self._sufficiency_threshold(query_intent)
        ):
            missing_context = self._missing_context(
                query_intent=query_intent,
                active_goal_count=active_goal_count,
                relevant_goal_count=relevant_goal_count,
                relevant_memory_count=relevant_memory_count,
                relevant_belief_count=relevant_belief_count,
                planner_available=planner_available,
            )
        response_template = self._resolve_response_template(
            event.content,
            query_intent=query_intent,
            context_sufficiency=context_sufficiency,
            missing_context=missing_context,
        )
        deterministic_response_available = (
            response_needed
            and context_sufficiency >= self._sufficiency_threshold(query_intent)
        )
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
            response_needed=response_needed,
            response_reason=response_reason,
            response_template=response_template,
            deterministic_response_available=deterministic_response_available,
            query_intent=query_intent,
            ambiguity_score=ambiguity_score,
            has_relevant_memory=has_relevant_memory,
            has_preference_memory=has_preference_memory,
            relevant_memory_strength=relevant_memory_strength,
            has_goal=has_goal,
            top_goal_priority=top_goal_priority,
            goal_signal_strength=goal_signal_strength,
            domain_match=domain_match,
            event_domain=event_domain,
            included_memory_roles=included_memory_roles,
            included_memory_domains=included_memory_domains,
            active_goal_count=active_goal_count,
            relevant_goal_count=relevant_goal_count,
            relevant_memory_count=relevant_memory_count,
            relevant_belief_count=relevant_belief_count,
            context_item_count=context_item_count,
            planner_available=planner_available,
            context_sufficiency=context_sufficiency,
            missing_context=missing_context,
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
            belief_confidence=belief_confidence,
            belief_contradiction=belief_contradiction,
            belief_reason=belief_reason,
            policy_result=policy_result,
            policy_requires_approval=policy_requires_approval,
            policy_blocks_act=policy_blocks_act,
            policy_reason=policy_reason,
            context_item_count_signal=context_item_count_signal,
            context_max_items_signal=context_max_items_signal,
            context_load_ratio=context_load_ratio,
            context_overloaded=context_overloaded,
            latent_pressure=latent_pressure,
            conversational_intent=conversational_intent,
            conversational_intent_score=conversational_intent_score,
            conversational_intent_reasons=conversational_intent_reasons,
            follow_up_reference_detected=follow_up_reference_detected,
            short_follow_up=short_follow_up,
            grounding_need=grounding_need,
            grounding_need_reasons=grounding_need_reasons,
            grounding_available=grounding_available,
            grounding_confidence=grounding_confidence,
            continuity_confidence=continuity_confidence,
            self_consistency_confidence=self_consistency_confidence,
            challenge_confidence_penalty=challenge_confidence_penalty,
            ambiguity_kind=ambiguity_kind,
            ambiguity_reasons=ambiguity_reasons,
            repeated_dissatisfaction=repeated_dissatisfaction,
            included_working_memory_turns=included_working_memory_turns,
            working_memory_turn_count=working_memory_turn_count,
            included_context_types=included_context_types,
            included_lpb_entry_ids=included_lpb_entry_ids,
            included_belief_ids=included_belief_ids,
            context_strategy=context_strategy,
            related_context_item_ids=self._related_context_item_ids(context_signal),
            related_memory_ids=self._related_memory_ids(
                context_signal,
                memory_context,
            ),
            related_belief_ids=self._related_belief_ids(context_signal, world_context),
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
    def _extract_planner_context(state: NexusState) -> dict[str, Any] | None:
        state_planner = state.facet_state.get("planner")
        return state_planner if isinstance(state_planner, dict) else None

    @staticmethod
    def _extract_context_signal(
        metadata: dict[str, Any],
        state: NexusState,
    ) -> dict[str, Any]:
        candidate = metadata.get("context")
        if isinstance(candidate, dict):
            return candidate
        candidate = metadata.get("context_window")
        if isinstance(candidate, dict):
            return {"last_context_window": candidate}
        state_context = state.facet_state.get("context")
        return dict(state_context) if isinstance(state_context, dict) else {}

    @staticmethod
    def _extract_policy_signal(
        metadata: dict[str, Any],
        state: NexusState,
    ) -> tuple[str, bool, bool, str | None]:
        policy_payload = metadata.get("policy")
        if not isinstance(policy_payload, dict):
            policy_payload = state.facet_state.get("policy")
        if not isinstance(policy_payload, dict):
            return ("allow", False, False, None)

        raw_status = policy_payload.get("policy_status") or policy_payload.get(
            "last_policy_status"
        )
        status = str(raw_status).strip().lower() if isinstance(raw_status, str) else ""
        if status not in {"allowed", "denied", "approval_required", "no_match"}:
            status = "allow"
        requires_approval = status == "approval_required"
        blocks_act = status in {"denied", "approval_required"}
        reason = None
        raw_reasons = policy_payload.get("reasons")
        if isinstance(raw_reasons, list):
            reason = "; ".join(str(item) for item in raw_reasons if isinstance(item, str))
        if not reason:
            raw_effective = policy_payload.get("effective_policy")
            if isinstance(raw_effective, dict):
                policy_name = raw_effective.get("name")
                if isinstance(policy_name, str) and policy_name.strip():
                    reason = policy_name.strip()
        return (status, requires_approval, blocks_act, reason)

    @staticmethod
    def _extract_context_load_signal(
        context_signal: dict[str, Any],
        state: NexusState,
    ) -> tuple[int, int, float, bool]:
        load = context_signal.get("context_load")
        if not isinstance(load, dict):
            window = BehaviorFacet._context_window_from_signal(context_signal)
            metadata = window.get("metadata") if isinstance(window, dict) else {}
            load = metadata.get("context_load") if isinstance(metadata, dict) else None
        if isinstance(load, dict):
            item_count = int(load.get("item_count", 0) or 0)
            max_items = int(load.get("max_items", 0) or 0)
            raw_ratio = load.get("load_ratio")
            if isinstance(raw_ratio, (int, float)):
                ratio = _clamp_unit(float(raw_ratio))
            else:
                ratio = _clamp_unit(item_count / max(float(max_items), 1.0))
            overloaded = bool(load.get("overloaded", False))
            if not overloaded and max_items > 0:
                overloaded = ratio >= CONTEXT_OVERLOAD_RATIO_THRESHOLD
            return item_count, max_items, round(ratio, 3), overloaded

        item_count = BehaviorFacet._context_item_count(state)
        max_items = 0
        context_state = state.facet_state.get("context")
        if isinstance(context_state, dict):
            max_items = int(context_state.get("last_context_max_items", 0) or 0)
        ratio = _clamp_unit(item_count / max(float(max_items), 1.0)) if max_items > 0 else 0.0
        return item_count, max_items, round(ratio, 3), ratio >= CONTEXT_OVERLOAD_RATIO_THRESHOLD

    @staticmethod
    def _extract_latent_pressure(metadata: dict[str, Any], state: NexusState) -> float:
        explicit = BehaviorFacet._numeric_unit_value(metadata.get("latent_pressure"))
        if explicit is not None:
            return explicit
        signals_state = state.facet_state.get("signals")
        if isinstance(signals_state, dict):
            latent_state = signals_state.get("latent_pressure")
            if isinstance(latent_state, dict):
                last_result = latent_state.get("last_result")
                if isinstance(last_result, dict):
                    total = BehaviorFacet._numeric_unit_value(
                        last_result.get("latent_pressure_total")
                    )
                    if total is not None:
                        return total
        attention_state = state.facet_state.get("attention")
        if isinstance(attention_state, dict):
            contribution = BehaviorFacet._numeric_unit_value(
                attention_state.get("last_attention_pressure_contribution")
            )
            if contribution is not None:
                return contribution
        return 0.0

    @staticmethod
    def _context_window_from_signal(
        context_signal: dict[str, Any],
    ) -> dict[str, Any] | None:
        for key in ("context_window", "last_context_window"):
            candidate = context_signal.get(key)
            if isinstance(candidate, dict):
                return candidate
        return None

    @staticmethod
    def _context_window_items(context_signal: dict[str, Any]) -> list[dict[str, Any]]:
        context_window = BehaviorFacet._context_window_from_signal(context_signal)
        if context_window is None:
            return []
        raw_items = context_window.get("items")
        if not isinstance(raw_items, list):
            return []
        return [item for item in raw_items if isinstance(item, dict)]

    @staticmethod
    def _context_item_type_count(
        context_signal: dict[str, Any],
        item_type: str,
    ) -> int:
        return sum(
            1
            for item in BehaviorFacet._context_window_items(context_signal)
            if item.get("item_type") == item_type
        )

    @staticmethod
    def _context_relevant_memory_count(context_signal: dict[str, Any]) -> int:
        count = 0
        for item in BehaviorFacet._context_window_items(context_signal):
            if item.get("item_type") != "memory":
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and metadata.get("context_source") == "relevant":
                count += 1
        return count

    @staticmethod
    def _resolve_query_intent(
        content: str,
        *,
        metadata: dict[str, Any],
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> str:
        detected_intent = BehaviorFacet._detect_query_intent(content)
        if detected_intent != "unknown":
            return detected_intent
        for candidate in (
            metadata.get("query_intent"),
            context_signal.get("query_intent"),
            context_signal.get("last_query_intent"),
            memory_context.get("query_intent") if memory_context else None,
            memory_context.get("last_query_intent") if memory_context else None,
        ):
            coerced = BehaviorFacet._coerce_query_intent(candidate)
            if coerced != "unknown":
                return coerced
        return "unknown"

    @staticmethod
    def _coerce_query_intent(raw_intent: Any) -> str:
        if not isinstance(raw_intent, str):
            return "unknown"
        cleaned = raw_intent.strip().lower()
        aliases = {
            "recommendation_request": "recommendation",
            "recommendation": "recommendation",
            "advice": "recommendation",
            "planning_request": "planning",
            "planning": "planning",
            "factual_request": "factual",
            "factual": "factual",
            "status_request": "factual",
            "memory_summary": "memory_summary",
            "unknown": "unknown",
            "clarification_needed": "unknown",
        }
        return aliases.get(cleaned, "unknown")

    @staticmethod
    def _included_memory_roles(
        *,
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        roles: set[str] = set()
        for key in ("included_memory_roles", "last_included_memory_roles"):
            raw_roles = context_signal.get(key)
            if isinstance(raw_roles, list):
                roles.update(_clean_strings(raw_roles))
        if memory_context is not None:
            for key in ("included_memory_roles", "last_included_memory_roles"):
                raw_roles = memory_context.get(key)
                if isinstance(raw_roles, list):
                    roles.update(_clean_strings(raw_roles))
            for key in ("relevant_memories",):
                raw_memories = memory_context.get(key)
                if isinstance(raw_memories, list):
                    roles.update(
                        _clean_strings(
                            memory.get("role")
                            for memory in raw_memories
                            if isinstance(memory, dict)
                        )
                    )
        for item in BehaviorFacet._context_window_items(context_signal):
            if item.get("item_type") != "memory":
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                roles.update(_clean_strings([metadata.get("role")]))
        roles.discard("unknown")
        return sorted(roles)

    @staticmethod
    def _included_memory_domains(
        *,
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        domains: set[str] = set()
        for key in ("included_memory_domains", "last_included_memory_domains"):
            raw_domains = context_signal.get(key)
            if isinstance(raw_domains, list):
                domains.update(_clean_strings(raw_domains))
        if memory_context is not None:
            for key in ("included_memory_domains", "last_included_memory_domains"):
                raw_domains = memory_context.get(key)
                if isinstance(raw_domains, list):
                    domains.update(_clean_strings(raw_domains))
            for key in ("relevant_memories",):
                raw_memories = memory_context.get(key)
                if isinstance(raw_memories, list):
                    domains.update(
                        _clean_strings(
                            memory.get("domain")
                            for memory in raw_memories
                            if isinstance(memory, dict)
                        )
                    )
        for item in BehaviorFacet._context_window_items(context_signal):
            if item.get("item_type") != "memory":
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                domains.update(_clean_strings([metadata.get("domain")]))
        return sorted(domains)

    @staticmethod
    def _resolve_event_domain(
        event: Event,
        *,
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> str | None:
        for candidate in (
            context_signal.get("event_domain"),
            context_signal.get("last_event_domain"),
            memory_context.get("event_domain") if memory_context else None,
            memory_context.get("last_event_domain") if memory_context else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()
        return infer_domain(event.content, infer_tags(event.content))

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
    def _active_goal_count(goal_context: dict[str, Any] | None) -> int:
        if goal_context is None:
            return 0
        raw_count = goal_context.get("active_goal_count")
        if isinstance(raw_count, int) and raw_count >= 0:
            return raw_count
        for key in ("last_active_goals", "active_goals", "goals", "last_active_goal_ids"):
            candidate = goal_context.get(key)
            if isinstance(candidate, list):
                return len(candidate)
        return 0

    @staticmethod
    def _relevant_memory_count(memory_context: dict[str, Any] | None) -> int:
        if memory_context is None:
            return 0
        for key in (
            "relevant_memories",
            "last_relevant_memories",
            "last_relevant_memory_ids",
        ):
            candidate = memory_context.get(key)
            if isinstance(candidate, list):
                return len(candidate)
        return 0

    @staticmethod
    def _context_item_count(state: NexusState) -> int:
        context_state = state.facet_state.get("context")
        if not isinstance(context_state, dict):
            return 0
        raw_count = context_state.get("last_context_item_count")
        if isinstance(raw_count, int) and raw_count >= 0:
            return raw_count
        raw_ids = context_state.get("last_context_item_ids")
        if isinstance(raw_ids, list):
            return len(raw_ids)
        return 0

    @staticmethod
    def _planner_available(planner_context: dict[str, Any] | None) -> bool:
        if planner_context is None:
            return False
        last_plan = planner_context.get("last_plan")
        if isinstance(last_plan, dict) and isinstance(last_plan.get("steps"), list):
            return bool(last_plan["steps"])
        return bool(planner_context.get("last_plan_id"))

    @staticmethod
    def _detect_query_intent(content: str) -> str | None:
        normalized = _normalize_content(content)
        if not normalized:
            return "unknown"
        if any(phrase in normalized for phrase in MEMORY_SUMMARY_RESPONSE_PHRASES):
            return "memory_summary"
        if any(phrase in normalized for phrase in RECOMMENDATION_RESPONSE_PHRASES):
            return "recommendation"
        if any(phrase in normalized for phrase in PLANNING_RESPONSE_PHRASES):
            return "planning"
        if any(phrase in normalized for phrase in FACTUAL_RESPONSE_PHRASES):
            return "factual"
        if any(phrase in normalized for phrase in STATUS_RESPONSE_PHRASES):
            return "factual"
        return "unknown"

    @staticmethod
    def _context_sufficiency(
        *,
        query_intent: str | None,
        active_goal_count: int,
        relevant_goal_count: int,
        relevant_memory_count: int,
        relevant_belief_count: int,
        planner_available: bool,
    ) -> float:
        relevant_goal_signal = 1.0 if relevant_goal_count > 0 else 0.0
        if (
            relevant_goal_signal == 0.0
            and active_goal_count > 0
            and query_intent in {"planning", "recommendation"}
        ):
            relevant_goal_signal = 1.0
        relevant_memory_signal = 1.0 if relevant_memory_count > 0 else 0.0
        relevant_belief_signal = 1.0 if relevant_belief_count > 0 else 0.0
        planner_signal = 1.0 if planner_available else 0.0
        return round(
            relevant_goal_signal
            + relevant_memory_signal
            + relevant_belief_signal
            + planner_signal,
            3,
        )

    @staticmethod
    def _sufficiency_threshold(query_intent: str | None) -> float:
        if query_intent == "factual":
            return 0.0
        return CONTEXT_SUFFICIENCY_THRESHOLD

    @staticmethod
    def _missing_context(
        *,
        query_intent: str | None,
        active_goal_count: int,
        relevant_goal_count: int,
        relevant_memory_count: int,
        relevant_belief_count: int,
        planner_available: bool,
    ) -> list[str]:
        missing: list[str] = []
        if query_intent == "recommendation":
            if (
                active_goal_count == 0
                and relevant_goal_count == 0
                and relevant_memory_count == 0
                and relevant_belief_count == 0
            ):
                missing.extend(["preferences", "purpose"])
            return missing
        if query_intent == "planning":
            if active_goal_count == 0 and relevant_goal_count == 0:
                missing.append("active_goals")
            if not planner_available:
                missing.append("planner_summary")
            return missing
        if query_intent == "unknown":
            return ["specific_request", "relevant_context"]
        return missing

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
    def _resolve_relevant_memory_strength(
        *,
        retrieval_strength: float,
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
        relevant_memory_count: int,
        has_explicit_strength: bool,
        has_preference_memory: bool,
        query_intent: str | None,
    ) -> float:
        candidates = [retrieval_strength] if has_explicit_strength or relevant_memory_count > 0 else []
        candidates.extend(
            BehaviorFacet._memory_scores_from_context_signal(context_signal)
        )
        if memory_context is not None:
            candidates.extend(BehaviorFacet._memory_scores_from_memory_context(memory_context))
        if has_preference_memory and query_intent in {"recommendation", "planning"}:
            candidates.append(0.75)
        return round(_clamp_unit(max(candidates, default=0.0)), 3)

    @staticmethod
    def _memory_scores_from_context_signal(
        context_signal: dict[str, Any],
    ) -> list[float]:
        scores: list[float] = []
        raw_breakdowns = context_signal.get("memory_score_breakdowns")
        if not isinstance(raw_breakdowns, list):
            context_window = BehaviorFacet._context_window_from_signal(context_signal)
            if isinstance(context_window, dict):
                metadata = context_window.get("metadata")
                raw_breakdowns = (
                    metadata.get("memory_score_breakdowns")
                    if isinstance(metadata, dict)
                    else None
                )
        if isinstance(raw_breakdowns, list):
            for item in raw_breakdowns:
                if not isinstance(item, dict):
                    continue
                scores.append(BehaviorFacet._score_from_memory_breakdown(item))

        for item in BehaviorFacet._context_window_items(context_signal):
            if item.get("item_type") != "memory":
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("context_source") == "relevant":
                scores.append(0.5)
            scores.append(_coerce_unit(metadata.get("hybrid_score")))
            scores.append(_coerce_unit(metadata.get("salience")) * 0.8)
        return scores

    @staticmethod
    def _memory_scores_from_memory_context(memory_context: dict[str, Any]) -> list[float]:
        scores: list[float] = []
        for key in ("relevant_memories",):
            raw_memories = memory_context.get(key)
            if not isinstance(raw_memories, list):
                continue
            for memory in raw_memories:
                if not isinstance(memory, dict):
                    continue
                scores.append(_coerce_unit(memory.get("hybrid_score")))
                scores.append(_coerce_unit(memory.get("salience")) * 0.8)
                score_breakdown = memory.get("score_breakdown")
                if isinstance(score_breakdown, dict):
                    scores.append(_coerce_unit(score_breakdown.get("total")))
        return scores

    @staticmethod
    def _score_from_memory_breakdown(breakdown: dict[str, Any]) -> float:
        candidates: list[float] = []
        hybrid_score = _coerce_unit(breakdown.get("hybrid_score"))
        if hybrid_score > 0.0:
            candidates.append(hybrid_score)
        score_breakdown = breakdown.get("score_breakdown")
        if isinstance(score_breakdown, dict):
            candidates.append(_coerce_unit(score_breakdown.get("total")))
            candidates.append(_coerce_unit(score_breakdown.get("salience")) * 0.8)
            if score_breakdown.get("domain_match") == 1.0:
                candidates.append(0.55)
            if score_breakdown.get("role_bonus_raw"):
                candidates.append(0.45)
        if breakdown.get("context_source") == "relevant":
            candidates.append(0.5)
        return max(candidates, default=0.0)

    @staticmethod
    def _resolve_top_goal_priority(
        *,
        goal_context: dict[str, Any] | None,
        context_signal: dict[str, Any],
        aligned_goals: list[dict[str, Any]],
    ) -> float:
        priorities: list[float] = []
        for goal in aligned_goals:
            priorities.append(_coerce_unit(goal.get("priority")))
        if goal_context is not None:
            for key in ("last_active_goals", "active_goals", "goals", "last_relevant_goals", "relevant_goals"):
                raw_goals = goal_context.get(key)
                if isinstance(raw_goals, list):
                    priorities.extend(
                        _coerce_unit(goal.get("priority"))
                        for goal in raw_goals
                        if isinstance(goal, dict)
                    )
        for item in BehaviorFacet._context_window_items(context_signal):
            if item.get("item_type") != "goal":
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                priorities.append(_coerce_unit(metadata.get("priority")))
        return round(max(priorities, default=0.0), 3)

    @staticmethod
    def _resolve_goal_signal_strength(
        *,
        query_intent: str | None,
        active_goal_count: int,
        relevant_goal_count: int,
        top_goal_priority: float,
        goal_alignment_score: float,
        context_signal: dict[str, Any],
    ) -> float:
        if relevant_goal_count > 0 or goal_alignment_score > 0.0:
            return round(max(top_goal_priority, _clamp_unit(goal_alignment_score / 2.0)), 3)
        has_context_goal = any(
            item.get("item_type") == "goal"
            for item in BehaviorFacet._context_window_items(context_signal)
        )
        if query_intent in {"planning", "recommendation"} and (
            active_goal_count > 0 or has_context_goal
        ):
            return round(max(top_goal_priority, 0.5), 3)
        return 0.0

    @staticmethod
    def _has_domain_match(
        *,
        event_domain: str | None,
        included_memory_domains: list[str],
        context_signal: dict[str, Any],
    ) -> bool:
        if event_domain and event_domain in included_memory_domains:
            return True
        raw_breakdowns = context_signal.get("memory_score_breakdowns")
        if isinstance(raw_breakdowns, list):
            for item in raw_breakdowns:
                if not isinstance(item, dict):
                    continue
                score_breakdown = item.get("score_breakdown")
                if isinstance(score_breakdown, dict) and score_breakdown.get("domain_match") == 1.0:
                    return True
        return False

    @staticmethod
    def _compute_ambiguity_score(
        content: str,
        *,
        query_intent: str | None,
        has_relevant_memory: bool,
        relevant_memory_strength: float,
        has_preference_memory: bool,
        has_goal: bool,
        goal_signal_strength: float,
        relevant_belief_count: int,
        domain_match: bool,
        event_domain: str | None,
    ) -> float:
        normalized = _normalize_content(content)
        tokens = tokenize(normalized)
        if any(phrase in normalized for phrase in STATUS_RESPONSE_PHRASES):
            return 0.1
        score = 0.15

        if query_intent == "recommendation" and not (has_relevant_memory or has_goal):
            score = max(score, 0.85)
        if query_intent == "planning" and not (has_relevant_memory or has_goal):
            score = max(score, 0.75)
        if query_intent == "factual" and not (has_relevant_memory or relevant_belief_count > 0):
            score = max(score, 0.75)
        if query_intent == "memory_summary" and not has_relevant_memory:
            score = max(score, 0.75)
        if query_intent == "unknown" and content.strip().endswith("?"):
            score = max(score, 0.8)
        if BehaviorFacet._is_very_short_or_vague(normalized, tokens):
            score = max(score, 0.8)
        if query_intent in {"recommendation", "planning"} and event_domain and not domain_match:
            score += 0.15

        if relevant_memory_strength >= RELEVANT_MEMORY_STRENGTH_THRESHOLD:
            score -= 0.55
        if has_preference_memory and query_intent in {"recommendation", "planning"}:
            score -= 0.25
        if goal_signal_strength >= GOAL_RELEVANCE_THRESHOLD:
            score -= 0.45
        if domain_match:
            score -= 0.2
        return round(_clamp_unit(score), 3)

    @staticmethod
    def _is_very_short_or_vague(normalized: str, tokens: set[str]) -> bool:
        vague_queries = {
            "what should i do",
            "what should we do",
            "what now",
            "what next",
            "help me",
            "can you help",
        }
        stripped = normalized.rstrip(" ?!.")
        return stripped in vague_queries or (stripped.endswith("?") and len(tokens) <= 3)

    @staticmethod
    def _included_context_types(context_signal: dict[str, Any]) -> list[str]:
        raw = context_signal.get("included_context_types") or context_signal.get(
            "last_included_context_types"
        )
        return sorted(set(_clean_strings(raw if isinstance(raw, list) else [])))

    @staticmethod
    def _included_working_memory_turns(context_signal: dict[str, Any]) -> list[str]:
        raw = context_signal.get("included_working_memory_turns")
        if not isinstance(raw, list):
            window = BehaviorFacet._context_window_from_signal(context_signal)
            metadata = window.get("metadata") if isinstance(window, dict) else {}
            raw = metadata.get("included_working_memory_turns") if isinstance(metadata, dict) else []
        return [str(item) for item in raw] if isinstance(raw, list) else []

    @staticmethod
    def _working_memory_turn_count(context_signal: dict[str, Any]) -> int:
        raw_count = context_signal.get("working_memory_turn_count")
        if isinstance(raw_count, int) and raw_count >= 0:
            return raw_count
        return len(BehaviorFacet._included_working_memory_turns(context_signal))

    @staticmethod
    def _included_lpb_entry_ids(context_signal: dict[str, Any]) -> list[str]:
        raw = context_signal.get("included_lpb_entry_ids")
        if not isinstance(raw, list):
            window = BehaviorFacet._context_window_from_signal(context_signal)
            metadata = window.get("metadata") if isinstance(window, dict) else {}
            raw = metadata.get("included_lpb_entry_ids") if isinstance(metadata, dict) else []
        return [str(item) for item in raw] if isinstance(raw, list) else []

    @staticmethod
    def _included_belief_ids(context_signal: dict[str, Any]) -> list[str]:
        raw = context_signal.get("included_belief_ids")
        if not isinstance(raw, list):
            window = BehaviorFacet._context_window_from_signal(context_signal)
            metadata = window.get("metadata") if isinstance(window, dict) else {}
            raw = metadata.get("included_belief_ids") if isinstance(metadata, dict) else []
        return [str(item) for item in raw] if isinstance(raw, list) else []

    @staticmethod
    def _context_strategy(context_signal: dict[str, Any]) -> str | None:
        for key in ("context_strategy", "last_context_strategy", "strategy"):
            raw = context_signal.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        window = BehaviorFacet._context_window_from_signal(context_signal)
        if isinstance(window, dict):
            raw = window.get("strategy")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None

    @staticmethod
    def _classify_conversational_intent(
        event: Event,
        *,
        query_intent: str | None,
        working_memory_turn_count: int,
        has_recent_assistant_output: bool,
        has_previous_user_turn: bool,
        has_context_items: bool,
    ) -> tuple[bool, bool, str, float, list[str]]:
        text = _normalize_content(event.content)
        tokens = tokenize(text)
        short_follow_up = len(tokens) <= 4
        follow_up_reference_detected = bool(tokens & FOLLOW_UP_REFERENCE_WORDS)
        reasons: list[str] = []
        if any(phrase in text for phrase in ("where did that come from", "how do you know", "what is your source", "where did you get that", "based on what")):
            return follow_up_reference_detected, short_follow_up, "source_request", 0.95, ["explicit_source_request"]
        if any(phrase in text for phrase in ("that's not right", "that does not answer", "you keep saying", "why did you say", "i didn't ask that", "that's not what i meant")):
            return follow_up_reference_detected, short_follow_up, "challenge", 0.9, ["explicit_challenge"]
        if any(phrase in text for phrase in ("i mean", "to clarify", "what i mean is")):
            return follow_up_reference_detected, short_follow_up, "clarification_supplied", 0.85, ["clarification_language"]
        if any(phrase in text for phrase in ("actually", "correction")) or text.startswith("no,"):
            return follow_up_reference_detected, short_follow_up, "correction", 0.8, ["correction_language"]
        if any(phrase in text for phrase in ("contradiction", "that conflicts", "inconsistent with")):
            return follow_up_reference_detected, short_follow_up, "contradiction_report", 0.8, ["contradiction_language"]
        if query_intent == "planning":
            return follow_up_reference_detected, short_follow_up, "planning_request", 0.75, ["planning_intent"]
        if any(phrase in text for phrase in ("remember", "update memory", "store this")):
            return follow_up_reference_detected, short_follow_up, "memory_update", 0.75, ["memory_update_language"]
        if query_intent == "factual" and any(phrase in text for phrase in STATUS_RESPONSE_PHRASES):
            return follow_up_reference_detected, short_follow_up, "status_request", 0.7, ["status_request_language"]
        if short_follow_up and follow_up_reference_detected and (
            has_recent_assistant_output or has_previous_user_turn or has_context_items
        ):
            reasons.append("short_referential_with_recent_context")
            return follow_up_reference_detected, True, "follow_up", 0.7, reasons
        if any(phrase in text for phrase in ("you still", "again", "still not", "you keep")):
            return follow_up_reference_detected, short_follow_up, "repeated_dissatisfaction", 0.8, ["repeated_dissatisfaction_language"]
        if event.content.strip().endswith("?"):
            return follow_up_reference_detected, short_follow_up, "new_question", 0.6, ["question_mark"]
        if any(phrase in text for phrase in ("do ", "run ", "execute ", "update ", "change ")):
            return follow_up_reference_detected, short_follow_up, "action_request", 0.55, ["action_request_language"]
        return follow_up_reference_detected, short_follow_up, "unknown", 0.4, ["no_strong_intent_signal"]

    @staticmethod
    def _classify_grounding_need(conversational_intent: str) -> tuple[str, list[str]]:
        mapping = {
            "source_request": ("runtime_state", ["source_request_needs_provenance"]),
            "challenge": ("world_model", ["challenge_needs_world_state"]),
            "correction": ("world_model", ["correction_needs_world_state"]),
            "contradiction_report": ("verifier", ["contradiction_needs_verifier"]),
            "follow_up": ("working_memory", ["follow_up_needs_working_memory"]),
            "planning_request": ("policy", ["planning_needs_policy_and_goal_context"]),
            "action_request": ("executor", ["action_request_needs_executor_policy"]),
            "memory_update": ("memory", ["memory_update_needs_memory_store"]),
            "status_request": ("runtime_state", ["status_request_needs_runtime_state"]),
        }
        return mapping.get(conversational_intent, ("none", ["no_specific_grounding_required"]))

    @staticmethod
    def _resolve_grounding(
        *,
        grounding_need: str,
        working_memory_turn_count: int,
        relevant_memory_count: int,
        relevant_belief_count: int,
        policy_result: str,
        planner_available: bool,
        context_item_count: int,
    ) -> tuple[bool, float]:
        if grounding_need == "working_memory":
            available = working_memory_turn_count > 0
            return available, 0.8 if available else 0.25
        if grounding_need == "memory":
            available = relevant_memory_count > 0
            return available, 0.75 if available else 0.2
        if grounding_need in {"world_model", "verifier"}:
            available = relevant_belief_count > 0
            return available, 0.7 if available else 0.25
        if grounding_need == "policy":
            available = planner_available or context_item_count > 0
            return available, 0.7 if available else 0.3
        if grounding_need == "executor":
            available = policy_result in {"allowed", "allow"}
            return available, 0.7 if available else 0.2
        if grounding_need == "runtime_state":
            available = context_item_count > 0
            return available, 0.65 if available else 0.3
        return True, 0.6

    @staticmethod
    def _classify_ambiguity_kind(
        *,
        ambiguity_score: float,
        conversational_intent: str,
        follow_up_reference_detected: bool,
        short_follow_up: bool,
        grounding_available: bool,
        working_memory_turn_count: int,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if conversational_intent == "clarification_supplied":
            return "none", ["clarification_supplied_reduces_ambiguity"]
        if conversational_intent in {"source_request", "challenge", "contradiction_report"} and not grounding_available:
            return "missing_grounding", ["grounding_required_but_missing"]
        if conversational_intent == "repeated_dissatisfaction":
            return "repeated_unresolved", ["repeated_dissatisfaction_detected"]
        if follow_up_reference_detected and short_follow_up:
            if working_memory_turn_count > 0:
                return "referential", ["short_referential_with_continuity"]
            return "generic", ["short_referential_without_continuity"]
        if ambiguity_score >= HIGH_AMBIGUITY_THRESHOLD:
            reasons.append("high_ambiguity_score")
            return "generic", reasons
        return "none", ["ambiguity_within_threshold"]

    @staticmethod
    def _continuity_confidence(
        *,
        follow_up_reference_detected: bool,
        working_memory_turn_count: int,
        short_follow_up: bool,
    ) -> float:
        if follow_up_reference_detected and working_memory_turn_count > 0:
            return 0.85
        if short_follow_up and working_memory_turn_count > 0:
            return 0.7
        if working_memory_turn_count > 0:
            return 0.6
        return 0.25

    @staticmethod
    def _self_consistency_confidence(
        *,
        belief_confidence: float,
        belief_contradiction: bool,
    ) -> float:
        if belief_contradiction:
            return 0.2
        if belief_confidence > 0.0:
            return round(_clamp_unit(belief_confidence), 3)
        return 0.5

    @staticmethod
    def _challenge_penalty(
        *,
        conversational_intent: str,
        repeated_dissatisfaction: bool,
        belief_contradiction: bool,
        grounding_available: bool,
    ) -> float:
        penalty = 0.0
        if conversational_intent in {"challenge", "contradiction_report", "correction"}:
            penalty += 0.12
        if repeated_dissatisfaction:
            penalty += 0.1
        if belief_contradiction:
            penalty += 0.1
        if not grounding_available:
            penalty += 0.08
        return round(_clamp_unit(penalty), 3)

    @staticmethod
    def _related_context_item_ids(context_signal: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for key in ("included_item_ids", "last_context_item_ids"):
            raw = context_signal.get(key)
            if isinstance(raw, list):
                ids.extend(str(item) for item in raw)
        return sorted(set(ids))

    @staticmethod
    def _related_memory_ids(
        context_signal: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        ids: set[str] = set()
        raw = context_signal.get("included_memory_ids")
        if isinstance(raw, list):
            ids.update(str(item) for item in raw)
        if memory_context is not None:
            for key in ("last_relevant_memory_ids",):
                raw_mem = memory_context.get(key)
                if isinstance(raw_mem, list):
                    ids.update(str(item) for item in raw_mem)
        return sorted(ids)

    @staticmethod
    def _related_belief_ids(
        context_signal: dict[str, Any],
        world_context: dict[str, Any] | None,
    ) -> list[str]:
        ids: set[str] = set()
        raw = context_signal.get("included_belief_ids")
        if isinstance(raw, list):
            ids.update(str(item) for item in raw)
        if world_context is not None:
            raw_world = world_context.get("last_relevant_beliefs")
            if isinstance(raw_world, list):
                for row in raw_world:
                    if isinstance(row, dict) and isinstance(row.get("id"), str):
                        ids.add(row["id"])
        return sorted(ids)

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
    def _resolve_belief_signal(
        aligned_beliefs: list[dict[str, Any]],
    ) -> tuple[float, bool, str | None]:
        confidence = BehaviorFacet._resolve_world_alignment_confidence(aligned_beliefs)
        contradicted = False
        for belief in aligned_beliefs:
            status = belief.get("status")
            if isinstance(status, str) and status.strip().lower() == "contradicted":
                contradicted = True
                break
        if contradicted:
            return confidence, True, "world_model_contradiction"
        if confidence > 0.0 and confidence < LOW_BELIEF_CONFIDENCE_THRESHOLD:
            return confidence, False, "low_belief_confidence"
        return confidence, False, None

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
        query_intent: str | None,
        context_sufficiency: float,
        missing_context: list[str],
    ) -> str | None:
        normalized = _normalize_content(content)
        if query_intent == "factual" and any(
            phrase in normalized for phrase in STATUS_RESPONSE_PHRASES
        ):
            return "status_report"
        if query_intent == "recommendation" and missing_context:
            return "clarify_recommendation_preferences"
        if (
            any(phrase in normalized for phrase in NEXT_STEPS_RESPONSE_PHRASES)
            and context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD
        ):
            return "next_steps_available"
        if query_intent == "planning" and context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD:
            return "next_steps_available"
        if query_intent == "memory_summary":
            return (
                "grounded_response_available"
                if context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD
                else "clarification_needed"
            )
        if query_intent in {"recommendation", "planning", "factual"}:
            return "grounded_response_available" if context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD else "clarification_needed"
        if query_intent == "unknown":
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
        response_intent = "none"

        if selected_decision == DecisionAction.ASK:
            response_intent = "clarify"
        elif selected_decision == DecisionAction.ACT:
            if signals.query_intent == "memory_summary":
                response_intent = "memory_summary"
            elif signals.query_intent == "planning":
                response_intent = "plan"
            elif signals.query_intent == "factual":
                response_intent = "status" if response_template == "status_report" else "answer"
            elif signals.query_intent == "recommendation":
                response_intent = "plan"
            else:
                response_intent = "answer"
        elif selected_decision == DecisionAction.RECORD:
            response_intent = "acknowledge" if response_needed else "none"

        if response_template is None:
            response_template = {
                "clarify": "clarification_needed",
                "status": "status_report",
                "plan": "next_steps_available",
                "memory_summary": "grounded_response_available",
                "answer": "grounded_response_available",
                "acknowledge": "clarification_needed",
            }.get(response_intent)

        if selected_decision == DecisionAction.ASK:
            response_needed = True
            response_reason = response_reason or "clarification_needed"
            response_template = response_template or "clarification_needed"
        elif selected_decision == DecisionAction.ACT and response_needed:
            response_template = response_template or "clarification_needed"
        else:
            return {
                "response_needed": response_needed,
                "response_reason": response_reason,
                "response_template": response_template,
                "response_intent": response_intent,
            }

        metadata: dict[str, Any] = {
            "response_needed": response_needed,
            "response_reason": response_reason,
            "response_template": response_template,
            "response_intent": response_intent,
        }
        if selected_decision in {DecisionAction.ASK, DecisionAction.ACT}:
            metadata.update(
                {
                    "output_type": "text",
                    "tool": "text",
                }
            )
        metadata.update(
            {
                "query_intent": signals.query_intent,
                "active_goal_count": signals.active_goal_count,
                "relevant_goal_count": signals.relevant_goal_count,
                "relevant_memory_count": signals.relevant_memory_count,
                "relevant_belief_count": signals.relevant_belief_count,
                "context_item_count": signals.context_item_count,
                "planner_available": signals.planner_available,
                "context_sufficiency": signals.context_sufficiency,
                "missing_context": list(signals.missing_context),
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

        if event.event_type in {EventType.SYSTEM_TICK, EventType.INTERNAL}:
            reasons.append("internal_system_event_wait")
            add(DecisionAction.WAIT, "internal_system_event_wait", 0.8)
            BehaviorFacet._apply_pressure_biases(score_breakdown, reasons, signals)
            return BehaviorFacet._finalize_selected_decision(
                DecisionAction.WAIT,
                reasons,
                score_breakdown,
            )

        if not signals.meaningful_content and not signals.has_metadata_signal:
            reasons.append("empty_content_wait")
            add(DecisionAction.WAIT, "empty_content_wait", 0.8)
            return BehaviorFacet._finalize_selected_decision(
                DecisionAction.WAIT,
                reasons,
                score_breakdown,
            )

        if signals.explicit_action:
            if signals.low_risk and signals.ambiguity_score < HIGH_AMBIGUITY_THRESHOLD:
                reasons.append("explicit_action_low_risk")
                add(DecisionAction.ACT, "explicit_action_low_risk", 0.75)
                selected = DecisionAction.ACT
            else:
                reasons.append("explicit_action_without_low_risk")
                add(DecisionAction.ASK, "explicit_action_without_low_risk", 0.75)
                selected = DecisionAction.ASK
            BehaviorFacet._apply_pressure_biases(score_breakdown, reasons, signals)
            BehaviorFacet._apply_goal_biases(score_breakdown, reasons, signals)
            BehaviorFacet._apply_memory_biases(score_breakdown, reasons, signals)
            return BehaviorFacet._finalize_selected_decision(
                selected,
                reasons,
                score_breakdown,
            )

        if not signals.response_needed:
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
            BehaviorFacet._apply_pressure_biases(score_breakdown, reasons, signals)
            BehaviorFacet._apply_goal_biases(score_breakdown, reasons, signals)
            BehaviorFacet._apply_memory_biases(score_breakdown, reasons, signals)
            BehaviorFacet._apply_low_signal_bias(score_breakdown, reasons, signals)
            return BehaviorFacet._finalize_selected_decision(
                selected,
                reasons,
                score_breakdown,
            )

        grounded = (
            signals.has_relevant_memory
            or signals.has_goal
            or signals.deterministic_response_available
            or signals.context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD
        )
        low_ambiguity = signals.ambiguity_score <= LOW_AMBIGUITY_THRESHOLD
        high_ambiguity = signals.ambiguity_score >= HIGH_AMBIGUITY_THRESHOLD

        if (
            signals.query_intent in {"recommendation", "planning"}
            and signals.has_preference_memory
        ):
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

        if grounded and low_ambiguity and signals.query_intent in QUERY_INTENTS_REQUIRING_RESPONSE:
            reasons.append("grounded_low_ambiguity_act")
            add(DecisionAction.ACT, "grounded_low_ambiguity_act", 0.75)
            selected = DecisionAction.ACT
        elif high_ambiguity:
            reasons.append("high_ambiguity_insufficient_context")
            add(DecisionAction.ASK, "high_ambiguity_insufficient_context", 0.45)
            selected = DecisionAction.ASK
        elif grounded and signals.query_intent in QUERY_INTENTS_REQUIRING_RESPONSE:
            reasons.append("grounded_response_context_act")
            add(DecisionAction.ACT, "grounded_response_context_act", 0.45)
            selected = DecisionAction.ACT
        else:
            reasons.append("response_needed_but_unclear_ask")
            add(DecisionAction.ASK, "response_needed_but_unclear_ask", 0.55)
            selected = DecisionAction.ASK

        BehaviorFacet._apply_pressure_biases(score_breakdown, reasons, signals)
        BehaviorFacet._apply_goal_biases(score_breakdown, reasons, signals)
        BehaviorFacet._apply_memory_biases(score_breakdown, reasons, signals)
        return BehaviorFacet._finalize_selected_decision(
            selected,
            reasons,
            score_breakdown,
        )

    @staticmethod
    def _finalize_selected_decision(
        selected_decision: DecisionAction,
        reasons: list[str],
        score_breakdown: dict[DecisionAction, dict[str, float]],
    ) -> tuple[DecisionAction, list[str], dict[str, float]]:
        decision_scores = {
            action: round(_clamp_unit(sum(breakdown.values())), 3)
            for action, breakdown in score_breakdown.items()
        }
        reasons.append(f"selected_policy_rule:{selected_decision.value}")
        return (
            selected_decision,
            reasons,
            {action.value: decision_scores[action] for action in DECISION_BASE_SCORES},
        )

    @staticmethod
    def _apply_v2_candidate_adjustments(
        decision_scores: dict[str, float],
        signals: _BehaviorSignals,
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

        if signals.belief_confidence > 0.0 and signals.belief_confidence < LOW_BELIEF_CONFIDENCE_THRESHOLD:
            adjusted["act"] = _clamp_unit(adjusted["act"] - 0.35)
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
            reasons.append("low belief confidence suppressed ACT and boosted ASK")
        if signals.belief_contradiction:
            adjusted["act"] = _clamp_unit(adjusted["act"] - CONTRADICTION_ACT_PENALTY)
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.25)
            reasons.append("belief contradiction biased decision toward ASK")
        if signals.context_overloaded:
            adjusted["act"] = _clamp_unit(
                adjusted["act"] - (0.25 * (1.0 - (signals.pressure * signals.goal_relevance)))
            )
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.15)
            adjusted["record"] = _clamp_unit(adjusted["record"] + 0.1)
            reasons.append("context_overload biased away from ACT")
        return {key: round(value, 3) for key, value in adjusted.items()}

    @staticmethod
    def _select_highest_scored_decision(decision_scores: dict[str, float]) -> DecisionAction:
        ordering = {
            "wait": DecisionAction.WAIT,
            "record": DecisionAction.RECORD,
            "ask": DecisionAction.ASK,
            "act": DecisionAction.ACT,
        }
        selected_key = max(
            ordering,
            key=lambda key: (decision_scores.get(key, 0.0), DECISION_PRIORITY[ordering[key]]),
        )
        return ordering[selected_key]

    @staticmethod
    def _apply_policy_downgrade(
        selected_decision: DecisionAction,
        decision_scores: dict[str, float],
        signals: _BehaviorSignals,
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

    @staticmethod
    def _interrupt_recommendation(signals: _BehaviorSignals) -> tuple[bool, str | None]:
        if signals.pressure >= 0.85:
            return True, "pressure_spike"
        if signals.goal_relevance >= 0.85 and signals.pressure >= 0.6:
            return True, "high_goal_relevance_under_pressure"
        if signals.latent_pressure >= 0.75:
            return True, "latent_pressure_high"
        return False, None

    @staticmethod
    def _build_decision_trace(
        *,
        event: Event,
        signals: _BehaviorSignals,
        raw_candidate_scores: dict[str, float],
        adjusted_candidate_scores: dict[str, float],
        selected_decision: DecisionAction,
        confidence: float,
        reasons: list[str],
        interrupt_recommended: bool,
        interrupt_reason: str | None,
    ) -> dict[str, Any]:
        event_summary = " ".join(event.content.split())[:120]
        return {
            "event": {
                "id": event.event_id,
                "type": event.event_type.value,
                "content_summary": event_summary,
            },
            "pressure_score": signals.pressure,
            "latent_pressure": signals.latent_pressure,
            "memory_relevance_score": signals.relevant_memory_strength,
            "goal_relevance_score": signals.goal_relevance,
            "world_model_belief_confidence": signals.belief_confidence,
            "contradiction_flag": signals.belief_contradiction,
            "policy_result": signals.policy_result,
            "context_load_ratio": signals.context_load_ratio,
            "conversational_intent": signals.conversational_intent,
            "conversational_intent_score": signals.conversational_intent_score,
            "conversational_intent_reasons": list(
                signals.conversational_intent_reasons
            ),
            "follow_up_reference_detected": signals.follow_up_reference_detected,
            "short_follow_up": signals.short_follow_up,
            "grounding_need": signals.grounding_need,
            "grounding_need_reasons": list(signals.grounding_need_reasons),
            "grounding_available": signals.grounding_available,
            "grounding_confidence": signals.grounding_confidence,
            "ambiguity_kind": signals.ambiguity_kind,
            "ambiguity_score": signals.ambiguity_score,
            "ambiguity_reasons": list(signals.ambiguity_reasons),
            "continuity_confidence": signals.continuity_confidence,
            "self_consistency_confidence": signals.self_consistency_confidence,
            "challenge_confidence_penalty": signals.challenge_confidence_penalty,
            "raw_candidate_scores": raw_candidate_scores,
            "adjusted_candidate_scores": adjusted_candidate_scores,
            "final_decision": selected_decision.value,
            "confidence": confidence,
            "reasons": list(reasons),
            "interrupt_recommended": interrupt_recommended,
            "interrupt_reason": interrupt_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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
    def _attention_confidence_bias(
        state: NexusState,
        *,
        response_needed: bool,
    ) -> float:
        if not response_needed:
            return 0.0
        attention_state = state.facet_state.get("attention")
        if not isinstance(attention_state, dict):
            return 0.0
        raw_broadcast = attention_state.get("last_attention_broadcast")
        if not isinstance(raw_broadcast, dict):
            return 0.0
        if str(raw_broadcast.get("mode") or "").strip().lower() != "top_down":
            return 0.0
        return 0.05

    @staticmethod
    def _score_confidence(
        action: DecisionAction,
        *,
        pressure: float,
        memory_signal_strength: float,
        goal_signal_strength: float,
        ambiguity_score: float,
        goal_alignment_score: float,
        goal_alignment_priority: float,
        world_alignment_score: float,
        world_alignment_confidence: float,
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {
            "base": DECISION_BASE_SCORES[action],
            "pressure_contribution": round(_clamp_unit(pressure) * 0.25, 3),
            "goal_relevance_contribution": round(
                _clamp_unit(goal_signal_strength) * 0.30,
                3,
            ),
            "memory_retrieval_contribution": round(
                _clamp_unit(memory_signal_strength) * 0.30,
                3,
            ),
            "ambiguity_penalty": round(_clamp_unit(ambiguity_score) * -0.35, 3),
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
            "ambiguity_penalty",
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
                f"memory contribution: {signals.relevant_memory_strength:.3f} -> "
                f"{confidence_breakdown['memory_retrieval_contribution']:.3f}"
            ),
            (
                f"ambiguity contribution: {signals.ambiguity_score:.3f} -> "
                f"{confidence_breakdown['ambiguity_penalty']:.3f}"
            ),
            (
                "final confidence breakdown: "
                f"base={confidence_breakdown['base']:.3f}, "
                f"pressure={confidence_breakdown['pressure_contribution']:.3f}, "
                f"goal={confidence_breakdown['goal_relevance_contribution']:.3f}, "
                f"memory={confidence_breakdown['memory_retrieval_contribution']:.3f}, "
                f"ambiguity={confidence_breakdown['ambiguity_penalty']:.3f}, "
                f"total={confidence_breakdown['total']:.3f}"
            ),
        ]

    @staticmethod
    def _apply_v21_candidate_adjustments(
        decision_scores: dict[str, float],
        signals: _BehaviorSignals,
    ) -> tuple[dict[str, float], list[str]]:
        adjusted = {key: _clamp_unit(value) for key, value in decision_scores.items()}
        reasons: list[str] = []
        if signals.conversational_intent == "source_request":
            if signals.grounding_available:
                adjusted["act"] = _clamp_unit(adjusted["act"] + 0.25)
                reasons.append("source_request with grounding boosted ACT")
            else:
                adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.25)
                adjusted["act"] = _clamp_unit(adjusted["act"] - 0.2)
                reasons.append("source_request without grounding biased ASK")
        if signals.conversational_intent in {
            "challenge",
            "contradiction_report",
            "correction",
        }:
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
            adjusted["act"] = _clamp_unit(
                adjusted["act"] - (0.15 if signals.grounding_available else 0.25)
            )
            reasons.append("challenge_or_contradiction bias toward ASK")
        if signals.conversational_intent == "clarification_supplied":
            adjusted["act"] = _clamp_unit(adjusted["act"] + 0.2)
            adjusted["ask"] = _clamp_unit(adjusted["ask"] - 0.1)
            reasons.append("clarification_supplied reduced ASK bias")
        if signals.conversational_intent == "follow_up":
            if signals.continuity_confidence >= 0.5:
                adjusted["act"] = _clamp_unit(adjusted["act"] + 0.2)
                reasons.append("follow_up continuity boosted ACT")
            else:
                adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.15)
                reasons.append("follow_up unresolved reference boosted ASK")
        if signals.repeated_dissatisfaction:
            adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.1)
            adjusted["act"] = _clamp_unit(adjusted["act"] - 0.15)
            reasons.append("repeated dissatisfaction lowered ACT confidence path")
        if signals.conversational_intent == "planning_request":
            if signals.planner_available and signals.context_sufficiency >= 1.0:
                adjusted["act"] = _clamp_unit(adjusted["act"] + 0.2)
                reasons.append("planning_request with planner boosted ACT")
            else:
                adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.15)
                reasons.append("planning_request missing constraints boosted ASK")
        if signals.conversational_intent == "action_request":
            if signals.policy_blocks_act:
                adjusted["act"] = _clamp_unit(adjusted["act"] - 0.4)
                adjusted["ask"] = _clamp_unit(adjusted["ask"] + 0.2)
                reasons.append("action_request policy guardrail suppressed ACT")
        return ({k: round(v, 3) for k, v in adjusted.items()}, reasons)

    @staticmethod
    def _final_confidence_reasons(
        signals: _BehaviorSignals,
        confidence_breakdown: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        if not signals.grounding_available:
            reasons.append("grounding unavailable lowers confidence")
        if signals.belief_contradiction:
            reasons.append("belief contradiction lowers confidence")
        if signals.repeated_dissatisfaction:
            reasons.append("repeated dissatisfaction lowers confidence")
        if signals.continuity_confidence >= 0.6:
            reasons.append("working memory continuity supports confidence")
        reasons.append(f"final_total={confidence_breakdown.get('total', 0.0):.3f}")
        return reasons

    @staticmethod
    def _learning_signals(signals: _BehaviorSignals) -> list[str]:
        out: list[str] = []
        if signals.conversational_intent == "source_request" and not signals.grounding_available:
            out.append("source_request_unresolved")
        if signals.conversational_intent in {"challenge", "contradiction_report"} and not signals.grounding_available:
            out.append("challenge_unresolved")
        if signals.repeated_dissatisfaction:
            out.append("repeated_dissatisfaction")
        if signals.conversational_intent == "clarification_supplied":
            out.append("clarification_supplied")
        if signals.conversational_intent == "follow_up" and signals.continuity_confidence >= 0.6:
            out.append("follow_up_resolved_by_working_memory")
        if signals.grounding_confidence < 0.4:
            out.append("low_grounding_confidence")
        if signals.belief_contradiction:
            out.append("contradiction_pressure")
        return out

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


def _coerce_unit(raw_value: Any) -> float:
    if isinstance(raw_value, bool):
        return 0.0
    if not isinstance(raw_value, (int, float)):
        return 0.0
    return _clamp_unit(float(raw_value))


def _clean_strings(raw_values: Any) -> list[str]:
    cleaned: list[str] = []
    for raw_value in raw_values or ():
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip().lower()
        if value:
            cleaned.append(value)
    return cleaned


def _world_confidence_boost(
    *,
    world_alignment_score: float,
    world_alignment_confidence: float,
) -> float:
    if world_alignment_score <= 0.0 or world_alignment_confidence <= 0.0:
        return 0.0

    normalized_alignment = _clamp_unit(world_alignment_score / 3.0)
    boost = (
        0.06
        + (0.08 * _clamp_unit(world_alignment_confidence))
        + (0.04 * normalized_alignment)
    )
    return round(min(boost, 0.18), 3)
