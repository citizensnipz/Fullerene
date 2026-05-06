from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fullerene.nexus.models import DecisionAction

HIGH_PRIORITY_TAGS = frozenset(
    {"hard-rule-candidate", "correction", "urgent", "authority"}
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
LOW_BELIEF_CONFIDENCE_THRESHOLD = 0.4
CONTRADICTION_ACT_PENALTY = 0.35
CONTEXT_OVERLOAD_RATIO_THRESHOLD = 0.85


class ResponseIntent(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    STATUS = "status"
    PLAN = "plan"
    MEMORY_SUMMARY = "memory_summary"
    ACKNOWLEDGE = "acknowledge"
    NONE = "none"


@dataclass(slots=True)
class TextIntentScores:
    factual: float = 0.0
    recommendation: float = 0.0
    planning: float = 0.0
    memory_summary: float = 0.0
    status: float = 0.0
    action_request: float = 0.0
    source_request: float = 0.0
    challenge: float = 0.0
    correction: float = 0.0
    clarification_supplied: float = 0.0
    contradiction_report: float = 0.0
    repeated_dissatisfaction: float = 0.0
    follow_up: float = 0.0
    memory_update: float = 0.0
    unknown: float = 0.0


@dataclass(slots=True)
class BehaviorTextSignals:
    intents: TextIntentScores
    question_score: float
    shortness_score: float
    referential_score: float
    imperative_score: float
    vague_score: float
    challenge_score: float
    source_request_score: float
    clarification_score: float
    correction_score: float
    query_intent: str
    response_template: str | None
    deterministic_response_available: bool
    response_needed: bool
    response_reason: str | None
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
    ambiguity_score: float
    ambiguity_reasons: list[str]
    repeated_dissatisfaction: bool


@dataclass(slots=True)
class BehaviorSignals:
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
    pressure: float
    latent_pressure: float
    retrieval_strength: float
    relevant_memory_strength: float
    has_relevant_memory: bool
    has_preference_memory: bool
    has_goal: bool
    top_goal_priority: float
    goal_signal_strength: float
    goal_relevance: float
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
    memory_signal_available: bool
    goal_signal_available: bool
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
    included_working_memory_turns: list[str]
    working_memory_turn_count: int
    included_context_types: list[str]
    included_lpb_entry_ids: list[str]
    included_belief_ids: list[str]
    context_strategy: str | None
    related_context_item_ids: list[str]
    related_memory_ids: list[str]
    related_belief_ids: list[str]
    text: BehaviorTextSignals

