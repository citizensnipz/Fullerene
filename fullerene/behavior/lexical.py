"""Deterministic lexical signal extraction for Behavior.

This module is a bootstrap feature extractor. It converts text into structured
numeric signals for Behavior scoring and does not emit decisions or final prose.
Future learned heads can replace or augment these functions without changing the
decision/scoring interfaces.
"""

from __future__ import annotations

from fullerene.memory import tokenize

from .models import (
    CONTEXT_SUFFICIENCY_THRESHOLD,
    HIGH_AMBIGUITY_THRESHOLD,
    LOW_AMBIGUITY_THRESHOLD,
    BehaviorTextSignals,
    TextIntentScores,
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


def normalize_content(content: str) -> str:
    return " ".join(content.casefold().split())


def detect_query_intent(content: str) -> str:
    normalized = normalize_content(content)
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


def contains_response_phrase(content: str) -> bool:
    stripped = content.strip()
    if stripped.endswith("?"):
        return True
    normalized = normalize_content(content)
    return any(phrase in normalized for phrase in RESPONSE_PHRASES)


def classify_conversational_intent(
    content: str,
    *,
    query_intent: str,
    working_memory_turn_count: int,
    has_context_items: bool,
) -> tuple[bool, bool, str, float, list[str]]:
    text = normalize_content(content)
    tokens = tokenize(text)
    short_follow_up = len(tokens) <= 4
    follow_up_reference_detected = bool(tokens & FOLLOW_UP_REFERENCE_WORDS)
    reasons: list[str] = []
    if any(
        phrase in text
        for phrase in (
            "where did that come from",
            "how do you know",
            "what is your source",
            "where did you get that",
            "based on what",
        )
    ):
        return follow_up_reference_detected, short_follow_up, "source_request", 0.95, [
            "explicit_source_request"
        ]
    if any(
        phrase in text
        for phrase in (
            "that's not right",
            "that does not answer",
            "you keep saying",
            "why did you say",
            "i didn't ask that",
            "that's not what i meant",
        )
    ):
        return follow_up_reference_detected, short_follow_up, "challenge", 0.9, [
            "explicit_challenge"
        ]
    if any(phrase in text for phrase in ("i mean", "to clarify", "what i mean is")):
        return (
            follow_up_reference_detected,
            short_follow_up,
            "clarification_supplied",
            0.85,
            ["clarification_language"],
        )
    if any(phrase in text for phrase in ("actually", "correction")) or text.startswith(
        "no,"
    ):
        return follow_up_reference_detected, short_follow_up, "correction", 0.8, [
            "correction_language"
        ]
    if any(
        phrase in text
        for phrase in ("contradiction", "that conflicts", "inconsistent with")
    ):
        return (
            follow_up_reference_detected,
            short_follow_up,
            "contradiction_report",
            0.8,
            ["contradiction_language"],
        )
    if query_intent == "planning":
        return follow_up_reference_detected, short_follow_up, "planning_request", 0.75, [
            "planning_intent"
        ]
    if any(phrase in text for phrase in ("remember", "update memory", "store this")):
        return follow_up_reference_detected, short_follow_up, "memory_update", 0.75, [
            "memory_update_language"
        ]
    if query_intent == "factual" and any(
        phrase in text for phrase in STATUS_RESPONSE_PHRASES
    ):
        return follow_up_reference_detected, short_follow_up, "status_request", 0.7, [
            "status_request_language"
        ]
    if short_follow_up and follow_up_reference_detected and (
        working_memory_turn_count > 0 or has_context_items
    ):
        reasons.append("short_referential_with_recent_context")
        return follow_up_reference_detected, True, "follow_up", 0.7, reasons
    if any(phrase in text for phrase in ("you still", "again", "still not", "you keep")):
        return (
            follow_up_reference_detected,
            short_follow_up,
            "repeated_dissatisfaction",
            0.8,
            ["repeated_dissatisfaction_language"],
        )
    if content.strip().endswith("?"):
        return follow_up_reference_detected, short_follow_up, "new_question", 0.6, [
            "question_mark"
        ]
    if any(phrase in text for phrase in ("do ", "run ", "execute ", "update ", "change ")):
        return follow_up_reference_detected, short_follow_up, "action_request", 0.55, [
            "action_request_language"
        ]
    return follow_up_reference_detected, short_follow_up, "unknown", 0.4, [
        "no_strong_intent_signal"
    ]


def classify_grounding_need(conversational_intent: str) -> tuple[str, list[str]]:
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


def classify_ambiguity_kind(
    *,
    ambiguity_score: float,
    conversational_intent: str,
    follow_up_reference_detected: bool,
    short_follow_up: bool,
    grounding_available: bool,
    working_memory_turn_count: int,
) -> tuple[str, list[str]]:
    if conversational_intent == "clarification_supplied":
        return "none", ["clarification_supplied_reduces_ambiguity"]
    if conversational_intent in {"source_request", "challenge", "contradiction_report"} and (
        not grounding_available
    ):
        return "missing_grounding", ["grounding_required_but_missing"]
    if conversational_intent == "repeated_dissatisfaction":
        return "repeated_unresolved", ["repeated_dissatisfaction_detected"]
    if follow_up_reference_detected and short_follow_up:
        if working_memory_turn_count > 0:
            return "referential", ["short_referential_with_continuity"]
        return "generic", ["short_referential_without_continuity"]
    if ambiguity_score >= HIGH_AMBIGUITY_THRESHOLD:
        return "generic", ["high_ambiguity_score"]
    return "none", ["ambiguity_within_threshold"]


def extract_text_signals(
    content: str,
    *,
    query_intent: str,
    ambiguity_score: float,
    context_sufficiency: float,
    missing_context: list[str],
    working_memory_turn_count: int,
    has_context_items: bool,
    grounding_available: bool,
    grounding_confidence: float,
    self_consistency_confidence: float,
) -> BehaviorTextSignals:
    normalized = normalize_content(content)
    tokens = tokenize(normalized)
    question_score = 1.0 if content.strip().endswith("?") else 0.0
    shortness_score = 1.0 if len(tokens) <= 4 else 0.0
    referential_score = 1.0 if bool(tokens & FOLLOW_UP_REFERENCE_WORDS) else 0.0
    imperative_score = 1.0 if any(
        normalized.startswith(prefix) for prefix in ("do ", "run ", "execute ", "update ", "change ")
    ) else 0.0
    vague_score = 1.0 if ambiguity_score >= HIGH_AMBIGUITY_THRESHOLD else 0.0
    (
        follow_up_reference_detected,
        short_follow_up,
        conversational_intent,
        conversational_intent_score,
        conversational_intent_reasons,
    ) = classify_conversational_intent(
        content,
        query_intent=query_intent,
        working_memory_turn_count=working_memory_turn_count,
        has_context_items=has_context_items,
    )
    grounding_need, grounding_need_reasons = classify_grounding_need(conversational_intent)
    ambiguity_kind, ambiguity_reasons = classify_ambiguity_kind(
        ambiguity_score=ambiguity_score,
        conversational_intent=conversational_intent,
        follow_up_reference_detected=follow_up_reference_detected,
        short_follow_up=short_follow_up,
        grounding_available=grounding_available,
        working_memory_turn_count=working_memory_turn_count,
    )
    challenge_penalty = 0.0
    if conversational_intent in {"challenge", "contradiction_report", "correction"}:
        challenge_penalty += 0.12
    if conversational_intent == "repeated_dissatisfaction":
        challenge_penalty += 0.1
    if not grounding_available:
        challenge_penalty += 0.08
    response_needed = query_intent in {"recommendation", "planning", "factual", "memory_summary"} or contains_response_phrase(content)
    response_reason = "direct_question" if response_needed else None
    response_template: str | None = None
    if query_intent == "factual" and any(phrase in normalized for phrase in STATUS_RESPONSE_PHRASES):
        response_template = "status_report"
    elif query_intent == "recommendation" and missing_context:
        response_template = "clarify_recommendation_preferences"
    elif any(phrase in normalized for phrase in NEXT_STEPS_RESPONSE_PHRASES) and context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD:
        response_template = "next_steps_available"
    elif query_intent == "planning" and context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD:
        response_template = "next_steps_available"
    elif query_intent == "memory_summary":
        response_template = "grounded_response_available" if context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD else "clarification_needed"
    elif query_intent in {"recommendation", "planning", "factual"}:
        response_template = "grounded_response_available" if context_sufficiency >= CONTEXT_SUFFICIENCY_THRESHOLD else "clarification_needed"
    elif query_intent == "unknown":
        response_template = "clarification_needed"
    deterministic_response_available = bool(response_needed and context_sufficiency >= (0.0 if query_intent == "factual" else CONTEXT_SUFFICIENCY_THRESHOLD))
    intents = TextIntentScores(
        factual=1.0 if query_intent == "factual" else 0.0,
        recommendation=1.0 if query_intent == "recommendation" else 0.0,
        planning=1.0 if query_intent == "planning" else 0.0,
        memory_summary=1.0 if query_intent == "memory_summary" else 0.0,
        status=1.0 if conversational_intent == "status_request" else 0.0,
        action_request=1.0 if conversational_intent == "action_request" else 0.0,
        source_request=1.0 if conversational_intent == "source_request" else 0.0,
        challenge=1.0 if conversational_intent == "challenge" else 0.0,
        correction=1.0 if conversational_intent == "correction" else 0.0,
        clarification_supplied=1.0 if conversational_intent == "clarification_supplied" else 0.0,
        contradiction_report=1.0 if conversational_intent == "contradiction_report" else 0.0,
        repeated_dissatisfaction=1.0 if conversational_intent == "repeated_dissatisfaction" else 0.0,
        follow_up=1.0 if conversational_intent == "follow_up" else 0.0,
        memory_update=1.0 if conversational_intent == "memory_update" else 0.0,
        unknown=1.0 if query_intent == "unknown" else 0.0,
    )
    return BehaviorTextSignals(
        intents=intents,
        question_score=question_score,
        shortness_score=shortness_score,
        referential_score=referential_score,
        imperative_score=imperative_score,
        vague_score=vague_score,
        challenge_score=1.0 if conversational_intent in {"challenge", "contradiction_report"} else 0.0,
        source_request_score=1.0 if conversational_intent == "source_request" else 0.0,
        clarification_score=1.0 if conversational_intent == "clarification_supplied" else 0.0,
        correction_score=1.0 if conversational_intent == "correction" else 0.0,
        query_intent=query_intent,
        response_template=response_template,
        deterministic_response_available=deterministic_response_available,
        response_needed=response_needed,
        response_reason=response_reason,
        conversational_intent=conversational_intent,
        conversational_intent_score=conversational_intent_score,
        conversational_intent_reasons=conversational_intent_reasons,
        follow_up_reference_detected=follow_up_reference_detected,
        short_follow_up=short_follow_up,
        grounding_need=grounding_need,
        grounding_need_reasons=grounding_need_reasons,
        grounding_available=grounding_available,
        grounding_confidence=grounding_confidence,
        continuity_confidence=0.85 if follow_up_reference_detected and working_memory_turn_count > 0 else (0.7 if short_follow_up and working_memory_turn_count > 0 else (0.6 if working_memory_turn_count > 0 else 0.25)),
        self_consistency_confidence=self_consistency_confidence,
        challenge_confidence_penalty=min(challenge_penalty, 1.0),
        ambiguity_kind=ambiguity_kind,
        ambiguity_score=ambiguity_score,
        ambiguity_reasons=ambiguity_reasons,
        repeated_dissatisfaction=conversational_intent == "repeated_dissatisfaction",
    )

