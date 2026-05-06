"""Behavior v2.2 modular components."""

from .orchestrator import BehaviorFacet, HIGH_AMBIGUITY_THRESHOLD, LOW_AMBIGUITY_THRESHOLD
from .confidence import confidence_breakdown, contribution_reasons, final_confidence_reasons
from .learning import learning_signals
from .lexical import extract_text_signals, normalize_content
from .models import (
    HIGH_AMBIGUITY_THRESHOLD,
    LOW_AMBIGUITY_THRESHOLD,
    BehaviorSignals,
    BehaviorTextSignals,
    ResponseIntent,
    TextIntentScores,
)
from .scoring import (
    apply_policy_downgrade,
    apply_v2_candidate_adjustments,
    apply_v21_candidate_adjustments,
    interrupt_recommendation,
    select_decision,
    select_highest_scored_decision,
)
from .signals import extract_adapter_signals
from .trace import build_decision_trace

__all__ = [
    "BehaviorFacet",
    "BehaviorSignals",
    "BehaviorTextSignals",
    "HIGH_AMBIGUITY_THRESHOLD",
    "LOW_AMBIGUITY_THRESHOLD",
    "ResponseIntent",
    "TextIntentScores",
    "apply_policy_downgrade",
    "apply_v2_candidate_adjustments",
    "apply_v21_candidate_adjustments",
    "build_decision_trace",
    "confidence_breakdown",
    "contribution_reasons",
    "extract_adapter_signals",
    "extract_text_signals",
    "final_confidence_reasons",
    "interrupt_recommendation",
    "learning_signals",
    "normalize_content",
    "select_decision",
    "select_highest_scored_decision",
]
