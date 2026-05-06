"""Expression Gate v0 — support infrastructure (not a canonical facet)."""

from fullerene.expression.gate import evaluate_expression_gate
from fullerene.expression.models import (
    ExpressionBudgetState,
    ExpressionMode,
    ExpressionRecommendation,
    SuggestedIntent,
)
from fullerene.expression.scoring import ExpressionScoreComponents, compute_expression_score

__all__ = [
    "ExpressionBudgetState",
    "ExpressionMode",
    "ExpressionRecommendation",
    "ExpressionScoreComponents",
    "SuggestedIntent",
    "compute_expression_score",
    "evaluate_expression_gate",
]
