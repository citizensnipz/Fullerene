"""Deterministic Expression Gate v0 scoring — no models, no network."""

from __future__ import annotations

from dataclasses import dataclass

from fullerene.expression.models import _clamp01


@dataclass(slots=True)
class ExpressionScoreComponents:
    system_pressure: float
    latent_pressure: float
    interrupt_priority: float
    verifier_escalation: float
    policy_attention_need: float
    novelty: float
    confidence: float
    repetition_penalty: float
    recent_expression_penalty: float
    context_overload_penalty: float

    def raw_sum(self) -> float:
        return (
            self.system_pressure * 0.25
            + self.latent_pressure * 0.20
            + self.interrupt_priority * 0.20
            + self.verifier_escalation * 0.15
            + self.policy_attention_need * 0.10
            + self.novelty * 0.05
            + self.confidence * 0.05
            - self.repetition_penalty
            - self.recent_expression_penalty
            - self.context_overload_penalty
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "system_pressure_component": round(self.system_pressure * 0.25, 6),
            "latent_pressure_component": round(self.latent_pressure * 0.20, 6),
            "interrupt_priority_component": round(self.interrupt_priority * 0.20, 6),
            "verifier_escalation_component": round(self.verifier_escalation * 0.15, 6),
            "policy_attention_need_component": round(
                self.policy_attention_need * 0.10,
                6,
            ),
            "novelty_component": round(self.novelty * 0.05, 6),
            "confidence_component": round(self.confidence * 0.05, 6),
            "repetition_penalty": round(self.repetition_penalty, 6),
            "recent_expression_penalty": round(self.recent_expression_penalty, 6),
            "context_overload_penalty": round(self.context_overload_penalty, 6),
            "expression_score": _clamp01(self.raw_sum()),
        }


def compute_expression_score(components: ExpressionScoreComponents) -> float:
    return _clamp01(components.raw_sum())
