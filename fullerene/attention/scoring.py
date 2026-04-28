"""Deterministic fixed-weight scoring for Fullerene Attention v0."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fullerene.attention.models import ATTENTION_STRATEGY_FIXED_WEIGHT_V0

ATTENTION_COMPONENT_WEIGHTS: dict[str, float] = {
    "memory_salience": 0.25,
    "goal_priority": 0.25,
    "pressure": 0.20,
    "novelty": 0.15,
    "belief_uncertainty": 0.10,
    "execution_recency": 0.05,
}

ATTENTION_COMPONENT_NAMES = tuple(ATTENTION_COMPONENT_WEIGHTS)


class FixedWeightAttentionScorer:
    """Score inspectable attention candidates without mutating runtime state."""

    strategy = ATTENTION_STRATEGY_FIXED_WEIGHT_V0

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = self._resolve_weights(weights)

    def evaluate(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        top_n: int = 3,
    ) -> dict[str, Any]:
        bounded_top_n = max(int(top_n), 1)
        ranked = sorted(
            (self.score_candidate(candidate) for candidate in candidates),
            key=self._sort_key,
        )
        selected = ranked[:bounded_top_n]
        scores = {item["id"]: item["score"] for item in ranked}
        dominant_source = selected[0]["source"] if selected else None
        return {
            "ranked": ranked,
            "selected": selected,
            "scores": scores,
            "dominant_source": dominant_source,
            "strategy": self.strategy,
        }

    def score_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        raw_signals = {
            component: self._coerce_unit(candidate.get(component, 0.0))
            for component in ATTENTION_COMPONENT_NAMES
        }
        components = {
            component: raw_signals[component] * self.weights[component]
            for component in ATTENTION_COMPONENT_NAMES
        }
        score = self._clamp_unit(sum(components.values()))
        dominant_component = self._dominant_component(components)
        dominant_component_value = (
            components[dominant_component] if dominant_component is not None else 0.0
        )
        metadata = dict(candidate.get("metadata") or {})
        metadata.setdefault("raw_signals", raw_signals)
        metadata.setdefault("weights", dict(self.weights))
        metadata.setdefault("dominant_component_value", dominant_component_value)
        return {
            "id": str(candidate.get("id") or ""),
            "source": str(candidate.get("source") or "system"),
            "source_id": candidate.get("source_id"),
            "content": str(candidate.get("content") or ""),
            "score": score,
            "components": components,
            "dominant_component": dominant_component,
            "metadata": metadata,
        }

    @staticmethod
    def _sort_key(item: Mapping[str, Any]) -> tuple[float, float, str, str]:
        dominant_component = item.get("dominant_component")
        components = item.get("components") or {}
        dominant_value = 0.0
        if isinstance(components, Mapping) and dominant_component in components:
            dominant_value = float(components[dominant_component])
        return (
            -float(item.get("score", 0.0)),
            -dominant_value,
            str(item.get("source") or ""),
            str(item.get("id") or ""),
        )

    @staticmethod
    def _dominant_component(components: Mapping[str, float]) -> str | None:
        if not components:
            return None
        best_component = max(
            ATTENTION_COMPONENT_NAMES,
            key=lambda component: (components.get(component, 0.0), component),
        )
        if components.get(best_component, 0.0) <= 0.0:
            return None
        return best_component

    @staticmethod
    def _coerce_unit(value: Any) -> float:
        try:
            return FixedWeightAttentionScorer._clamp_unit(float(value))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _resolve_weights(
        weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        resolved = dict(ATTENTION_COMPONENT_WEIGHTS)
        if weights is None:
            return resolved
        for component, value in weights.items():
            if component not in resolved:
                continue
            resolved[component] = max(float(value), 0.0)
        return resolved
