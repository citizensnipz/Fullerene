"""Deterministic Affect v0 derivation helpers."""

from __future__ import annotations

from typing import Any

from fullerene.affect.models import (
    AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0,
    AffectResult,
    AffectState,
)
from fullerene.learning import SignalType, collect_learning_signals
from fullerene.memory import infer_tags, normalize_tags
from fullerene.nexus.models import Event, NexusState


class DeterministicAffectDeriver:
    """Derive internal VAD + novelty state from existing deterministic signals."""

    strategy = AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0

    def derive(
        self,
        event: Event,
        state: NexusState,
        *,
        history: list[AffectState] | None = None,
    ) -> AffectResult:
        previous_history = list(history or [])
        learning_signals = collect_learning_signals(event, state)

        valence, valence_components, valence_reasons = self._derive_valence(
            learning_signals
        )
        arousal, arousal_components, arousal_reasons = self._derive_arousal(event, state)
        dominance, dominance_components, dominance_reasons = self._derive_dominance(
            state,
            history=previous_history,
        )
        novelty, novelty_components, novelty_reasons = self._derive_novelty(event, state)

        affect_state = AffectState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            novelty=novelty,
            components={
                "valence": valence_components,
                "arousal": arousal_components,
                "dominance": dominance_components,
                "novelty": novelty_components,
            },
            metadata={
                "strategy": self.strategy,
                "history_count": len(previous_history),
                "learning_signal_count": len(learning_signals),
            },
        )
        reasons = self._dedupe(
            [
                *valence_reasons,
                *arousal_reasons,
                *dominance_reasons,
                *novelty_reasons,
            ]
        )
        return AffectResult(
            current_state=affect_state,
            history=previous_history,
            strategy=self.strategy,
            reasons=reasons,
            metadata={
                "learning_signals": [signal.to_dict() for signal in learning_signals],
                "signal_counts": {
                    "learning": len(learning_signals),
                    "world_model": len(self._world_belief_matches(state)),
                    "memory_relevant": len(self._memory_relevant_ids(state)),
                    "memory_working": len(self._memory_working_ids(state)),
                },
            },
        )

    def _derive_valence(
        self,
        learning_signals,
    ) -> tuple[float, dict[str, Any], list[str]]:
        positive_total = 0.0
        negative_total = 0.0
        positive_signals: list[dict[str, Any]] = []
        negative_signals: list[dict[str, Any]] = []
        reasons: list[str] = []

        for signal in learning_signals:
            if signal.signal_type == SignalType.POSITIVE:
                positive_total += signal.magnitude
                positive_signals.append(signal.to_dict())
                reasons.extend(signal.reasons or ["positive_signal"])
                continue
            if signal.signal_type in {SignalType.NEGATIVE, SignalType.FAILURE}:
                negative_total += signal.magnitude
                negative_signals.append(signal.to_dict())
                reasons.extend(signal.reasons or ["negative_signal"])

        raw = positive_total - negative_total
        value = self._clamp_signed(raw)
        components = {
            "positive_total": round(positive_total, 3),
            "negative_total": round(negative_total, 3),
            "raw": round(raw, 3),
            "value": value,
            "positive_signals": positive_signals,
            "negative_signals": negative_signals,
        }
        if not positive_signals and not negative_signals:
            reasons.append("valence_neutral_no_feedback_signals")
        return value, components, self._dedupe(reasons)

    def _derive_arousal(
        self,
        event: Event,
        state: NexusState,
    ) -> tuple[float, dict[str, Any], list[str]]:
        metadata = self._event_metadata(event)
        event_tags = self._event_tags(event)
        pressure = self._coerce_optional_unit(metadata.get("pressure")) or 0.0
        salience = self._coerce_optional_unit(metadata.get("salience")) or 0.0
        urgent = 1.0 if ("urgent" in event_tags or self._metadata_flag(metadata, "urgent")) else 0.0
        attention_peak = self._attention_peak_score(state)

        weighted_pressure = round(pressure * 0.55, 3)
        weighted_urgent = round(urgent * 0.20, 3)
        weighted_salience = round(salience * 0.10, 3)
        weighted_attention = round(attention_peak * 0.15, 3)
        raw = weighted_pressure + weighted_urgent + weighted_salience + weighted_attention
        value = self._clamp_unit(raw)

        reasons: list[str] = []
        if pressure > 0.0:
            reasons.append("arousal_pressure_signal")
        if urgent > 0.0:
            reasons.append("arousal_urgent_signal")
        if salience > 0.0:
            reasons.append("arousal_salience_signal")
        if attention_peak > 0.0:
            reasons.append("arousal_attention_peak_signal")
        if not reasons:
            reasons.append("arousal_calm_default")

        components = {
            "pressure": pressure,
            "urgent": urgent,
            "salience": salience,
            "attention_peak": attention_peak,
            "weighted_pressure": weighted_pressure,
            "weighted_urgent": weighted_urgent,
            "weighted_salience": weighted_salience,
            "weighted_attention_peak": weighted_attention,
            "raw": round(raw, 3),
            "value": value,
        }
        return value, components, reasons

    def _derive_dominance(
        self,
        state: NexusState,
        *,
        history: list[AffectState],
    ) -> tuple[float, dict[str, Any], list[str]]:
        execution_payload = self._execution_payload(state)
        current_execution_signal, execution_penalty, execution_reasons = (
            self._execution_signal(execution_payload)
        )
        rolling_execution_signal = self._rolling_component_value(
            history,
            component_name="dominance",
            value_name="execution_signal",
        )
        execution_signal = self._blend_signals(
            current_execution_signal,
            rolling_execution_signal,
        )

        world_matches = self._world_belief_matches(state)
        current_world_signal, world_penalty, world_reasons = self._world_signal(
            world_matches
        )
        rolling_world_signal = self._rolling_component_value(
            history,
            component_name="dominance",
            value_name="world_signal",
        )
        world_signal = self._blend_signals(
            current_world_signal,
            rolling_world_signal,
        )

        signal_values = [
            signal for signal in (execution_signal, world_signal) if signal is not None
        ]
        base_value = (
            round(sum(signal_values) / len(signal_values), 3)
            if signal_values
            else 0.5
        )
        total_penalty = round(execution_penalty + world_penalty, 3)
        value = self._clamp_unit(base_value - total_penalty)

        reasons = self._dedupe(
            [
                *execution_reasons,
                *world_reasons,
                "dominance_default_neutral"
                if not signal_values
                else "dominance_derived_from_available_control_signals",
            ]
        )
        components = {
            "current_execution_signal": current_execution_signal,
            "rolling_execution_signal": rolling_execution_signal,
            "execution_signal": execution_signal,
            "current_world_signal": current_world_signal,
            "rolling_world_signal": rolling_world_signal,
            "world_signal": world_signal,
            "execution_penalty": execution_penalty,
            "world_penalty": world_penalty,
            "total_penalty": total_penalty,
            "raw": base_value,
            "value": value,
        }
        return value, components, reasons

    def _derive_novelty(
        self,
        event: Event,
        state: NexusState,
    ) -> tuple[float, dict[str, Any], list[str]]:
        metadata = self._event_metadata(event)
        explicit_novelty = self._coerce_optional_unit(metadata.get("novelty"))
        if explicit_novelty is not None:
            return (
                explicit_novelty,
                {
                    "source": "explicit_metadata",
                    "memory_hit_rate": None,
                    "relevant_memory_count": len(self._memory_relevant_ids(state)),
                    "working_memory_count": len(self._memory_working_ids(state)),
                    "raw": explicit_novelty,
                    "value": explicit_novelty,
                },
                ["novelty_explicit_metadata"],
            )

        relevant_memory_count = len(self._memory_relevant_ids(state))
        working_memory_count = len(self._memory_working_ids(state))
        if relevant_memory_count or working_memory_count:
            denominator = max(relevant_memory_count, working_memory_count, 1)
            hit_rate = self._clamp_unit(relevant_memory_count / denominator)
            novelty = self._clamp_unit(1.0 - hit_rate)
            return (
                novelty,
                {
                    "source": "inverse_memory_hit_rate",
                    "memory_hit_rate": hit_rate,
                    "relevant_memory_count": relevant_memory_count,
                    "working_memory_count": working_memory_count,
                    "raw": round(1.0 - hit_rate, 3),
                    "value": novelty,
                },
                ["novelty_inverse_memory_hit_rate"],
            )

        return (
            0.5,
            {
                "source": "default_moderate_novelty",
                "memory_hit_rate": None,
                "relevant_memory_count": 0,
                "working_memory_count": 0,
                "raw": 0.5,
                "value": 0.5,
            },
            ["novelty_default_moderate"],
        )

    @staticmethod
    def _event_metadata(event: Event) -> dict[str, Any]:
        return event.metadata if isinstance(event.metadata, dict) else {}

    @staticmethod
    def _event_tags(event: Event) -> set[str]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        return set(normalize_tags(metadata.get("tags", [])) + infer_tags(event.content))

    @staticmethod
    def _execution_payload(state: NexusState) -> dict[str, Any] | None:
        executor_state = state.facet_state.get("executor")
        if not isinstance(executor_state, dict):
            return None
        payload = executor_state.get("last_execution_result")
        return dict(payload) if isinstance(payload, dict) else None

    def _execution_signal(
        self,
        payload: dict[str, Any] | None,
    ) -> tuple[float | None, float, list[str]]:
        if payload is None:
            return None, 0.0, ["dominance_no_executor_signal"]

        overall_status = str(payload.get("overall_status", "")).strip().casefold()
        dry_run = bool(payload.get("dry_run", True))
        reasons = [
            str(reason).strip().casefold()
            for reason in payload.get("reasons", [])
            if str(reason).strip()
        ]
        records = payload.get("records", [])
        record_statuses = [
            str(record.get("status", "")).strip().casefold()
            for record in records
            if isinstance(record, dict)
        ]
        success_count = record_statuses.count("success")
        skipped_count = record_statuses.count("skipped")
        total_count = len(record_statuses)
        ratio = (
            round((success_count + (0.5 * skipped_count)) / total_count, 3)
            if total_count
            else None
        )

        if overall_status == "success":
            base = max(ratio or 0.0, 0.8 if dry_run else 0.85)
        elif overall_status == "failed":
            base = min(ratio or 0.2, 0.2)
        elif overall_status == "skipped":
            base = min(ratio or 0.35, 0.4)
        else:
            base = ratio if ratio is not None else 0.5

        penalty = 0.0
        if {"blocked_by_policy", "requires_approval"} & set(reasons):
            penalty += 0.1
        if {
            "unsupported_action_type",
            "unsupported_target_type",
            "unsupported_live_action",
            "execution_failed",
            "invalid_action_payload",
        } & set(reasons):
            penalty += 0.15

        return (
            self._clamp_unit(base),
            round(penalty, 3),
            self._dedupe(["dominance_executor_signal", *reasons]),
        )

    def _world_signal(
        self,
        matches: list[dict[str, Any]],
    ) -> tuple[float | None, float, list[str]]:
        if not matches:
            return None, 0.0, ["dominance_no_world_model_signal"]

        confidences = [self._coerce_unit(match.get("confidence")) for match in matches]
        statuses = [
            str(match.get("status", "")).strip().casefold()
            for match in matches
            if str(match.get("status", "")).strip()
        ]
        average_confidence = round(sum(confidences) / len(confidences), 3)
        penalty = 0.0
        if "contradicted" in statuses:
            penalty += 0.25
        if "stale" in statuses:
            penalty += 0.1
        world_signal = self._clamp_unit(average_confidence - penalty)
        reasons = ["dominance_world_confidence_signal"]
        if "contradicted" in statuses:
            reasons.append("dominance_world_contradiction_penalty")
        if "stale" in statuses:
            reasons.append("dominance_world_staleness_penalty")
        return world_signal, round(penalty, 3), reasons

    @staticmethod
    def _attention_peak_score(state: NexusState) -> float:
        attention_state = state.facet_state.get("attention")
        if not isinstance(attention_state, dict):
            return 0.0
        payload = attention_state.get("last_attention_result")
        if not isinstance(payload, dict):
            return 0.0
        scores = payload.get("scores", {})
        if not isinstance(scores, dict):
            return 0.0
        numeric_scores = [
            DeterministicAffectDeriver._coerce_unit(score)
            for score in scores.values()
        ]
        return max(numeric_scores, default=0.0)

    @staticmethod
    def _world_belief_matches(state: NexusState) -> list[dict[str, Any]]:
        world_state = state.facet_state.get("world_model")
        if not isinstance(world_state, dict):
            return []
        matches = world_state.get("last_relevant_beliefs")
        if not isinstance(matches, list):
            return []
        return [dict(match) for match in matches if isinstance(match, dict)]

    @staticmethod
    def _memory_relevant_ids(state: NexusState) -> list[str]:
        memory_state = state.facet_state.get("memory")
        if not isinstance(memory_state, dict):
            return []
        ids = memory_state.get("last_relevant_memory_ids")
        if not isinstance(ids, list):
            return []
        return [str(item) for item in ids]

    @staticmethod
    def _memory_working_ids(state: NexusState) -> list[str]:
        memory_state = state.facet_state.get("memory")
        if not isinstance(memory_state, dict):
            return []
        ids = memory_state.get("last_working_memory_ids")
        if not isinstance(ids, list):
            return []
        return [str(item) for item in ids]

    @staticmethod
    def _rolling_component_value(
        history: list[AffectState],
        *,
        component_name: str,
        value_name: str,
    ) -> float | None:
        values: list[float] = []
        for state in history:
            component = state.components.get(component_name)
            if not isinstance(component, dict):
                continue
            raw_value = component.get(value_name)
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    @staticmethod
    def _blend_signals(
        current_value: float | None,
        historical_value: float | None,
    ) -> float | None:
        if current_value is None:
            return historical_value
        if historical_value is None:
            return current_value
        return round((current_value + historical_value) / 2.0, 3)

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
    def _coerce_unit(value: Any) -> float:
        try:
            return round(max(0.0, min(float(value), 1.0)), 3)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_optional_unit(value: Any) -> float | None:
        try:
            return round(max(0.0, min(float(value), 1.0)), 3)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp_signed(value: float) -> float:
        return round(max(-1.0, min(float(value), 1.0)), 3)

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return round(max(0.0, min(float(value), 1.0)), 3)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            deduped.append(cleaned)
            seen.add(key)
        return deduped
