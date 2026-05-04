"""Minimal Nexus runtime loop."""

from __future__ import annotations

from typing import Any, Iterable

from fullerene.facets.base import Facet
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusState,
)
from fullerene.policy.models import PolicyStatus
from fullerene.state.store import InMemoryStateStore, StateStore

# Higher score wins when multiple facets explicitly propose a decision.
# ACT > ASK > RECORD > WAIT.
DECISION_PRIORITY = {
    DecisionAction.WAIT: 0,
    DecisionAction.RECORD: 1,
    DecisionAction.ASK: 2,
    DecisionAction.ACT: 3,
}
PHASE_ORDER = (
    "input_context",
    "state",
    "decision",
    "planning_execution",
    "learning_signal",
    "verification_output",
)
PHASE_LABELS = {
    "input_context": "INPUT / CONTEXT",
    "state": "STATE",
    "decision": "DECISION",
    "planning_execution": "PLANNING / EXECUTION",
    "learning_signal": "LEARNING / SIGNAL",
    "verification_output": "VERIFICATION / OUTPUT",
}
FACET_PHASES = {
    "context": "input_context",
    "memory": "input_context",
    "goals": "state",
    "world_model": "state",
    "worldmodel": "state",
    "behavior": "decision",
    "policy": "decision",
    "planner": "planning_execution",
    "executor": "planning_execution",
    "learning": "learning_signal",
    "attention": "learning_signal",
    "affect": "learning_signal",
    "verifier": "verification_output",
    "echo": "verification_output",
}
PHASE_FACET_ORDER = {
    "input_context": ("context", "memory"),
    "state": ("goals", "world_model", "worldmodel"),
    "decision": ("behavior", "policy"),
    "planning_execution": ("planner", "executor"),
    "learning_signal": ("learning", "attention", "affect"),
    "verification_output": ("echo", "verifier"),
}


class Nexus:
    """Central interpreter/integrator loop for Fullerene v0."""

    def __init__(
        self,
        facets: Iterable[Facet] | None = None,
        store: StateStore | None = None,
        initial_state: NexusState | None = None,
    ) -> None:
        self._store = store or InMemoryStateStore()
        self._facets: list[Facet] = list(facets or [])
        self.state = initial_state or self._store.load_state() or NexusState()

    @property
    def facets(self) -> tuple[Facet, ...]:
        return tuple(self._facets)

    def register_facet(self, facet: Facet) -> None:
        self._facets.append(facet)

    def process_event(self, event: Event) -> NexusRecord:
        primary_record, internal_events = self._process_event_cycle(
            event,
            collect_internal_events=True,
        )
        internal_records: list[NexusRecord] = []
        if internal_events:
            internal_event = internal_events[0]
            internal_record, _ = self._process_event_cycle(
                internal_event,
                collect_internal_events=False,
            )
            internal_records.append(internal_record)
            primary_record.metadata["internal_events_processed"] = [
                internal_event.to_dict()
            ]
            primary_record.metadata["internal_events_dropped"] = max(
                len(internal_events) - 1,
                0,
            )

        self._store.save_state(self.state)
        self._store.append_record(primary_record)
        for internal_record in internal_records:
            self._store.append_record(internal_record)
        return primary_record

    def _process_event_cycle(
        self,
        event: Event,
        *,
        collect_internal_events: bool,
    ) -> tuple[NexusRecord, list[Event]]:
        working_state = NexusState.from_dict(self.state.to_dict())
        working_state.system_pressure = self._aggregate_pressure(
            event,
            working_state,
            [],
        )
        facet_results: list[FacetResult] = []
        internal_events: list[Event] = []
        phase_execution_order: list[dict[str, Any]] = []
        facet_outputs_by_phase: dict[str, list[dict[str, Any]]] = {}
        phase_buckets = self._phase_buckets(self._facets)

        for phase_name in PHASE_ORDER:
            phase_facets = phase_buckets.get(phase_name, [])
            ordered_facets = self._order_phase_facets(
                phase_name,
                phase_facets,
                system_pressure=working_state.system_pressure,
            )
            phase_label = PHASE_LABELS[phase_name]
            phase_execution_order.append(
                {
                    "phase": phase_label,
                    "facets": [self._facet_name(facet) for facet in ordered_facets],
                    "priority_weights": self._phase_priority_weights(
                        phase_name,
                        ordered_facets,
                        system_pressure=working_state.system_pressure,
                    ),
                }
            )
            phase_outputs: list[dict[str, Any]] = []
            for facet in ordered_facets:
                if self._is_post_decision_verifier(facet):
                    continue
                result = self._run_facet(facet, event, working_state)
                emitted_events = self._extract_internal_events(result)
                self._normalize_internal_event_metadata(result)
                if collect_internal_events:
                    internal_events.extend(emitted_events)
                facet_results.append(result)
                phase_outputs.append(self._facet_output_summary(result))
                self._apply_result_to_state(working_state, result)
                working_state.system_pressure = self._aggregate_pressure(
                    event,
                    working_state,
                    facet_results,
                )
            facet_outputs_by_phase[phase_label] = phase_outputs
        decision = self._integrate(event, facet_results)

        verifier_facets = [
            facet
            for facet in phase_buckets.get("verification_output", [])
            if self._is_post_decision_verifier(facet)
        ]
        verifier_outputs = facet_outputs_by_phase.setdefault(
            PHASE_LABELS["verification_output"],
            [],
        )
        for verifier in verifier_facets:
            verifier_result = self._run_verifier(
                verifier,
                event,
                working_state,
                facet_results,
                decision,
            )
            emitted_events = self._extract_internal_events(verifier_result)
            self._normalize_internal_event_metadata(verifier_result)
            if collect_internal_events:
                internal_events.extend(emitted_events)
            facet_results.append(verifier_result)
            verifier_outputs.append(self._facet_output_summary(verifier_result))
            self._apply_result_to_state(working_state, verifier_result)
            working_state.system_pressure = self._aggregate_pressure(
                event,
                working_state,
                facet_results,
            )
            decision = self._apply_verifier_decision(decision, verifier_result)
        system_pressure = self._aggregate_pressure(event, working_state, facet_results)
        working_state.system_pressure = system_pressure
        self.state = working_state
        self.state.apply(
            event,
            facet_results,
            decision,
            system_pressure=system_pressure,
        )

        record = NexusRecord(
            event=event,
            facet_results=facet_results,
            decision=decision,
            metadata={
                "system_pressure": system_pressure,
                "phase_execution_order": phase_execution_order,
                "facet_outputs_by_phase": facet_outputs_by_phase,
                "internal_events_processed": [],
            },
        )
        if not collect_internal_events and internal_events:
            record.metadata["internal_events_dropped"] = len(internal_events)
        return record, internal_events

    def _run_facet(
        self,
        facet: Facet,
        event: Event,
        state: NexusState,
    ) -> FacetResult:
        try:
            return facet.process(event, state)
        except Exception as exc:
            facet_name = self._facet_name(facet)
            error_message = str(exc) or "Facet raised without an error message."
            return FacetResult(
                facet_name=facet_name,
                summary=(
                    f"Facet '{facet_name}' failed while processing the event: "
                    f"{error_message}"
                ),
                proposed_decision=DecisionAction.RECORD,
                metadata={
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message,
                },
            )

    @staticmethod
    def _apply_result_to_state(state: NexusState, result: FacetResult) -> None:
        if not result.state_updates:
            return
        facet_bucket = state.facet_state.setdefault(result.facet_name, {})
        facet_bucket.update(result.state_updates)

    def _phase_buckets(self, facets: Iterable[Facet]) -> dict[str, list[Facet]]:
        buckets = {phase_name: [] for phase_name in PHASE_ORDER}
        for facet in facets:
            phase_name = self._phase_for_facet(facet)
            buckets[phase_name].append(facet)
        return buckets

    def _phase_for_facet(self, facet: Facet) -> str:
        facet_name = self._facet_name(facet).strip().casefold()
        return FACET_PHASES.get(facet_name, "decision")

    def _order_phase_facets(
        self,
        phase_name: str,
        facets: list[Facet],
        *,
        system_pressure: float,
    ) -> list[Facet]:
        if len(facets) < 2:
            return list(facets)
        canonical_order = {
            facet_name: index
            for index, facet_name in enumerate(PHASE_FACET_ORDER.get(phase_name, ()))
        }
        weights = self._phase_priority_weights(
            phase_name,
            facets,
            system_pressure=system_pressure,
        )
        indexed_facets = list(enumerate(facets))
        indexed_facets.sort(
            key=lambda item: (
                canonical_order.get(
                    self._facet_name(item[1]).strip().casefold(),
                    len(canonical_order),
                ),
                -weights.get(self._facet_name(item[1]), 0.0),
                item[0],
            )
        )
        return [facet for _, facet in indexed_facets]

    def _phase_priority_weights(
        self,
        phase_name: str,
        facets: list[Facet],
        *,
        system_pressure: float,
    ) -> dict[str, float]:
        pressure = self._clamp_unit(system_pressure)
        low_pressure = 1.0 - pressure
        weights: dict[str, float] = {}
        for facet in facets:
            facet_name = self._facet_name(facet)
            normalized = facet_name.strip().casefold()
            weight = 0.0
            if phase_name in {"decision", "planning_execution"} and normalized in {
                "behavior",
                "planner",
                "executor",
            }:
                weight = round(pressure * 0.05, 3)
            elif phase_name == "learning_signal" and normalized == "learning":
                weight = round(low_pressure * 0.05, 3)
            elif phase_name == "input_context" and normalized == "memory":
                # Context intentionally keeps its registered order; the trace still
                # records that low pressure would favor memory in future versions.
                weight = round(low_pressure * 0.01, 3)
            weights[facet_name] = weight
        return weights

    @staticmethod
    def _facet_output_summary(result: FacetResult) -> dict[str, Any]:
        return {
            "facet_name": result.facet_name,
            "proposed_decision": (
                result.proposed_decision.value if result.proposed_decision else None
            ),
            "state_updated": bool(result.state_updates),
        }

    def _aggregate_pressure(
        self,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
    ) -> float:
        signals: list[float] = []
        event_pressure = self._numeric_unit(event.metadata.get("pressure"))
        if event_pressure is not None:
            signals.append(event_pressure)
        attention_peak = self._attention_peak_from_results(facet_results)
        if attention_peak is None:
            attention_peak = self._attention_peak_from_state(state)
        if attention_peak is not None:
            signals.append(attention_peak)
        affect_arousal = self._affect_arousal_from_results(facet_results)
        if affect_arousal is None:
            affect_arousal = self._affect_arousal_from_state(state)
        if affect_arousal is not None:
            signals.append(affect_arousal)
        learning_signal = self._learning_pressure_from_results(facet_results)
        if learning_signal is not None:
            signals.append(learning_signal)
        if not signals:
            return 0.0
        return round(self._clamp_unit(sum(signals) / len(signals)), 3)

    @staticmethod
    def _attention_peak_from_results(facet_results: list[FacetResult]) -> float | None:
        for result in reversed(facet_results):
            if result.facet_name != "attention":
                continue
            peak = Nexus._attention_peak_from_payload(result.metadata)
            if peak is not None:
                return peak
        return None

    @staticmethod
    def _attention_peak_from_state(state: NexusState) -> float | None:
        facet_state = state.facet_state.get("attention")
        if not isinstance(facet_state, dict):
            return None
        peak = Nexus._attention_peak_from_payload(facet_state)
        if peak is not None:
            return peak
        return Nexus._attention_peak_from_payload(facet_state.get("last_attention_result"))

    @staticmethod
    def _attention_peak_from_payload(payload: Any) -> float | None:
        if not isinstance(payload, dict):
            return None
        candidates: list[float] = []
        scores = payload.get("scores") or payload.get("last_scores")
        if isinstance(scores, dict):
            candidates.extend(
                value
                for value in (Nexus._numeric_unit(score) for score in scores.values())
                if value is not None
            )
        for key in ("focus_items", "last_focus_items"):
            raw_items = payload.get(key)
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                score = Nexus._numeric_unit(item.get("score"))
                if score is not None:
                    candidates.append(score)
        return max(candidates) if candidates else None

    @staticmethod
    def _affect_arousal_from_results(facet_results: list[FacetResult]) -> float | None:
        for result in reversed(facet_results):
            if result.facet_name != "affect":
                continue
            arousal = Nexus._affect_arousal_from_payload(result.metadata)
            if arousal is not None:
                return arousal
        return None

    @staticmethod
    def _affect_arousal_from_state(state: NexusState) -> float | None:
        facet_state = state.facet_state.get("affect")
        if not isinstance(facet_state, dict):
            return None
        arousal = Nexus._affect_arousal_from_payload(facet_state)
        if arousal is not None:
            return arousal
        return Nexus._affect_arousal_from_payload(facet_state.get("last_affect_state"))

    @staticmethod
    def _affect_arousal_from_payload(payload: Any) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in ("arousal", "last_arousal"):
            arousal = Nexus._numeric_unit(payload.get(key))
            if arousal is not None:
                return arousal
        for key in ("affect_state", "current_state", "last_affect_state"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                arousal = Nexus._numeric_unit(nested.get("arousal"))
                if arousal is not None:
                    return arousal
        affect_result = payload.get("affect_result") or payload.get("last_affect_result")
        if isinstance(affect_result, dict):
            current_state = affect_result.get("current_state")
            if isinstance(current_state, dict):
                return Nexus._numeric_unit(current_state.get("arousal"))
        return None

    @staticmethod
    def _learning_pressure_from_results(facet_results: list[FacetResult]) -> float | None:
        for result in reversed(facet_results):
            if result.facet_name != "learning":
                continue
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            signals = metadata.get("signals")
            if not isinstance(signals, list) or not signals:
                return None
            magnitudes = [
                value
                for value in (
                    Nexus._numeric_unit(signal.get("magnitude"))
                    for signal in signals
                    if isinstance(signal, dict)
                )
                if value is not None
            ]
            if magnitudes:
                return max(magnitudes)
        return None

    @staticmethod
    def _extract_internal_events(result: FacetResult) -> list[Event]:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        raw_events: list[Any] = []
        if "internal_event" in metadata:
            raw_events.append(metadata.get("internal_event"))
        candidate_events = metadata.get("internal_events")
        if isinstance(candidate_events, list):
            raw_events.extend(candidate_events)
        events: list[Event] = []
        for raw_event in raw_events:
            event = Nexus._coerce_internal_event(raw_event)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _coerce_internal_event(raw_event: Any) -> Event | None:
        if isinstance(raw_event, Event):
            if raw_event.event_type == EventType.INTERNAL:
                return raw_event
            return Event(
                event_type=EventType.INTERNAL,
                content=raw_event.content,
                metadata=dict(raw_event.metadata),
            )
        if isinstance(raw_event, dict):
            payload = dict(raw_event)
            payload["event_type"] = EventType.INTERNAL.value
            if "timestamp" in payload and "event_id" in payload:
                try:
                    return Event.from_dict(payload)
                except (KeyError, ValueError, TypeError):
                    pass
            return Event(
                event_type=EventType.INTERNAL,
                content=str(payload.get("content", "")),
                metadata=(
                    dict(payload.get("metadata", {}))
                    if isinstance(payload.get("metadata"), dict)
                    else {}
                ),
            )
        return None

    @staticmethod
    def _normalize_internal_event_metadata(result: FacetResult) -> None:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if isinstance(metadata.get("internal_event"), Event):
            metadata["internal_event"] = metadata["internal_event"].to_dict()
        raw_events = metadata.get("internal_events")
        if isinstance(raw_events, list):
            metadata["internal_events"] = [
                item.to_dict() if isinstance(item, Event) else item for item in raw_events
            ]

    @staticmethod
    def _numeric_unit(raw_value: Any) -> float | None:
        if isinstance(raw_value, bool):
            return None
        if not isinstance(raw_value, (int, float)):
            return None
        return Nexus._clamp_unit(float(raw_value))

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    def _facet_name(self, facet: Facet) -> str:
        raw_name = getattr(facet, "name", "") or facet.__class__.__name__
        return str(raw_name)

    @staticmethod
    def _is_post_decision_verifier(facet: Facet) -> bool:
        return (
            callable(getattr(facet, "verify", None))
            and str(getattr(facet, "name", "") or "").strip().casefold() == "verifier"
        )

    def _run_verifier(
        self,
        facet: Facet,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
        decision: NexusDecision,
    ) -> FacetResult:
        verify = getattr(facet, "verify", None)
        if not callable(verify):
            return self._run_facet(facet, event, state)
        try:
            return verify(event, state, list(facet_results), decision)
        except Exception as exc:
            facet_name = self._facet_name(facet)
            error_message = str(exc) or "Verifier raised without an error message."
            return FacetResult(
                facet_name=facet_name,
                summary=(
                    f"Verifier '{facet_name}' failed while validating the decision: "
                    f"{error_message}"
                ),
                proposed_decision=DecisionAction.RECORD,
                metadata={
                    "verification_status": "failed",
                    "failed_checks": ["verifier_runtime_error"],
                    "warnings": [],
                    "results": [
                        {
                            "check_name": "verifier_runtime_error",
                            "status": "failed",
                            "severity": "critical",
                            "message": error_message,
                            "metadata": {
                                "recommended_action": DecisionAction.RECORD.value
                            },
                        }
                    ],
                    "reasons": [error_message],
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message,
                },
            )

    @staticmethod
    def _apply_verifier_decision(
        decision: NexusDecision,
        verifier_result: FacetResult,
    ) -> NexusDecision:
        metadata = (
            verifier_result.metadata if isinstance(verifier_result.metadata, dict) else {}
        )
        if metadata.get("verification_status") != "failed":
            metadata["override_applied"] = False
            metadata["override_reason"] = "verification_did_not_fail"
            return decision
        proposed_decision = verifier_result.proposed_decision
        if proposed_decision is None:
            metadata["override_applied"] = False
            metadata["override_reason"] = "no_verifier_proposal"
            return decision
        metadata["current_decision"] = decision.action.value
        metadata["proposed_override_decision"] = proposed_decision.value

        current_priority = DECISION_PRIORITY[decision.action]
        proposed_priority = DECISION_PRIORITY[proposed_decision]
        if proposed_priority > current_priority:
            metadata["override_applied"] = False
            metadata["override_reason"] = "ignored_higher_priority_verifier_proposal"
            return decision
        if proposed_priority == current_priority:
            metadata["override_applied"] = False
            if proposed_decision == decision.action:
                metadata["override_reason"] = "proposed_decision_matches_current"
            else:
                metadata["override_reason"] = "ignored_same_priority_verifier_proposal"
            return decision

        metadata["override_applied"] = True
        metadata["override_reason"] = "risk_reducing_downgrade"
        source_facets = list(decision.source_facets)
        if verifier_result.facet_name not in source_facets:
            source_facets.append(verifier_result.facet_name)
        return NexusDecision(
            action=proposed_decision,
            reason=(
                f"Verifier downgraded {decision.action.value.upper()} to "
                f"{proposed_decision.value.upper()}: {verifier_result.summary}"
            ),
            source_facets=source_facets,
        )

    def _integrate(
        self,
        event: Event,
        facet_results: list[FacetResult],
    ) -> NexusDecision:
        denied_policy_results = self._policy_results(
            facet_results,
            status=PolicyStatus.DENIED,
        )
        if denied_policy_results:
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason=self._policy_reason(
                    denied_policy_results,
                    default="Selected RECORD because policy denied the modeled action.",
                ),
                source_facets=[result.facet_name for result in denied_policy_results],
            )

        approval_policy_results = self._policy_results(
            facet_results,
            status=PolicyStatus.APPROVAL_REQUIRED,
        )
        if approval_policy_results:
            return NexusDecision(
                action=DecisionAction.ASK,
                reason=self._policy_reason(
                    approval_policy_results,
                    default=(
                        "Selected ASK because policy requires approval before the "
                        "modeled action."
                    ),
                ),
                source_facets=[result.facet_name for result in approval_policy_results],
            )

        explicit_results = [
            result for result in facet_results if result.proposed_decision is not None
        ]
        if explicit_results:
            selected_action = max(
                (result.proposed_decision for result in explicit_results),
                key=lambda action: DECISION_PRIORITY[action],
            )
            source_facets = [
                result.facet_name
                for result in explicit_results
                if result.proposed_decision == selected_action
            ]
            reason = (
                f"Selected {selected_action.value.upper()} from facet proposals: "
                f"{', '.join(source_facets)}."
            )
            return NexusDecision(
                action=selected_action,
                reason=reason,
                source_facets=source_facets,
            )

        if event.event_type == EventType.USER_MESSAGE:
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD for a user message event.",
            )

        if any(result.state_updates for result in facet_results):
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD because facets produced state updates.",
            )

        return NexusDecision(
            action=DecisionAction.WAIT,
            reason="Defaulted to WAIT because no facet proposed or updated anything.",
        )

    @staticmethod
    def _policy_results(
        facet_results: list[FacetResult],
        *,
        status: PolicyStatus,
    ) -> list[FacetResult]:
        matches: list[FacetResult] = []
        for result in facet_results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if metadata.get("policy_status") == status.value:
                matches.append(result)
        return matches

    @staticmethod
    def _policy_reason(
        policy_results: list[FacetResult],
        *,
        default: str,
    ) -> str:
        policy_names: list[str] = []
        for result in policy_results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            matched_policies = metadata.get("matched_policies")
            if not isinstance(matched_policies, list):
                continue
            for policy in matched_policies:
                if not isinstance(policy, dict):
                    continue
                name = policy.get("name")
                if isinstance(name, str) and name not in policy_names:
                    policy_names.append(name)
        if not policy_names:
            return default
        return f"{default} Matched policies: {', '.join(policy_names)}."


class NexusRuntime(Nexus):
    """Explicit runtime alias for callers that prefer a runtime-oriented name."""
