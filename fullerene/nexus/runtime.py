"""Minimal Nexus runtime loop."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from fullerene.facets.base import Facet
from fullerene.nexus.models import (
    CycleSignalMap,
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusState,
)
from fullerene.policy.models import PolicyStatus
from fullerene.nexus.interrupts import (
    apply_suppression,
    build_nexus_internal_event,
    extract_interrupt_candidates,
    update_cooldown_entry,
)
from fullerene.expression import ExpressionBudgetState, evaluate_expression_gate
from fullerene.signals.latent_pressure import update_latent_pressure
from fullerene.state.store import InMemoryStateStore, StateStore
from fullerene.verifier.artifacts import validate_expression_gate_v0

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
def _compact_cooldowns_view(store: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted(store.keys(), key=lambda k: str(k))[:40]
    for key in keys:
        raw_v = store.get(key)
        if not isinstance(raw_v, dict):
            continue
        out[str(key)] = {
            kk: raw_v[kk]
            for kk in (
                "cooldown_key",
                "trigger_count",
                "last_triggered_event_count",
                "last_candidate_id",
            )
            if kk in raw_v
        }
    return out


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
        processed_internal_events: list[dict[str, Any]] = []
        if internal_events:
            internal_event = internal_events[0]
            internal_record, _ = self._process_event_cycle(
                internal_event,
                collect_internal_events=False,
            )
            internal_records.append(internal_record)
            processed_internal_events = [internal_event.to_dict()]
            primary_record.metadata["internal_events_processed"] = processed_internal_events
            primary_record.metadata["internal_events_dropped"] = max(
                len(internal_events) - 1,
                0,
            )
            signal_map = self._dict_payload(primary_record.metadata.get("signal_map"))
            signal_map["internal_event_processed"] = True
            primary_record.metadata["signal_map"] = signal_map
            cycle_trace = self._dict_payload(primary_record.metadata.get("cycle_trace"))
            cycle_trace["internal_events_processed"] = list(processed_internal_events)
            primary_record.metadata["cycle_trace"] = cycle_trace
            self.state.facet_state.setdefault("nexus", {})[
                "last_internal_events_processed"
            ] = list(processed_internal_events)

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
        facet_results: list[FacetResult] = []
        internal_events: list[Event] = []
        cycle_learning_events: list[dict[str, Any]] = []
        phase_execution_order: list[dict[str, Any]] = []
        facet_outputs_by_phase: dict[str, list[dict[str, Any]]] = {}
        phase_buckets = self._phase_buckets(self._facets)
        signal_map = self._build_cycle_signal_map(
            event=event,
            state=working_state,
            facet_results=[],
            learning_events=[],
            internal_event_queued=False,
            internal_event_processed=False,
            latent_pressure_total=0.0,
        )
        working_state.system_pressure = signal_map.system_pressure
        pressure_components = dict(signal_map.pressure_components)
        pressure_before = self._clamp_unit(signal_map.system_pressure)
        initial_decision: NexusDecision | None = None
        verifier_adjustments: list[dict[str, Any]] = []
        decision_source_facets: list[str] = []
        internal_events_queued: list[dict[str, Any]] = []

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
            if phase_name == "learning_signal":
                nexus_preview = working_state.facet_state.setdefault("nexus", {})
                nexus_preview["current_cycle_signal_map"] = signal_map.to_dict()
                nexus_preview["current_cycle_learning_events"] = [
                    dict(item) for item in cycle_learning_events
                ]
            for facet in ordered_facets:
                if self._is_post_decision_verifier(facet):
                    continue
                result = self._run_facet(facet, event, working_state)
                emitted_events = self._extract_internal_events(result)
                self._normalize_internal_event_metadata(result)
                if collect_internal_events:
                    internal_events.extend(emitted_events)
                    for queued in emitted_events:
                        internal_events_queued.append(queued.to_dict())
                        signal_map.internal_event_queued = True
                facet_results.append(result)
                cycle_learning_events.extend(self._extract_learning_events(result))
                phase_outputs.append(self._facet_output_summary(result))
                self._apply_result_to_state(working_state, result)
                signal_map = self._build_cycle_signal_map(
                    event=event,
                    state=working_state,
                    facet_results=facet_results,
                    learning_events=cycle_learning_events,
                    internal_event_queued=bool(internal_events_queued),
                    internal_event_processed=False,
                    latent_pressure_total=self._latest_latent_pressure_total(working_state),
                )
                working_state.system_pressure = signal_map.system_pressure
                pressure_components = dict(signal_map.pressure_components)
            facet_outputs_by_phase[phase_label] = phase_outputs
        decision = self._integrate(event, facet_results)
        initial_decision = decision
        decision_source_facets = list(decision.source_facets)

        verifier_facets = [
            facet
            for facet in phase_buckets.get("verification_output", [])
            if self._is_post_decision_verifier(facet)
        ]
        verifier_outputs = facet_outputs_by_phase.setdefault(
            PHASE_LABELS["verification_output"],
            [],
        )
        verifier_preview = working_state.facet_state.setdefault("nexus", {})
        verifier_preview["verifier_cycle_context"] = {
            "signal_map": signal_map.to_dict(),
            "pressure_components": dict(pressure_components),
            "learning_events": [dict(item) for item in cycle_learning_events],
            "internal_events_queued": list(internal_events_queued),
            "internal_events_processed": [],
            "facet_order": [self._facet_name(facet) for facet in self._facets],
            "facet_results_seen": [result.facet_name for result in facet_results],
            "final_decision": decision.action.value,
        }
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
                for queued in emitted_events:
                    internal_events_queued.append(queued.to_dict())
                    signal_map.internal_event_queued = True
            facet_results.append(verifier_result)
            cycle_learning_events.extend(self._extract_learning_events(verifier_result))
            verifier_outputs.append(self._facet_output_summary(verifier_result))
            self._apply_result_to_state(working_state, verifier_result)
            signal_map = self._build_cycle_signal_map(
                event=event,
                state=working_state,
                facet_results=facet_results,
                learning_events=cycle_learning_events,
                internal_event_queued=bool(internal_events_queued),
                internal_event_processed=False,
                latent_pressure_total=self._latest_latent_pressure_total(working_state),
            )
            working_state.system_pressure = signal_map.system_pressure
            pressure_components = dict(signal_map.pressure_components)
            pre_verifier_action = decision.action
            decision = self._apply_verifier_decision(decision, verifier_result)
            if decision.action != pre_verifier_action:
                verifier_adjustments.append(
                    {
                        "from": pre_verifier_action.value,
                        "to": decision.action.value,
                        "reason": verifier_result.summary,
                    }
                )
        lpb_state, lpb_result = update_latent_pressure(
            event=event,
            state=working_state,
            facet_results=facet_results,
        )
        working_state.facet_state.setdefault("signals", {})["latent_pressure"] = lpb_state
        signal_map = self._build_cycle_signal_map(
            event=event,
            state=working_state,
            facet_results=facet_results,
            learning_events=cycle_learning_events,
            internal_event_queued=bool(internal_events_queued),
            internal_event_processed=False,
            latent_pressure_total=lpb_result.latent_pressure_total,
        )
        suppress_summary_text = ""
        interrupt_candidates_out: list[dict[str, Any]] = []
        suppression_decisions_out: list[dict[str, Any]] = []
        interrupt_processed_blob: dict[str, Any] | None = None
        allowed_interrupt_candidate: dict[str, Any] | None = None
        suppressed_interrupts_blob: list[str] = []

        nb = working_state.facet_state.setdefault("nexus", {})
        cooldowns_blob = nb.get("interrupt_cooldowns")
        cooldown_store: dict[str, Any] = (
            dict(cooldowns_blob) if isinstance(cooldowns_blob, dict) else {}
        )
        dup_scope: set[str] = set()
        if event.event_type == EventType.INTERNAL:
            cand_objs = []
        else:
            cand_objs = extract_interrupt_candidates(
                event=event,
                facet_results=facet_results,
                signal_map_dict=signal_map.to_dict(),
                latent_pressure_total=float(lpb_result.latent_pressure_total),
                lpb_ignition_recommended=bool(lpb_result.ignition_recommended),
                lpb_ignition_reason=lpb_result.ignition_reason,
                lpb_ignition_entry_id=getattr(lpb_result, "ignition_entry_id", None),
                lpb_ignition_entry_type=getattr(lpb_result, "ignition_entry_type", None),
            )
        interrupt_candidates_out = [c.to_dict() for c in cand_objs]
        cooldown_seq = int(getattr(working_state, "event_count", 0) or 0) + 1

        winner_obj = None
        if collect_internal_events and event.event_type != EventType.INTERNAL:
            _decisions, winner_obj, suppressed_interrupts_blob = apply_suppression(
                scored_sorted=cand_objs,
                cycle_duplicate_keys=dup_scope,
                context_overloaded=bool(signal_map.context_overloaded),
                cooldowns=cooldown_store,
                current_event_count=cooldown_seq,
            )
            suppression_decisions_out = [d.to_dict() for d in _decisions]
            suppress_summary_text = ";".join(suppressed_interrupts_blob[:12])
            if winner_obj is not None and len(internal_events) == 0:
                nexus_evt = build_nexus_internal_event(event, winner_obj)
                internal_events.append(nexus_evt)
                internal_events_queued.append(nexus_evt.to_dict())
                signal_map.internal_event_queued = True
                ck = winner_obj.cooldown_key
                cooldown_store[ck] = update_cooldown_entry(
                    cooldown_key=ck,
                    candidate_id=winner_obj.id,
                    reason=winner_obj.reason,
                    event_count=cooldown_seq,
                    prior=cooldown_store.get(ck),
                )
                interrupt_processed_blob = {
                    "queued": True,
                    "candidate_id": winner_obj.id,
                    "interrupt_type": winner_obj.interrupt_type,
                    "cooldown_key": ck,
                }
                allowed_interrupt_candidate = winner_obj.to_dict()
            else:
                if winner_obj is not None:
                    allowed_interrupt_candidate = winner_obj.to_dict()
                    interrupt_processed_blob = {
                        "queued": False,
                        "blocked_by_explicit_internal_queue": True,
                        "candidate_id": winner_obj.id,
                    }

        nb["interrupt_cooldowns"] = cooldown_store
        nb["last_interrupt_candidates"] = list(interrupt_candidates_out)
        nb["last_suppression_decisions"] = list(suppression_decisions_out)
        nb["last_allowed_interrupt_candidate"] = allowed_interrupt_candidate
        nb["last_suppressed_interrupts"] = list(suppressed_interrupts_blob)
        nb["last_interrupt_processed"] = interrupt_processed_blob

        ih = list(nb.get("interrupt_history") or [])
        if not isinstance(ih, list):
            ih = []
        ih.append(
            {
                "event_id": event.event_id,
                "candidate_ids": [c["id"] for c in interrupt_candidates_out if c.get("id")],
                "winner_id": winner_obj.id if winner_obj else None,
            }
        )
        nb["interrupt_history"] = ih[-20:]
        nb["interrupt_queue_size"] = min(len(interrupt_candidates_out), 5)

        signal_map = self._build_cycle_signal_map(
            event=event,
            state=working_state,
            facet_results=facet_results,
            learning_events=cycle_learning_events,
            internal_event_queued=bool(internal_events_queued),
            internal_event_processed=False,
            latent_pressure_total=lpb_result.latent_pressure_total,
        )
        if winner_obj is not None:
            signal_map.interrupt_recommended = True
            signal_map.interrupt_reason = winner_obj.reason
        elif cand_objs:
            top = max(cand_objs, key=lambda c: c.priority)
            signal_map.interrupt_reason = signal_map.interrupt_reason or top.reason

        expression_recommendation: dict[str, Any] | None = None
        expression_budget_state: dict[str, Any] | None = None
        expression_budget_summary: dict[str, Any] | None = None
        expression_score_components: dict[str, Any] | None = None
        prior_eg_raw = nb.get("expression_gate")
        prior_budget = ExpressionBudgetState.from_dict(
            prior_eg_raw.get("budget_state") if isinstance(prior_eg_raw, dict) else None,
        )
        cycle_seq_expr = int(getattr(working_state, "event_count", 0) or 0) + 1
        reco, expr_budget = evaluate_expression_gate(
            event=event,
            decision=decision,
            facet_results=list(facet_results),
            signal_map=signal_map.to_dict(),
            latent_pressure_total=float(lpb_result.latent_pressure_total),
            lpb_ignition_recommended=bool(lpb_result.ignition_recommended),
            interrupt_candidates=list(interrupt_candidates_out),
            allowed_interrupt_candidate=allowed_interrupt_candidate,
            suppression_decisions=list(suppression_decisions_out),
            budget=prior_budget,
            cycle_wall_time=event.timestamp,
            cycle_seq=cycle_seq_expr,
        )
        expression_recommendation = reco.to_dict()
        expression_budget_state = expr_budget.to_dict()
        expression_budget_summary = expr_budget.compact_summary()
        meta_expr = reco.metadata if isinstance(reco.metadata, dict) else {}
        expression_score_components = meta_expr.get("score_components")
        eg_store: dict[str, Any] = {
            "last_recommendation": expression_recommendation,
            "budget_state": expression_budget_state,
            "expression_history": list(expr_budget.history),
        }
        nb["expression_gate"] = eg_store

        gate_rows = validate_expression_gate_v0(expression_recommendation)
        for fr in reversed(facet_results):
            if fr.facet_name != "verifier" or not isinstance(fr.metadata, dict):
                continue
            merged_checks = list(fr.metadata.get("artifact_checks") or [])
            merged_checks.extend(gate_rows)
            fr.metadata["artifact_checks"] = merged_checks
            sm = fr.metadata.get("summary_metadata")
            if isinstance(sm, dict):
                sm2 = dict(sm)
                sm_ac = list(sm2.get("artifact_checks") or [])
                sm_ac.extend(gate_rows)
                sm2["artifact_checks"] = sm_ac
                fr.metadata["summary_metadata"] = sm2
            break

        system_pressure = signal_map.system_pressure
        pressure_components = dict(signal_map.pressure_components)
        working_state.system_pressure = system_pressure
        self.state = working_state
        self.state.apply(
            event,
            facet_results,
            decision,
            system_pressure=system_pressure,
        )
        cycle_trace = {
            "event_id": event.event_id,
            "facet_order": [self._facet_name(facet) for facet in self._facets],
            "facet_results_seen": [result.facet_name for result in facet_results],
            "initial_decision": (
                initial_decision.action.value if initial_decision is not None else None
            ),
            "final_decision": decision.action.value,
            "pressure_before": round(pressure_before, 3),
            "pressure_after": round(system_pressure, 3),
            "pressure_components": dict(pressure_components),
            "signal_map": signal_map.to_dict(),
            "latent_pressure_result": lpb_result.to_dict(),
            "learning_events": list(cycle_learning_events),
            "internal_events_queued": list(internal_events_queued),
            "internal_events_processed": [],
            "verifier_adjustments": verifier_adjustments,
            "decision_source_facets": list(decision_source_facets),
            "interrupt_candidates": list(interrupt_candidates_out),
            "suppression_decisions": list(suppression_decisions_out),
            "allowed_interrupt_candidate": allowed_interrupt_candidate,
            "suppressed_interrupts": list(suppressed_interrupts_blob),
            "interrupt_cooldowns": _compact_cooldowns_view(cooldown_store),
            "interrupt_queue_size": int(nb.get("interrupt_queue_size") or 0),
            "interrupt_processed": interrupt_processed_blob,
            "suppression_summary": suppress_summary_text,
            "expression_recommendation": expression_recommendation,
            "expression_score": (
                expression_recommendation.get("expression_score")
                if expression_recommendation
                else None
            ),
            "expression_mode": (
                expression_recommendation.get("mode")
                if expression_recommendation
                else None
            ),
            "expression_suppressed": (
                expression_recommendation.get("suppressed")
                if expression_recommendation
                else None
            ),
            "expression_suppression_reason": (
                expression_recommendation.get("suppression_reason")
                if expression_recommendation
                else None
            ),
            "expression_budget_summary": expression_budget_summary,
            "expression_score_components": expression_score_components,
        }
        nexus_state_updates = {
            "last_cycle_signal_map": signal_map.to_dict(),
            "last_cycle_trace": cycle_trace,
            "last_latent_pressure_result": lpb_result.to_dict(),
            "last_system_pressure": round(system_pressure, 3),
            "last_pressure_components": dict(pressure_components),
            "last_learning_events": list(cycle_learning_events),
            "last_internal_events_queued": list(internal_events_queued),
            "last_internal_events_processed": [],
            "interrupt_cooldowns": dict(cooldown_store),
            "interrupt_queue_size": int(nb.get("interrupt_queue_size") or 0),
            "last_expression_recommendation": expression_recommendation,
            "expression_budget_state": expression_budget_state,
            "expression_history": (
                eg_store.get("expression_history") if expression_recommendation else None
            ),
        }
        self.state.facet_state.setdefault("nexus", {}).update(
            {
                **nexus_state_updates,
                "expression_gate": eg_store,
            }
        )

        record = NexusRecord(
            event=event,
            facet_results=facet_results,
            decision=decision,
            metadata={
                "system_pressure": system_pressure,
                "pressure_components": dict(pressure_components),
                "signal_map": signal_map.to_dict(),
                "latent_pressure": signal_map.latent_pressure,
                "latent_pressure_result": lpb_result.to_dict(),
                "top_latent_pressure_entries": lpb_result.to_dict().get("top_entries", []),
                "latent_pressure_ignition_recommended": lpb_result.ignition_recommended,
                "latent_pressure_ignition_reason": lpb_result.ignition_reason,
                "cycle_learning_events": list(cycle_learning_events),
                "learning_event_count": len(cycle_learning_events),
                "cycle_trace": cycle_trace,
                "phase_execution_order": phase_execution_order,
                "facet_outputs_by_phase": facet_outputs_by_phase,
                "internal_events_processed": [],
                "interrupt_candidates": list(interrupt_candidates_out),
                "suppression_decisions": list(suppression_decisions_out),
                "allowed_interrupt_candidate": allowed_interrupt_candidate,
                "suppressed_interrupts": list(suppressed_interrupts_blob),
                "interrupt_cooldowns": _compact_cooldowns_view(cooldown_store),
                "interrupt_queue_size": int(nb.get("interrupt_queue_size") or 0),
                "interrupt_processed": interrupt_processed_blob,
                "suppression_summary": suppress_summary_text,
                "expression_recommendation": expression_recommendation,
                "expression_score": (
                    expression_recommendation.get("expression_score")
                    if expression_recommendation
                    else None
                ),
                "expression_mode": (
                    expression_recommendation.get("mode")
                    if expression_recommendation
                    else None
                ),
                "expression_suppressed": (
                    expression_recommendation.get("suppressed")
                    if expression_recommendation
                    else None
                ),
                "expression_suppression_reason": (
                    expression_recommendation.get("suppression_reason")
                    if expression_recommendation
                    else None
                ),
                "expression_budget_summary": expression_budget_summary,
                "expression_score_components": expression_score_components,
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

    def _build_cycle_signal_map(
        self,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
        learning_events: list[dict[str, Any]],
        internal_event_queued: bool,
        internal_event_processed: bool,
        latent_pressure_total: float,
    ) -> CycleSignalMap:
        behavior_metadata = self._latest_facet_metadata(facet_results, "behavior")
        attention_metadata = self._latest_facet_metadata(facet_results, "attention")
        planner_metadata = self._latest_facet_metadata(facet_results, "planner")
        verifier_metadata = self._latest_facet_metadata(facet_results, "verifier")
        policy_metadata = self._latest_facet_metadata(facet_results, "policy")
        context_metadata = self._latest_facet_metadata(facet_results, "context")

        event_pressure = self._numeric_unit(event.metadata.get("pressure")) or 0.0
        attention_pressure = self._resolve_attention_pressure(
            attention_metadata=attention_metadata,
            state=state,
        )
        latent_pressure = self._clamp_unit(latent_pressure_total)
        context_load_ratio, context_overloaded = self._resolve_context_load(
            context_metadata=context_metadata,
            behavior_metadata=behavior_metadata,
            state=state,
        )
        belief_contradiction = bool(behavior_metadata.get("belief_contradiction", False))
        contradiction_pressure = 0.15 if belief_contradiction else 0.0
        context_overload_pressure = 0.1 if context_overloaded else 0.0
        interrupt_recommended = bool(
            behavior_metadata.get("interrupt_recommended", False)
        )
        interrupt_pressure = 0.1 if interrupt_recommended else 0.0
        pressure_components = {
            "event_pressure": round(event_pressure, 3),
            "attention_pressure": round(attention_pressure, 3),
            "latent_pressure": round(latent_pressure, 3),
            "contradiction_pressure": round(contradiction_pressure, 3),
            "context_overload_pressure": round(context_overload_pressure, 3),
            "interrupt_pressure": round(interrupt_pressure, 3),
        }
        system_pressure = round(
            self._clamp_unit(sum(pressure_components.values())),
            3,
        )

        policy_status = str(
            policy_metadata.get("policy_status")
            or behavior_metadata.get("policy_result")
            or "unknown"
        )
        verifier_status = str(verifier_metadata.get("verification_status") or "unknown")
        verifier_downgraded = bool(verifier_metadata.get("override_applied", False))

        return CycleSignalMap(
            event_id=event.event_id,
            event_type=event.event_type.value,
            event_pressure=event_pressure,
            system_pressure=system_pressure,
            pressure_components=pressure_components,
            attention_pressure=attention_pressure,
            latent_pressure=latent_pressure,
            context_load_ratio=context_load_ratio,
            context_overloaded=context_overloaded,
            interrupt_recommended=interrupt_recommended,
            interrupt_reason=behavior_metadata.get("interrupt_reason"),
            policy_status=policy_status,
            policy_blocks_act=bool(behavior_metadata.get("policy_blocks_act", False)),
            policy_requires_approval=bool(
                behavior_metadata.get("policy_requires_approval", False)
            ),
            verifier_status=verifier_status,
            verifier_downgraded=verifier_downgraded,
            goal_relevance=float(behavior_metadata.get("goal_relevance", 0.0) or 0.0),
            memory_relevance=float(
                behavior_metadata.get("retrieval_strength", 0.0) or 0.0
            ),
            belief_confidence=float(
                behavior_metadata.get("belief_confidence", 0.0) or 0.0
            ),
            belief_contradiction=belief_contradiction,
            planner_grounding_status=str(
                planner_metadata.get("grounding_status", "unknown")
            ),
            planner_grounding_score=float(
                planner_metadata.get("grounding_score", 0.0) or 0.0
            ),
            learning_event_count=float(len(learning_events)),
            internal_event_queued=internal_event_queued,
            internal_event_processed=internal_event_processed,
        )

    @staticmethod
    def _latest_facet_metadata(
        facet_results: list[FacetResult],
        facet_name: str,
    ) -> dict[str, Any]:
        for result in reversed(facet_results):
            if result.facet_name != facet_name:
                continue
            if isinstance(result.metadata, dict):
                return result.metadata
        return {}

    def _resolve_attention_pressure(
        self,
        *,
        attention_metadata: dict[str, Any],
        state: NexusState,
    ) -> float:
        contribution = self._numeric_unit(attention_metadata.get("pressure_contribution"))
        if contribution is not None:
            return contribution
        attention_state = state.facet_state.get("attention")
        if isinstance(attention_state, dict):
            contribution = self._numeric_unit(
                attention_state.get("last_attention_pressure_contribution")
            )
            if contribution is not None:
                return contribution
        peak = self._attention_peak_from_payload(attention_metadata)
        if peak is not None:
            return round(peak * 0.2, 3)
        return 0.0

    @staticmethod
    def _latest_latent_pressure_total(state: NexusState) -> float:
        signals = state.facet_state.get("signals")
        if not isinstance(signals, dict):
            return 0.0
        latent = signals.get("latent_pressure")
        if not isinstance(latent, dict):
            return 0.0
        result = latent.get("last_result")
        if isinstance(result, dict):
            return Nexus._clamp_unit(result.get("latent_pressure_total", 0.0))
        return 0.0

    def _resolve_context_load(
        self,
        *,
        context_metadata: dict[str, Any],
        behavior_metadata: dict[str, Any],
        state: NexusState,
    ) -> tuple[float, bool]:
        context_load = behavior_metadata.get("context_load")
        if not isinstance(context_load, dict):
            context_load = context_metadata.get("context_load")
        if isinstance(context_load, dict):
            ratio = self._numeric_unit(context_load.get("load_ratio")) or 0.0
            overloaded = bool(context_load.get("overloaded", False))
            return ratio, overloaded
        context_state = state.facet_state.get("context")
        if isinstance(context_state, dict):
            ratio = self._numeric_unit(context_state.get("last_context_load_ratio")) or 0.0
            overloaded = bool(context_state.get("last_context_overloaded", False))
            return ratio, overloaded
        return 0.0, False

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
    def _extract_learning_events(result: FacetResult) -> list[dict[str, Any]]:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        learning_event = metadata.get("learning_event")
        if isinstance(learning_event, dict):
            return [dict(learning_event)]
        learning_events = metadata.get("learning_events")
        if isinstance(learning_events, list):
            return [dict(item) for item in learning_events if isinstance(item, dict)]
        return []

    @staticmethod
    def _dict_payload(raw_value: Any) -> dict[str, Any]:
        return dict(raw_value) if isinstance(raw_value, dict) else {}

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
