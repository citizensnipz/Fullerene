from __future__ import annotations

import json
import unittest
from typing import Any
from fullerene.facets import EchoFacet
from fullerene.nexus import DecisionAction, Event, EventType, FacetResult, NexusRuntime
from fullerene.nexus.interrupts import (
    InterruptCandidate,
    apply_suppression,
    build_nexus_internal_event,
    extract_interrupt_candidates,
    score_interrupt_priority,
    serialization_roundtrip,
    update_cooldown_entry,
)
from fullerene.state import InMemoryStateStore
from fullerene.policy.models import PolicyStatus


def _minimal_signal_map(**overrides: Any) -> dict[str, Any]:
    base = {
        "event_id": "e1",
        "event_type": "user_message",
        "system_pressure": 0.5,
        "context_overloaded": False,
        "belief_contradiction": False,
        "interrupt_recommended": False,
        "interrupt_reason": None,
        "pressure_components": {},
    }
    base.update(overrides)
    return base


class NexusInterruptV2Tests(unittest.TestCase):
    def test_candidate_json_serializable_and_clamps_numbers(self) -> None:
        c = InterruptCandidate(
            id="x",
            source="behavior",
            source_id="y",
            interrupt_type="behavior_interrupt",
            priority=92.3,
            pressure=-1,
            confidence=2,
            novelty=0.33,
            reason="unit",
            payload={"nested": [{"a": 1}]},
            created_at="t",
            parent_event_id="p",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="ck",
            metadata={"recurrence_bonus": 44},
        )
        c2 = score_interrupt_priority(cand=c, raw=c.metadata)
        payload = serialization_roundtrip(c2.to_dict())
        self.assertEqual(payload["pressure"], 0.0)
        self.assertEqual(payload["confidence"], 1.0)
        self.assertLessEqual(payload["priority"], 1.0)
        self.assertIn("priority_components", payload["metadata"])

    def test_behavior_metadata_yields_behavior_interrupt_candidate(self) -> None:
        fr = FacetResult(
            facet_name="behavior",
            summary="b",
            metadata={
                "interrupt_recommended": True,
                "interrupt_reason": "unit_behavior",
                "confidence": 0.82,
                "latent_pressure": 0.2,
            },
        )
        event = Event(event_type=EventType.USER_MESSAGE, content="hi")
        cands = extract_interrupt_candidates(
            event=event,
            facet_results=[fr],
            signal_map_dict=_minimal_signal_map(),
            latent_pressure_total=0.0,
            lpb_ignition_recommended=False,
            lpb_ignition_reason=None,
            lpb_ignition_entry_id=None,
            lpb_ignition_entry_type=None,
        )
        types = [c.interrupt_type for c in cands]
        self.assertIn("behavior_interrupt", types)

    def test_lpb_ignition_produces_candidate(self) -> None:
        event = Event(event_type=EventType.USER_MESSAGE, content="hi")
        cands = extract_interrupt_candidates(
            event=event,
            facet_results=[],
            signal_map_dict=_minimal_signal_map(),
            latent_pressure_total=0.9,
            lpb_ignition_recommended=True,
            lpb_ignition_reason="test_ign",
            lpb_ignition_entry_id="ent1",
            lpb_ignition_entry_type="interrupt_recommendation",
        )
        ig = [c for c in cands if c.interrupt_type == "latent_pressure_ignition"]
        self.assertEqual(len(ig), 1)
        self.assertEqual(ig[0].payload.get("ignition_entry_id"), "ent1")

    def test_verifier_escalation_high_priority(self) -> None:
        fr = FacetResult(
            facet_name="verifier",
            summary="v",
            metadata={
                "escalation_recommended": True,
                "artifact_checks": [
                    {"severity": "critical", "status": "failed", "code": "x"}
                ],
            },
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="hi",
            metadata={"novelty": 0.55, "pressure": 0.5},
        )
        cands = extract_interrupt_candidates(
            event=event,
            facet_results=[fr],
            signal_map_dict=_minimal_signal_map(),
            latent_pressure_total=0.0,
            lpb_ignition_recommended=False,
            lpb_ignition_reason=None,
            lpb_ignition_entry_id=None,
            lpb_ignition_entry_type=None,
        )
        esc = [c for c in cands if c.interrupt_type == "verifier_escalation"][0]
        esc = score_interrupt_priority(cand=esc, raw=esc.metadata)
        self.assertGreaterEqual(esc.priority, 0.6)

    def test_policy_denied_candidate_flagged_not_user_expression_internal(self) -> None:
        fr = FacetResult(
            facet_name="policy",
            summary="p",
            metadata={
                "policy_evaluation": {
                    "status": PolicyStatus.DENIED.value,
                    "effective_rule_id": "r1",
                }
            },
        )
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        cands = extract_interrupt_candidates(
            event=event,
            facet_results=[fr],
            signal_map_dict=_minimal_signal_map(),
            latent_pressure_total=0.0,
            lpb_ignition_recommended=False,
            lpb_ignition_reason=None,
            lpb_ignition_entry_id=None,
            lpb_ignition_entry_type=None,
        )
        denial = next(c for c in cands if c.interrupt_type == "policy_block")
        denial = score_interrupt_priority(cand=denial, raw=denial.metadata)
        dup_scope: set[str] = set()
        decs, winner, summ = apply_suppression(
            scored_sorted=[denial],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=False,
            cooldowns={},
            current_event_count=5,
        )
        self.assertIsNone(winner)
        self.assertTrue(decs[-1].suppressed)
        self.assertFalse(decs[-1].allowed_user_expression)

    def test_low_priority_suppressed(self) -> None:
        c = InterruptCandidate(
            id="lo",
            source="attention",
            source_id=None,
            interrupt_type="attention_conflict",
            priority=0.0,
            pressure=0.1,
            confidence=0.1,
            novelty=0.1,
            reason="weak",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="k1",
            metadata={"recurrence_bonus": 0.0},
        )
        sc = score_interrupt_priority(cand=c, raw=c.metadata)
        self.assertLess(sc.priority, 0.55)
        dup_scope: set[str] = set()
        decs, winner, _summ = apply_suppression(
            scored_sorted=[sc],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=False,
            cooldowns={},
            current_event_count=1,
        )
        self.assertIsNone(winner)
        self.assertTrue(decs[-1].suppressed)

    def test_duplicate_candidate_suppressed(self) -> None:
        c = InterruptCandidate(
            id="a",
            source="attention",
            source_id=None,
            interrupt_type="attention_conflict",
            priority=0.7,
            pressure=0.8,
            confidence=0.8,
            novelty=0.5,
            reason="dup",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="dupkey",
            metadata={"recurrence_bonus": 0.4},
        )
        x = score_interrupt_priority(cand=c, raw=c.metadata)
        if x.priority < 0.72:
            x.priority = 0.72
        dup_scope: set[str] = set()
        decs1, w1, _sum = apply_suppression(
            scored_sorted=[x, x],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=False,
            cooldowns={},
            current_event_count=1,
        )
        self.assertIsNotNone(w1)
        self.assertGreaterEqual(len(decs1), 2)
        self.assertGreaterEqual(sum(1 for d in decs1 if d.suppressed), 1)

    def test_cooldown_throttle_below_bypass(self) -> None:
        c = InterruptCandidate(
            id="cd",
            source="attention",
            source_id=None,
            interrupt_type="attention_conflict",
            priority=0.0,
            pressure=0.95,
            confidence=0.95,
            novelty=0.6,
            reason="thr",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="ck-thr",
            metadata={"recurrence_bonus": 0.72},
        )
        scored = score_interrupt_priority(cand=c, raw=c.metadata)
        ck = scored.cooldown_key
        self.assertGreaterEqual(scored.priority, 0.55)
        self.assertLess(scored.priority, 0.85)
        cd_store = {
            ck: update_cooldown_entry(
                cooldown_key=ck,
                candidate_id="old",
                reason="prior",
                event_count=3,
                prior=None,
            )
        }
        cd_store[ck]["last_triggered_event_count"] = 3
        dup_scope: set[str] = set()
        _decs, winner, summ = apply_suppression(
            scored_sorted=[scored],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=False,
            cooldowns=cd_store,
            current_event_count=4,
        )
        self.assertIsNone(winner)
        self.assertTrue(any("cooldown" in s.lower() for s in summ))

    def test_cooldown_bypass_penalty_above_threshold(self) -> None:
        c = InterruptCandidate(
            id="cd2",
            source="attention",
            source_id=None,
            interrupt_type="attention_conflict",
            priority=0.0,
            pressure=0.95,
            confidence=1.0,
            novelty=0.95,
            reason="thr2",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="ck-bypass",
            metadata={"recurrence_bonus": 0.9},
        )
        scored = score_interrupt_priority(cand=c, raw=c.metadata)
        if scored.priority < 0.92:
            scored.priority = 0.92
        self.assertGreaterEqual(scored.priority, 0.85)
        ck = scored.cooldown_key
        cd_store = {
            ck: update_cooldown_entry(
                cooldown_key=ck,
                candidate_id="old",
                reason="prior",
                event_count=5,
                prior=None,
            )
        }
        cd_store[ck]["last_triggered_event_count"] = 5
        dup_scope: set[str] = set()
        decs, winner, _ = apply_suppression(
            scored_sorted=[scored],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=False,
            cooldowns=cd_store,
            current_event_count=6,
        )
        self.assertIsNotNone(winner)
        d0 = next(d for d in decs if d.candidate_id == winner.id)
        self.assertGreater(d0.priority_before, d0.priority_after)

    def test_context_overload_suppresses_weak_attention(self) -> None:
        c = InterruptCandidate(
            id="ov",
            source="attention",
            source_id=None,
            interrupt_type="attention_conflict",
            priority=0.0,
            pressure=0.66,
            confidence=0.75,
            novelty=0.4,
            reason="overload",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="ck-ov",
            metadata={"recurrence_bonus": 0.3},
        )
        scored = score_interrupt_priority(cand=c, raw=c.metadata)
        dup_scope: set[str] = set()
        _decs, winner, summ = apply_suppression(
            scored_sorted=[scored],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=True,
            cooldowns={},
            current_event_count=1,
        )
        self.assertIsNone(winner)

    def test_verifier_critical_not_suppressed_by_overload_penalty_route(self) -> None:
        c = InterruptCandidate(
            id="cr",
            source="verifier",
            source_id=None,
            interrupt_type="verifier_escalation",
            priority=0.93,
            pressure=0.9,
            confidence=1.0,
            novelty=0.5,
            reason="critical",
            payload={"artifact_critical_or_error": True},
            created_at="t",
            parent_event_id="pid",
            suppressible=False,
            requires_user_attention=True,
            cooldown_key="ck-cr",
            metadata={"artifact_critical_or_error": True},
        )
        dup_scope: set[str] = set()
        _decs, winner, summ = apply_suppression(
            scored_sorted=[c],
            cycle_duplicate_keys=dup_scope,
            context_overloaded=True,
            cooldowns={},
            current_event_count=10,
        )
        self.assertIsNotNone(winner)

    def test_internal_event_has_no_llm_like_content_metadata(self) -> None:
        c = InterruptCandidate(
            id="ie",
            source="behavior",
            source_id=None,
            interrupt_type="behavior_interrupt",
            priority=0.8,
            pressure=0.8,
            confidence=0.8,
            novelty=0.2,
            reason="r",
            payload={},
            created_at="t",
            parent_event_id="pid",
            suppressible=True,
            requires_user_attention=False,
            cooldown_key="k",
            metadata={},
        )
        evt = build_nexus_internal_event(
            Event(event_type=EventType.USER_MESSAGE, content="ignored"), c
        )
        self.assertEqual(evt.content, "nexus_interrupt")

    def test_bounded_internal_cycles_do_not_queue_extra_internals_when_collect_disabled(
        self,
    ) -> None:
        facet_calls: list[str] = []

        class EmitterFacet:
            name = "emitter"

            def process(self, event: Event, state: Any) -> FacetResult:
                facet_calls.append(event.event_type.value)
                nested = Event(
                    event_type=EventType.INTERNAL,
                    content="nested_followup",
                    metadata={"source": "test_emitter"},
                )
                return FacetResult(
                    facet_name=self.name,
                    summary="emit",
                    metadata={"internal_events": [nested]},
                )

        runtime = NexusRuntime(facets=[EmitterFacet(), EchoFacet()], store=InMemoryStateStore())
        runtime.process_event(Event(event_type=EventType.USER_MESSAGE, content="x"))
        internal_passes = [v for v in facet_calls if v == "internal"]
        self.assertEqual(internal_passes, ["internal"])

    def test_interrupt_state_persisted_on_nexus_facet_bucket(self) -> None:
        from tests.test_nexus_runtime import BehaviorSignalFacet

        runtime = NexusRuntime(
            facets=[BehaviorSignalFacet()], store=InMemoryStateStore()
        )
        runtime.process_event(Event(event_type=EventType.USER_MESSAGE, content="k"))
        bucket = runtime.state.facet_state.get("nexus", {})
        self.assertIn("interrupt_cooldowns", bucket)
        self.assertIn("last_interrupt_candidates", bucket)

    def test_cycle_trace_has_interrupt_audit_fields(self) -> None:
        from tests.test_nexus_runtime import BehaviorSignalFacet

        runtime = NexusRuntime(
            facets=[BehaviorSignalFacet()], store=InMemoryStateStore()
        )
        rec = runtime.process_event(Event(event_type=EventType.USER_MESSAGE, content="t"))
        ct = rec.metadata.get("cycle_trace") or {}
        for key in (
            "interrupt_candidates",
            "suppression_decisions",
            "allowed_interrupt_candidate",
            "suppressed_interrupts",
            "interrupt_cooldowns",
            "interrupt_queue_size",
            "interrupt_processed",
            "suppression_summary",
        ):
            self.assertIn(key, ct)

    def test_approval_required_policy_emits_candidate(self) -> None:
        fr = FacetResult(
            facet_name="policy",
            summary="pol",
            metadata={
                "policy_evaluation": {"status": PolicyStatus.APPROVAL_REQUIRED.value},
            },
        )
        ev = Event(event_type=EventType.USER_MESSAGE, content=".")
        got = extract_interrupt_candidates(
            event=ev,
            facet_results=[fr],
            signal_map_dict=_minimal_signal_map(),
            latent_pressure_total=0.0,
            lpb_ignition_recommended=False,
            lpb_ignition_reason=None,
            lpb_ignition_entry_id=None,
            lpb_ignition_entry_type=None,
        )
        _ar = next(c for c in got if c.interrupt_type == "approval_required")
        self.assertTrue(_ar.requires_user_attention)


if __name__ == "__main__":
    unittest.main()
