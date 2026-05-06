"""Tests for Presentation Vector v0 (read-only projection, not a facet)."""

from __future__ import annotations

import copy
import json
import unittest

from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
)
from fullerene.presentation import (
    PresentationChannel,
    PresentationMode,
    PresentationMotion,
    PresentationVector,
    derive_presentation_vector,
    derive_presentation_vector_from_summary,
)
from fullerene.verifier.artifacts import validate_presentation_vector_v0


class PresentationModelTests(unittest.TestCase):
    def test_round_trip_and_numeric_clamping(self) -> None:
        pv = PresentationVector(
            mode=PresentationMode.thinking,
            intensity=99.0,
            motion=PresentationMotion.ellipsis,
            channel=PresentationChannel.internal,
            user_attention_needed=True,
            expression_active=False,
            expression_mode="silent",
            pressure=-1,
            latent_pressure=2,
            confidence=3,
            novelty=-2,
            attention_motion=0.4,
            blocked=False,
            overloaded=False,
            warning=False,
            speaking=False,
            thinking=True,
            idle=False,
            face_state="neutral",
            eye_state="unknown",
            mouth_state="unknown",
            animation_hint="thinking_ellipsis",
            reason="r",
            reasons=["one"],
            source_event_id="e1",
            source_event_type="system_tick",
            source_record_id="rec1",
            metadata={"nested": {"k": 1}},
        )
        raw = pv.to_dict()
        self.assertGreaterEqual(raw["novelty"], 0.0)
        self.assertLessEqual(raw["intensity"], 1.0)
        r2 = PresentationVector.from_dict(raw)
        self.assertEqual(r2.mode, PresentationMode.thinking)
        json.dumps(raw)


class PresentationDeriveTests(unittest.TestCase):
    def _minimal_record(
        self,
        *,
        event_type: EventType,
        md: dict,
        facets: list[FacetResult] | None = None,
    ) -> NexusRecord:
        ev = Event(event_type=event_type, content="", metadata={})
        return NexusRecord(
            event=ev,
            facet_results=list(facets or []),
            decision=NexusDecision(action=DecisionAction.RECORD, reason="test"),
            metadata=dict(md),
        )

    def test_low_pressure_maps_idle(self) -> None:
        md = {
            "signal_map": {
                "system_pressure": 0.05,
                "latent_pressure": 0.02,
                "context_overloaded": False,
            },
            "interrupt_candidates": [],
        }
        ev = Event(event_type=EventType.INTERNAL, content="", metadata={})
        rec = NexusRecord(
            event=ev,
            facet_results=[],
            decision=NexusDecision(action=DecisionAction.WAIT, reason="idle"),
            metadata=md,
        )
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.idle)
        self.assertEqual(pv.motion.value, "blink")
        self.assertTrue(pv.idle)

    def test_system_tick_moderate_latent_thinking(self) -> None:
        md = {
            "signal_map": {
                "system_pressure": 0.2,
                "latent_pressure": 0.44,
                "context_overloaded": False,
            },
            "latent_pressure": 0.44,
        }
        ev = Event(event_type=EventType.SYSTEM_TICK, content="", metadata={})
        rec = NexusRecord(
            event=ev,
            facet_results=[],
            decision=NexusDecision(action=DecisionAction.RECORD, reason="t"),
            metadata=md,
        )
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.thinking)
        self.assertEqual(pv.motion.value, "ellipsis")
        self.assertTrue(pv.thinking)

    def test_ask_user_clarification_speaking(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.3, "latent_pressure": 0.1},
            "expression_recommendation": {
                "mode": "ask_user",
                "expression_score": 0.7,
                "suppressed": False,
                "suggested_intent": "ask_clarification",
            },
        }
        rec = self._minimal_record(event_type=EventType.USER_MESSAGE, md=md)
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.speaking)
        self.assertTrue(pv.user_attention_needed)
        self.assertTrue(pv.expression_active)

    def test_policy_approval_blocked(self) -> None:
        md = {
            "signal_map": {
                "system_pressure": 0.2,
                "latent_pressure": 0.1,
                "policy_status": "approval_required",
                "policy_requires_approval": True,
            },
        }
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md)
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.blocked)
        self.assertTrue(pv.user_attention_needed)
        self.assertTrue(pv.blocked)

    def test_verifier_critical_warning(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.2, "latent_pressure": 0.1},
        }
        fr = FacetResult(
            facet_name="verifier",
            summary="v",
            metadata={
                "artifact_checks": [
                    {
                        "status": "failed",
                        "severity": "critical",
                        "code": "x",
                    }
                ],
            },
        )
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md, facets=[fr])
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.warning)
        self.assertTrue(pv.warning)
        self.assertGreaterEqual(pv.intensity, 0.75)

    def test_context_overload_overloaded(self) -> None:
        md = {
            "signal_map": {
                "system_pressure": 0.2,
                "latent_pressure": 0.1,
                "context_overloaded": True,
                "attention_pressure": 0.0,
            },
        }
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md)
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.overloaded)
        self.assertEqual(pv.motion.value, "jitter")

    def test_learning_applied(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.1, "latent_pressure": 0.1},
        }
        fr = FacetResult(
            facet_name="learning",
            summary="l",
            metadata={
                "learning_result": {
                    "applied": [{"id": "a1"}],
                    "metadata": {"cross_facet_routes": []},
                },
            },
        )
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md, facets=[fr])
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.learning)

    def test_suppressed_no_expression_active(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.2, "latent_pressure": 0.1},
            "expression_recommendation": {
                "mode": "short_utterance",
                "suppressed": True,
                "expression_score": 0.9,
            },
        }
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md)
        pv = derive_presentation_vector(rec)
        self.assertFalse(pv.expression_active)

    def test_priority_warning_beats_learning(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.2, "latent_pressure": 0.1},
        }
        fr_v = FacetResult(
            facet_name="verifier",
            summary="v",
            metadata={
                "artifact_checks": [
                    {"status": "failed", "severity": "critical", "code": "c"},
                ],
            },
        )
        fr_l = FacetResult(
            facet_name="learning",
            summary="l",
            metadata={
                "learning_result": {
                    "applied": [{"id": "x"}],
                    "metadata": {},
                },
            },
        )
        rec = self._minimal_record(
            event_type=EventType.INTERNAL,
            md=md,
            facets=[fr_v, fr_l],
        )
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.warning)

    def test_priority_blocked_beats_thinking(self) -> None:
        md = {
            "signal_map": {
                "system_pressure": 0.5,
                "latent_pressure": 0.5,
                "policy_status": "denied",
            },
        }
        ev = Event(event_type=EventType.SYSTEM_TICK, content="", metadata={})
        rec = NexusRecord(
            event=ev,
            facet_results=[],
            decision=NexusDecision(action=DecisionAction.RECORD, reason="r"),
            metadata=md,
        )
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.blocked)

    def test_priority_speaking_beats_idle(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.05, "latent_pressure": 0.02},
            "expression_recommendation": {
                "mode": "short_utterance",
                "suppressed": False,
                "expression_score": 0.6,
                "suggested_intent": "status_update",
            },
        }
        ev = Event(event_type=EventType.INTERNAL, content="", metadata={})
        rec = NexusRecord(
            event=ev,
            facet_results=[],
            decision=NexusDecision(action=DecisionAction.RECORD, reason="r"),
            metadata=md,
        )
        pv = derive_presentation_vector(rec)
        self.assertEqual(pv.mode, PresentationMode.speaking)

    def test_no_mutation(self) -> None:
        md = {"signal_map": {"system_pressure": 0.1, "latent_pressure": 0.1}}
        ev = Event(event_type=EventType.USER_MESSAGE, content="hi", metadata={"k": 1})
        rec = NexusRecord(
            event=ev,
            facet_results=[],
            decision=NexusDecision(action=DecisionAction.RECORD, reason="r"),
            metadata=md,
        )
        snap_m = copy.deepcopy(rec.metadata)
        snap_e = copy.deepcopy(rec.event.metadata)
        derive_presentation_vector(rec)
        self.assertEqual(rec.metadata, snap_m)
        self.assertEqual(rec.event.metadata, snap_e)

    def test_verifier_schema_helper(self) -> None:
        md = {
            "signal_map": {"system_pressure": 0.2, "latent_pressure": 0.1},
        }
        rec = self._minimal_record(event_type=EventType.INTERNAL, md=md)
        pv = derive_presentation_vector(rec)
        rows = validate_presentation_vector_v0(pv.to_dict())
        self.assertTrue(any(r.get("code") == "ok" for r in rows))


class PresentationSummaryTests(unittest.TestCase):
    def test_from_summary_round_trip(self) -> None:
        base = PresentationVector(
            mode=PresentationMode.idle,
            intensity=0.2,
            motion=PresentationMotion.blink,
            channel=PresentationChannel.internal,
            user_attention_needed=False,
            expression_active=False,
            expression_mode="silent",
            pressure=0.1,
            latent_pressure=0.1,
            confidence=0.5,
            novelty=0.5,
            attention_motion=0.0,
            blocked=False,
            overloaded=False,
            warning=False,
            speaking=False,
            thinking=False,
            idle=True,
            face_state="neutral",
            eye_state="open",
            mouth_state="closed",
            animation_hint="idle_blink",
            reason="r",
            reasons=[],
            source_event_id="e",
            source_event_type="internal",
            source_record_id="r",
            metadata={},
        )
        summary = {"presentation_vector": base.to_dict()}
        pv = derive_presentation_vector_from_summary(summary)
        self.assertEqual(pv.mode, PresentationMode.idle)


if __name__ == "__main__":
    unittest.main()
