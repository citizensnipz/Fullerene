"""Tests for Expression Gate v0 (support infrastructure, not a facet)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from fullerene.expression import (
    ExpressionBudgetState,
    ExpressionMode,
    ExpressionRecommendation,
    evaluate_expression_gate,
)
from fullerene.expression.gate import _mode_from_threshold
from fullerene.expression.models import SuggestedIntent, expression_mode_rank
from fullerene.expression.scoring import ExpressionScoreComponents, compute_expression_score
from fullerene.facets.echo import EchoFacet
from fullerene.facets.verifier import VerifierFacet
from fullerene.nexus import Event, EventType, FacetResult, NexusDecision, NexusRuntime
from fullerene.nexus.models import DecisionAction
from fullerene.verifier.artifacts import validate_expression_gate_v0


def _run_eg(**kwargs):
    suppressions_list = list(kwargs.pop("suppressions", []))
    event = kwargs.pop("event", None) or Event(
        event_type=EventType.USER_MESSAGE, content="hello", metadata={}
    )
    decision = kwargs.pop("decision", None) or NexusDecision(
        action=DecisionAction.RECORD, reason="r"
    )
    facet_results = list(kwargs.pop("facet_results", None) or [])
    signal_map = kwargs.pop(
        "signal_map", None,
    ) or {
        "system_pressure": 0.5,
        "context_overloaded": False,
    }
    latent_pressure_total = float(kwargs.pop("latent_pressure_total", 0.0))
    lpb_ignition = bool(kwargs.pop("lpb_ignition", False))
    interrupt_candidates = list(kwargs.pop("interrupt_candidates", None) or [])
    allowed_interrupt_candidate = kwargs.pop("allowed_interrupt_candidate", None)
    budget = kwargs.pop("budget", None) or ExpressionBudgetState(
        None, 0, None, 0, {}, [],
    )
    cycle_seq = int(kwargs.pop("cycle_seq", 1))
    assert not kwargs, f"unexpected kwargs {kwargs!r}"
    return evaluate_expression_gate(
        event=event,
        decision=decision,
        facet_results=facet_results,
        signal_map=signal_map,
        latent_pressure_total=latent_pressure_total,
        lpb_ignition_recommended=lpb_ignition,
        interrupt_candidates=interrupt_candidates,
        allowed_interrupt_candidate=allowed_interrupt_candidate,
        suppression_decisions=suppressions_list,
        budget=budget,
        cycle_wall_time=event.timestamp,
        cycle_seq=cycle_seq,
    )


class ExpressionModelTests(unittest.TestCase):
    def test_round_trip_and_clamping(self) -> None:
        reco = ExpressionRecommendation(
            mode=ExpressionMode.short_utterance,
            expression_score=1.5,
            reasons=["a"],
            source_event_id="e1",
            source_cycle_id="1",
            source_candidate_id=None,
            allowed_user_facing=True,
            requires_user_attention=True,
            cooldown_applied=False,
            budget_applied=False,
            suppressed=False,
            suppression_reason="",
            max_words=9999,
            suggested_intent=SuggestedIntent.ask_approval,
            payload={"k": "v"},
            metadata={"nested": {"x": 1}},
        )
        raw = reco.to_dict()
        self.assertEqual(raw["expression_score"], 1.0)
        self.assertGreaterEqual(raw["max_words"], 0)
        r2 = ExpressionRecommendation.from_dict(raw)
        self.assertEqual(r2.mode, ExpressionMode.short_utterance)
        json.dumps(raw)

    def test_budget_round_trip(self) -> None:
        b = ExpressionBudgetState(
            last_expression_at="2026-05-06T12:00:00+00:00",
            expression_count_window=3,
            last_expression_hash="abc",
            ignored_expression_count=1,
            cooldowns={
                "c1": {"last_user_facing_at": "2026-05-06T12:00:00+00:00"},
            },
            history=[{"mode": "silent"}],
        )
        b2 = ExpressionBudgetState.from_dict(b.to_dict())
        self.assertEqual(b2.expression_count_window, 3)


class ExpressionScoringTests(unittest.TestCase):
    def test_low_score_silent(self) -> None:
        c = ExpressionScoreComponents(
            system_pressure=0.1,
            latent_pressure=0.1,
            interrupt_priority=0.1,
            verifier_escalation=0.0,
            policy_attention_need=0.0,
            novelty=0.0,
            confidence=0.0,
            repetition_penalty=0.0,
            recent_expression_penalty=0.0,
            context_overload_penalty=0.0,
        )
        s = compute_expression_score(c)
        self.assertLess(s, 0.45)
        self.assertEqual(_mode_from_threshold(s), ExpressionMode.silent)

    def test_medium_score_log_or_status(self) -> None:
        c = ExpressionScoreComponents(
            system_pressure=0.9,
            latent_pressure=0.55,
            interrupt_priority=0.45,
            verifier_escalation=0.0,
            policy_attention_need=0.0,
            novelty=0.25,
            confidence=0.35,
            repetition_penalty=0.0,
            recent_expression_penalty=0.0,
            context_overload_penalty=0.0,
        )
        s = compute_expression_score(c)
        self.assertGreaterEqual(s, 0.45)
        m = _mode_from_threshold(s)
        self.assertIn(m, (ExpressionMode.log_only, ExpressionMode.status_only))


class ExpressionGateScenarioTests(unittest.TestCase):
    def test_high_lpb_ignition_and_allowed_interrupt_can_short_utterance(self) -> None:
        reco, budget = _run_eg(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="pressure",
                metadata={"novelty": 1.0},
            ),
            facet_results=[
                FacetResult(
                    facet_name="behavior",
                    summary="b",
                    metadata={"confidence": 1.0, "ambiguity_score": 0.0},
                ),
            ],
            signal_map={"system_pressure": 1.0, "context_overloaded": False},
            latent_pressure_total=1.0,
            lpb_ignition=True,
            interrupt_candidates=[
                {
                    "id": "ic-test",
                    "interrupt_type": "latent_pressure_ignition",
                    "priority": 1.0,
                    "novelty": 1.0,
                }
            ],
            allowed_interrupt_candidate={
                "id": "ic-test",
                "interrupt_type": "latent_pressure_ignition",
                "priority": 1.0,
                "novelty": 1.0,
            },
            suppressions=[],
        )
        self.assertGreaterEqual(reco.expression_score, 0.75)
        self.assertGreaterEqual(
            expression_mode_rank(reco.mode),
            expression_mode_rank(ExpressionMode.short_utterance),
        )
        self.assertTrue(reco.allowed_user_facing)

    def test_policy_approval_requires_ask_user_when_budget_ok(self) -> None:
        reco, _ = _run_eg(
            facet_results=[
                FacetResult(
                    facet_name="policy",
                    summary="p",
                    metadata={"policy_status": "approval_required"},
                ),
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.8}),
            ],
            decision=NexusDecision(action=DecisionAction.ASK, reason="approval"),
            signal_map={"system_pressure": 0.5, "context_overloaded": False},
            latent_pressure_total=0.5,
        )
        self.assertEqual(reco.suggested_intent, SuggestedIntent.ask_approval)
        self.assertGreaterEqual(
            expression_mode_rank(reco.mode),
            expression_mode_rank(ExpressionMode.short_utterance),
        )

    def test_verifier_escalation_raises_minimum_mode(self) -> None:
        reco, _ = _run_eg(
            facet_results=[
                FacetResult(
                    facet_name="verifier",
                    summary="v",
                    metadata={
                        "escalation_recommended": True,
                        "verification_status": "passed",
                        "results": [{"severity": "critical", "check_name": "x"}],
                    },
                ),
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.55}),
            ],
            signal_map={
                "system_pressure": 0.2,
                "context_overloaded": False,
            },
            latent_pressure_total=0.1,
        )
        self.assertGreaterEqual(
            expression_mode_rank(reco.mode),
            expression_mode_rank(ExpressionMode.short_utterance),
        )

    def test_suppressed_interrupt_kills_pressure_surface(self) -> None:
        reco, _ = _run_eg(
            facet_results=[
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.8}),
            ],
            signal_map={"system_pressure": 0.9, "context_overloaded": False},
            latent_pressure_total=0.92,
            lpb_ignition=True,
            interrupt_candidates=[
                {
                    "id": "only-ignite",
                    "interrupt_type": "latent_pressure_ignition",
                    "priority": 0.92,
                    "novelty": 0.8,
                }
            ],
            allowed_interrupt_candidate=None,
            suppressions=[
                {"candidate_id": "only-ignite", "suppressed": True},
            ],
        )
        self.assertTrue(reco.suppressed)
        self.assertIn("E_nexus_interrupt_suppressed", reco.suppression_rules_triggered)

    def test_internal_blocks_user_facing_by_default(self) -> None:
        ev = Event(
            event_type=EventType.INTERNAL,
            content="nexus_interrupt",
            metadata={"internal": True, "novelty": 1.0},
        )
        reco, _ = evaluate_expression_gate(
            event=ev,
            decision=NexusDecision(action=DecisionAction.WAIT, reason="i"),
            facet_results=[
                FacetResult(
                    facet_name="policy",
                    summary="p",
                    metadata={"policy_status": "allowed"},
                ),
                FacetResult(
                    facet_name="behavior",
                    summary="b",
                    metadata={"confidence": 1.0},
                ),
            ],
            signal_map={
                "system_pressure": 1.0,
                "context_overloaded": False,
            },
            latent_pressure_total=1.0,
            lpb_ignition_recommended=True,
            interrupt_candidates=[],
            allowed_interrupt_candidate={
                "id": "ix",
                "interrupt_type": "behavior_interrupt",
                "priority": 1.0,
                "novelty": 1.0,
            },
            suppression_decisions=[],
            budget=ExpressionBudgetState(None, 0, None, 0, {}, []),
            cycle_wall_time=ev.timestamp,
            cycle_seq=2,
        )
        self.assertFalse(reco.allowed_user_facing)
        self.assertGreaterEqual(expression_mode_rank(reco.mode), expression_mode_rank(ExpressionMode.status_only))
        self.assertIn("E_internal_default", reco.suppression_rules_triggered)

    def test_budget_blocks_second_user_facing_in_window(self) -> None:
        ev = Event(event_type=EventType.USER_MESSAGE, content="hello", metadata={})
        b1 = ExpressionBudgetState(
            None,
            1,
            None,
            0,
            {},
            [],
            window_seconds=600,
            last_user_facing_at=ev.timestamp.isoformat(),
        )
        reco, budget = evaluate_expression_gate(
            event=ev,
            decision=NexusDecision(action=DecisionAction.RECORD, reason=""),
            facet_results=[
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.92}),
                FacetResult(
                    facet_name="policy",
                    summary="p",
                    metadata={"policy_status": "approval_required"},
                ),
            ],
            signal_map={
                "system_pressure": 0.9,
                "context_overloaded": False,
            },
            latent_pressure_total=0.7,
            lpb_ignition_recommended=True,
            interrupt_candidates=[
                {"id": "c1", "interrupt_type": "latent_pressure_ignition", "priority": 0.9},
            ],
            allowed_interrupt_candidate={
                "id": "c1",
                "interrupt_type": "latent_pressure_ignition",
                "priority": 0.9,
            },
            suppression_decisions=[],
            budget=b1,
            cycle_wall_time=ev.timestamp,
            cycle_seq=3,
        )
        self.assertTrue(reco.budget_applied)
        self.assertFalse(reco.allowed_user_facing)

    def test_candidate_cooldown_blocks_repeat(self) -> None:
        ev = Event(event_type=EventType.USER_MESSAGE, content="hello", metadata={})
        b0 = ExpressionBudgetState(
            last_expression_at=None,
            expression_count_window=0,
            last_expression_hash=None,
            ignored_expression_count=0,
            cooldowns={
                "c2": {"last_user_facing_at": ev.timestamp.isoformat()},
            },
            history=[],
            window_seconds=600,
            last_user_facing_at=None,
        )
        reco, _ = evaluate_expression_gate(
            event=ev,
            decision=NexusDecision(action=DecisionAction.ASK, reason="policy"),
            facet_results=[
                FacetResult(
                    facet_name="policy",
                    summary="p",
                    metadata={"policy_status": "approval_required"},
                ),
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.8}),
            ],
            signal_map={"system_pressure": 0.6, "context_overloaded": False},
            latent_pressure_total=0.4,
            lpb_ignition_recommended=False,
            interrupt_candidates=[],
            allowed_interrupt_candidate={
                "id": "c2",
                "interrupt_type": "approval_required",
                "priority": 0.7,
            },
            suppression_decisions=[],
            budget=b0,
            cycle_wall_time=ev.timestamp,
            cycle_seq=1,
        )
        self.assertTrue(reco.cooldown_applied)

    def test_metadata_suppress_expression_forces_hard_suppress(self) -> None:
        reco, _ = _run_eg(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="x",
                metadata={"suppress_expression": True},
            ),
            facet_results=[
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.99}),
            ],
            signal_map={
                "system_pressure": 0.99,
                "context_overloaded": False,
            },
            latent_pressure_total=0.99,
            lpb_ignition=False,
            allowed_interrupt_candidate={
                "id": "win",
                "interrupt_type": "behavior_interrupt",
                "priority": 0.73,
            },
        )
        self.assertTrue(reco.suppressed)

    def test_allow_expression_on_internal_allows_when_score_qualifies(self) -> None:
        ev = Event(
            event_type=EventType.INTERNAL,
            content="tick",
            metadata={"allow_expression": True, "novelty": 1.0},
        )
        reco, _ = evaluate_expression_gate(
            event=ev,
            decision=NexusDecision(action=DecisionAction.RECORD, reason=""),
            facet_results=[
                FacetResult(
                    facet_name="behavior", summary="b", metadata={"confidence": 1.0},
                ),
            ],
            signal_map={"system_pressure": 1.0, "context_overloaded": False},
            latent_pressure_total=1.0,
            lpb_ignition_recommended=True,
            interrupt_candidates=[
                {
                    "id": "ig",
                    "interrupt_type": "latent_pressure_ignition",
                    "priority": 1.0,
                    "novelty": 1.0,
                }
            ],
            allowed_interrupt_candidate={
                "id": "ig",
                "interrupt_type": "latent_pressure_ignition",
                "priority": 1.0,
                "novelty": 1.0,
            },
            suppression_decisions=[],
            budget=ExpressionBudgetState(None, 0, None, 0, {}, []),
            cycle_wall_time=ev.timestamp,
            cycle_seq=1,
        )
        self.assertTrue(reco.allowed_user_facing)

    def test_context_overload_suppresses_non_critical_user_facing(self) -> None:
        reco, _ = _run_eg(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="overload",
                metadata={"novelty": 1.0},
            ),
            facet_results=[
                FacetResult(
                    facet_name="verifier",
                    summary="v",
                    metadata={
                        "escalation_recommended": True,
                        "verification_status": "passed",
                        "results": [],
                    },
                ),
                FacetResult(
                    facet_name="behavior",
                    summary="b",
                    metadata={"confidence": 1.0, "ambiguity_score": 0.0},
                ),
            ],
            signal_map={
                "system_pressure": 1.0,
                "context_overloaded": True,
            },
            latent_pressure_total=1.0,
            lpb_ignition=False,
            allowed_interrupt_candidate={
                "id": "w",
                "interrupt_type": "behavior_interrupt",
                "priority": 1.0,
                "novelty": 1.0,
            },
        )
        self.assertFalse(reco.allowed_user_facing)
        self.assertIn("E_context_overload_non_critical", reco.suppression_rules_triggered)

    def test_verifier_critical_can_bypass_context_overload_penalty(self) -> None:
        reco, _ = _run_eg(
            facet_results=[
                FacetResult(
                    facet_name="verifier",
                    summary="v",
                    metadata={
                        "escalation_recommended": True,
                        "verification_status": "failed",
                        "results": [{"severity": "critical"}],
                    },
                ),
                FacetResult(facet_name="behavior", summary="b", metadata={"confidence": 0.7}),
            ],
            signal_map={
                "system_pressure": 0.4,
                "context_overloaded": True,
            },
            latent_pressure_total=0.2,
        )
        md = reco.metadata if isinstance(reco.metadata, dict) else {}
        self.assertTrue(md.get("verifier_critical_hint"))
        comp = md.get("score_components") or {}
        self.assertLessEqual(float(comp.get("context_overload_penalty", 0.2)), 0.01)

    def test_never_calls_llm(self) -> None:
        import fullerene.expression.gate as gate_mod

        src = Path(gate_mod.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("openai", src)
        self.assertNotIn("ollama", src)

    def test_nexus_persists_expression_gate(self) -> None:
        root = Path(__file__).resolve().parents[1] / "state" / ".test-expression-gate"
        if root.exists():
            for p in root.glob("*"):
                p.unlink(missing_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        from fullerene.state import FileStateStore

        store = FileStateStore(root)
        runtime = NexusRuntime(facets=[EchoFacet(), VerifierFacet(state_dir=root)], store=store)
        runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="persist expression")
        )
        st = runtime.state.facet_state.get("nexus") or {}
        self.assertIn("expression_gate", st)
        self.assertIsInstance(st["expression_gate"]["last_recommendation"], dict)
        self.assertIsInstance(st["expression_gate"]["budget_state"], dict)

    def test_cli_json_contains_expression_when_json(self) -> None:
        root = Path(__file__).resolve().parents[1] / "state" / ".test-cli-expression"
        root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "fullerene",
            "--json",
            "--verify",
            "--content",
            "cli expression probe",
            "--state-dir",
            str(root),
        ]
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, stderr + stdout)
        payload = json.loads(stdout.strip())
        md = payload.get("metadata") or {}
        self.assertIn("expression_recommendation", md)

    def test_verifier_validator_accepts_nominal_payload(self) -> None:
        rows = validate_expression_gate_v0(
            {
                "mode": "status_only",
                "expression_score": 0.5,
                "max_words": 10,
                "suppressed": False,
                "allowed_user_facing": False,
                "suggested_intent": "status_update",
                "payload": {"x": 1},
            },
        )
        statuses = [r.get("status") for r in rows]
        self.assertIn("passed", statuses)

    def test_verifier_validator_rejects_prose_like_payload(self) -> None:
        rows = validate_expression_gate_v0(
            {
                "mode": "short_utterance",
                "expression_score": 0.92,
                "max_words": 40,
                "suppressed": False,
                "allowed_user_facing": True,
                "suggested_intent": "surface_warning",
                "payload": {"text": "nope"},
            },
        )
        self.assertTrue(
            any(
                isinstance(r, dict) and str(r.get("code")) == "candidate_prose_keys"
                for r in rows
            ),
        )


if __name__ == "__main__":
    unittest.main()
