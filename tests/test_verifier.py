from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.facets import (
    BehaviorFacet,
    GoalsFacet,
    MemoryFacet,
    PolicyFacet,
    VerifierFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRuntime,
    NexusState,
)
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicySource,
    PolicyStatus,
    PolicyTargetType,
    SQLitePolicyStore,
)
from fullerene.state import FileStateStore, InMemoryStateStore
from fullerene.verifier import (
    ActRequiresApprovalCheck,
    ArtifactSchemaCheck,
    DecisionShapeCheck,
    FacetResultShapeCheck,
    PolicyComplianceCheck,
    VerificationContext,
    VerificationResult,
    VerificationSeverity,
    VerificationStatus,
)
from fullerene.verifier import artifacts as verifier_artifacts
from fullerene.world_model import Belief, SQLiteWorldModelStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-verifier-{uuid4().hex}"


def make_context(
    *,
    event: Event | None = None,
    facet_results: list[FacetResult] | None = None,
    decision: NexusDecision | None = None,
    state: NexusState | None = None,
    state_dir: Path | None = None,
) -> VerificationContext:
    return VerificationContext(
        event=event or Event(event_type=EventType.USER_MESSAGE, content="hello"),
        state=state or NexusState(),
        facet_results=facet_results or [],
        decision=decision,
        state_dir=state_dir,
    )


def make_policy_result(
    *,
    status: PolicyStatus,
    target_type: PolicyTargetType,
    effective_rule_type: PolicyRuleType | None = None,
    built_in: bool = False,
) -> FacetResult:
    effective_policy = None
    if effective_rule_type is not None:
        effective_policy = {
            "id": f"{effective_rule_type.value}-{target_type.value}",
            "name": f"{effective_rule_type.value}-{target_type.value}",
            "rule_type": effective_rule_type.value,
            "target_type": target_type.value,
            "target": "*",
            "priority": 1.0,
            "enabled": True,
            "source": PolicySource.USER.value,
            "built_in": built_in,
        }

    matched_policies = [effective_policy] if effective_policy is not None else []
    return FacetResult(
        facet_name="policy",
        summary="Policy evaluation result.",
        metadata={
            "policy_status": status.value,
            "matched_policies": matched_policies,
            "effective_policy": effective_policy,
            "target_type": target_type.value,
            "target": "*" if target_type != PolicyTargetType.INTERNAL_STATE else "state-dir",
            "is_internal_state_action": target_type == PolicyTargetType.INTERNAL_STATE,
            "within_state_dir": target_type == PolicyTargetType.INTERNAL_STATE,
        },
    )


class UnsafeActFacet:
    name = "unsafe_actor"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Proposed an unsafe ACT decision.",
            proposed_decision=DecisionAction.ACT,
        )


class RecordFacet:
    name = "recorder"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Proposed a RECORD decision.",
            proposed_decision=DecisionAction.RECORD,
        )


class AskFacet:
    name = "asker"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Proposed an ASK decision.",
            proposed_decision=DecisionAction.ASK,
        )


class ForceVerifierFacet:
    name = "verifier"

    def __init__(self, proposed_decision: DecisionAction | None) -> None:
        self.proposed_decision = proposed_decision

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Post-decision verifier placeholder.",
            metadata={"post_processing_only": True},
        )

    def verify(
        self,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
        decision: NexusDecision | None,
    ) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Forced verifier override proposal.",
            proposed_decision=self.proposed_decision,
            metadata={
                "verification_status": "failed",
                "failed_checks": ["forced_override"],
                "warnings": [],
                "results": [],
                "reasons": ["Forced verifier proposal for runtime hardening tests."],
            },
        )


class VerifierModelAndCheckTests(unittest.TestCase):
    def test_verification_result_round_trips_through_dict(self) -> None:
        result = VerificationResult(
            check_name="decision_shape",
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Decision is valid.",
            metadata={"action": "record"},
        )

        round_tripped = VerificationResult.from_dict(result.to_dict())

        self.assertEqual(round_tripped, result)

    def test_decision_shape_check_passes_valid_decision(self) -> None:
        result = DecisionShapeCheck().run(
            make_context(
                decision=NexusDecision(
                    action=DecisionAction.RECORD,
                    reason="Selected RECORD from facet proposals: behavior.",
                    source_facets=["behavior"],
                )
            )
        )

        self.assertEqual(result.status, VerificationStatus.PASSED)
        self.assertEqual(result.severity, VerificationSeverity.INFO)

    def test_decision_shape_check_fails_missing_or_invalid_decision(self) -> None:
        missing = DecisionShapeCheck().run(make_context(decision=None))
        invalid = DecisionShapeCheck().run(
            make_context(
                decision=NexusDecision(
                    action="explode",  # type: ignore[arg-type]
                    reason="bad decision",
                )
            )
        )

        self.assertEqual(missing.status, VerificationStatus.FAILED)
        self.assertEqual(missing.severity, VerificationSeverity.CRITICAL)
        self.assertEqual(invalid.status, VerificationStatus.FAILED)
        self.assertEqual(invalid.severity, VerificationSeverity.CRITICAL)

    def test_facet_result_shape_check_catches_malformed_result(self) -> None:
        malformed = FacetResult(
            facet_name="",
            summary="Malformed result.",
            proposed_decision="boom",  # type: ignore[arg-type]
            metadata="bad-metadata",  # type: ignore[arg-type]
        )

        result = FacetResultShapeCheck().run(
            make_context(
                facet_results=[malformed],
                decision=NexusDecision(
                    action=DecisionAction.RECORD,
                    reason="Fallback record.",
                ),
            )
        )

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("issues", result.metadata)

    def test_policy_compliance_check_fails_when_policy_denies_act(self) -> None:
        result = PolicyComplianceCheck().run(
            make_context(
                facet_results=[
                    make_policy_result(
                        status=PolicyStatus.DENIED,
                        target_type=PolicyTargetType.SHELL,
                    )
                ],
                decision=NexusDecision(
                    action=DecisionAction.ACT,
                    reason="Selected ACT from facet proposals: actor.",
                ),
            )
        )

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(result.severity, VerificationSeverity.CRITICAL)

    def test_policy_compliance_check_fails_when_approval_is_required_but_final_is_act(
        self,
    ) -> None:
        result = PolicyComplianceCheck().run(
            make_context(
                facet_results=[
                    make_policy_result(
                        status=PolicyStatus.APPROVAL_REQUIRED,
                        target_type=PolicyTargetType.SHELL,
                    )
                ],
                decision=NexusDecision(
                    action=DecisionAction.ACT,
                    reason="Selected ACT from facet proposals: actor.",
                ),
            )
        )

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(result.severity, VerificationSeverity.CRITICAL)

    def test_act_requires_approval_check_fails_unsafe_act(self) -> None:
        result = ActRequiresApprovalCheck().run(
            make_context(
                event=Event(
                    event_type=EventType.USER_MESSAGE,
                    content="run a shell command",
                    metadata={"explicit_action": True, "target_type": "shell"},
                ),
                decision=NexusDecision(
                    action=DecisionAction.ACT,
                    reason="Selected ACT from facet proposals: actor.",
                ),
            )
        )

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(result.severity, VerificationSeverity.CRITICAL)

    def test_act_requires_approval_check_allows_internal_state_act(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        result = ActRequiresApprovalCheck().run(
            make_context(
                event=Event(
                    event_type=EventType.USER_MESSAGE,
                    content="update the runtime state",
                    metadata={
                        "explicit_action": True,
                        "target_type": "internal_state",
                        "target": "state-dir",
                    },
                ),
                decision=NexusDecision(
                    action=DecisionAction.ACT,
                    reason="Selected ACT from facet proposals: actor.",
                ),
                state_dir=root,
            )
        )

        self.assertEqual(result.status, VerificationStatus.PASSED)
        self.assertEqual(result.severity, VerificationSeverity.INFO)


class VerifierRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_behavior_goals_world_policy_and_verifier(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(root / "world.sqlite3")
        policy_store = SQLitePolicyStore(root / "policy.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Keep the internal state healthy",
                priority=0.9,
                tags=["state"],
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="The configured state directory is safe for internal CRUD",
                confidence=0.9,
                tags=["state"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store, active_limit=5, relevant_limit=3),
                WorldModelFacet(world_store, active_limit=5, relevant_limit=3),
                BehaviorFacet(),
                PolicyFacet(policy_store, state_dir=root),
                VerifierFacet(state_dir=root),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="update the runtime state",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "internal_state",
                    "target": "state-dir",
                    "tags": ["state"],
                },
            )
        )

        verifier_result = record.facet_results[-1]

        self.assertEqual(record.decision.action, DecisionAction.ACT)
        self.assertEqual(verifier_result.facet_name, "verifier")
        self.assertEqual(verifier_result.metadata["verification_status"], "passed")

    def test_unsafe_act_is_downgraded_by_verifier(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[UnsafeActFacet(), VerifierFacet(state_dir=root)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            )
        )
        verifier_result = record.facet_results[-1]

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(verifier_result.metadata["verification_status"], "failed")
        self.assertIn("act_requires_approval", verifier_result.metadata["failed_checks"])

    def test_verifier_can_downgrade_act_to_record(self) -> None:
        runtime = NexusRuntime(
            facets=[UnsafeActFacet(), ForceVerifierFacet(DecisionAction.RECORD)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="forced override")
        )
        verifier_result = record.facet_results[-1]

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertTrue(verifier_result.metadata["override_applied"])
        self.assertEqual(
            verifier_result.metadata["override_reason"],
            "risk_reducing_downgrade",
        )

    def test_verifier_cannot_upgrade_record_to_act(self) -> None:
        runtime = NexusRuntime(
            facets=[RecordFacet(), ForceVerifierFacet(DecisionAction.ACT)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="forced upgrade")
        )
        verifier_result = record.facet_results[-1]

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertFalse(verifier_result.metadata["override_applied"])
        self.assertEqual(
            verifier_result.metadata["override_reason"],
            "ignored_higher_priority_verifier_proposal",
        )

    def test_verifier_cannot_upgrade_ask_to_act(self) -> None:
        runtime = NexusRuntime(
            facets=[AskFacet(), ForceVerifierFacet(DecisionAction.ACT)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="forced upgrade")
        )
        verifier_result = record.facet_results[-1]

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertFalse(verifier_result.metadata["override_applied"])
        self.assertEqual(
            verifier_result.metadata["override_reason"],
            "ignored_higher_priority_verifier_proposal",
        )

    def test_approval_required_policy_prevents_final_act(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(SQLitePolicyStore(root / "policy.sqlite3"), state_dir=root),
                VerifierFacet(state_dir=root),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run the shell command",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "shell",
                },
            )
        )

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(
            record.facet_results[-1].metadata["verification_status"],
            "passed",
        )

    def test_deny_policy_prevents_final_act(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        policy_store = SQLitePolicyStore(root / "policy.sqlite3")
        policy_store.add_policy(
            PolicyRule(
                id="deny-shell",
                name="Deny shell",
                description="Deny shell execution.",
                rule_type=PolicyRuleType.DENY,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=10.0,
            )
        )
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(policy_store, state_dir=root),
                VerifierFacet(state_dir=root),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run the shell command",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "shell",
                },
            )
        )

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertEqual(
            record.facet_results[-1].metadata["verification_status"],
            "passed",
        )

    def test_normal_record_and_ask_decisions_pass_verification(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[BehaviorFacet(), VerifierFacet(state_dir=root)],
            store=InMemoryStateStore(),
        )

        record_result = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="record this note")
        )
        ask_result = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="what should I do next?")
        )

        self.assertEqual(record_result.decision.action, DecisionAction.RECORD)
        self.assertEqual(
            record_result.facet_results[-1].metadata["verification_status"],
            "passed",
        )
        self.assertEqual(ask_result.decision.action, DecisionAction.ASK)
        self.assertEqual(
            ask_result.facet_results[-1].metadata["verification_status"],
            "passed",
        )

    def test_verifier_metadata_is_persisted_in_runtime_log(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[UnsafeActFacet(), VerifierFacet(state_dir=root)],
            store=FileStateStore(root),
        )

        runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            )
        )

        payload = json.loads((root / "runtime-log.jsonl").read_text(encoding="utf-8").strip())
        verifier_result = payload["facet_results"][-1]

        self.assertEqual(verifier_result["facet_name"], "verifier")
        self.assertEqual(verifier_result["metadata"]["verification_status"], "failed")
        self.assertIn("failed_checks", verifier_result["metadata"])


class CLIVerifierIntegrationTests(unittest.TestCase):
    def test_cli_verify_flag_runs_without_error(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--verify",
                    "--content",
                    "record this note",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["decision"]["action"], "record")
        self.assertEqual(payload["facet_results"][-1]["facet_name"], "verifier")

    def test_cli_behavior_policy_verify_blocks_shell_action(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--behavior",
                    "--policy",
                    "--verify",
                    "--content",
                    "run a shell command",
                    "--metadata",
                    '{"explicit_action": true, "target_type": "shell"}',
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        verifier_result = payload["facet_results"][-1]

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["decision"]["action"], "ask")
        self.assertEqual(verifier_result["facet_name"], "verifier")
        self.assertIn(verifier_result["metadata"]["verification_status"], {"passed", "warning"})


def _behavior_trace_complete() -> dict:
    scores = {"wait": 0.2, "record": 0.2, "ask": 0.2, "act": 0.2}
    return {
        "event": {
            "id": "e1",
            "type": "user_message",
            "content_summary": "hello sample",
        },
        "pressure_score": 0.4,
        "latent_pressure": 0.0,
        "memory_relevance_score": 0.5,
        "goal_relevance_score": 0.3,
        "world_model_belief_confidence": 0.5,
        "contradiction_flag": False,
        "policy_result": "allowed",
        "context_load_ratio": 0.2,
        "raw_candidate_scores": dict(scores),
        "adjusted_candidate_scores": dict(scores),
        "final_decision": "act",
        "confidence": 0.7,
        "reasons": ["ok"],
        "interrupt_recommended": False,
        "timestamp": "2026-05-06T12:00:00+00:00",
    }


class VerifierV1ArtifactTests(unittest.TestCase):
    def test_behavior_v2_trace_complete_passes(self) -> None:
        rows, reco = verifier_artifacts.validate_behavior_decision_trace_v2(
            _behavior_trace_complete(), decision_is_act=False
        )
        self.assertIsNone(reco)
        codes = [r["code"] for r in rows if r["status"] == "passed"]
        self.assertIn("ok", codes)

    def test_behavior_v2_missing_core_fields_warns(self) -> None:
        bad = dict(_behavior_trace_complete())
        del bad["confidence"]
        del bad["timestamp"]
        rows, reco = verifier_artifacts.validate_behavior_decision_trace_v2(
            bad, decision_is_act=False
        )
        warnish = [
            r
            for r in rows
            if r["code"] in ("missing_optional_numeric", "missing_timestamp")
        ]
        self.assertTrue(any(r.get("retry_recommended") for r in warnish))
        self.assertIsNone(reco)

    def test_malformed_act_behavior_trace_downgrades_to_ask(self) -> None:
        bad_event = dict(_behavior_trace_complete())
        bad_event["event"] = "not-a-dict"  # type: ignore[assignment]
        state = NexusState()
        state.facet_state["nexus"] = {
            "verifier_cycle_context": {
                "facet_order": ["behavior"],
                "signal_map": {"system_pressure": 0.5, "pressure_components": {}},
                "pressure_components": {},
                "learning_events": [],
                "internal_events_queued": [],
                "internal_events_processed": [],
                "final_decision": "act",
            }
        }
        ctx = make_context(
            facet_results=[
                FacetResult(
                    facet_name="behavior",
                    summary="stub",
                    metadata={"decision_trace": bad_event},
                )
            ],
            decision=NexusDecision(action=DecisionAction.ACT, reason="test"),
            state=state,
        )
        result = ArtifactSchemaCheck().run(ctx)
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(result.metadata["recommended_action"], "ask")

    def test_pressure_components_sum_checked(self) -> None:
        sm = {
            "system_pressure": 0.05,
            "pressure_components": {
                "event_pressure": 1.0,
                "attention_pressure": 1.0,
                "latent_pressure": 0.0,
                "contradiction_pressure": 0.0,
                "context_overload_pressure": 0.0,
                "interrupt_pressure": 0.0,
            },
        }
        rows = verifier_artifacts.validate_cycle_signal_map_pressure(sm)
        self.assertTrue(
            any(r["code"] == "system_pressure_mismatch" for r in rows),
            rows,
        )

    def test_cycle_trace_internal_events_processed_cap(self) -> None:
        trace = {
            "facet_order": ["a"],
            "signal_map": {"x": 1},
            "pressure_components": {},
            "final_decision": "wait",
            "learning_events": [],
            "internal_events_queued": [],
            "internal_events_processed": [{"event_id": "1"}, {"event_id": "2"}],
        }
        rows, reco = verifier_artifacts.validate_cycle_trace_structure(trace)
        self.assertEqual(reco, DecisionAction.RECORD)
        self.assertTrue(
            any(
                r.get("severity") == "critical"
                and r["code"] == "internal_events_over_limit"
                for r in rows
            ),
            rows,
        )

    def test_planner_high_risk_missing_approval_escalates(self) -> None:
        state = NexusState()
        state.facet_state["nexus"] = {"verifier_cycle_context": {}}
        plan = {
            "id": "p1",
            "status": "proposed",
            "confidence": 0.9,
            "steps": [
                {
                    "id": "s1",
                    "description": "danger",
                    "status": "proposed",
                    "risk_level": "high",
                    "requires_approval": False,
                    "policy_status": None,
                    "target_type": "general",
                    "metadata": {"action_type": "noop"},
                }
            ],
        }
        ctx = make_context(
            facet_results=[
                FacetResult(
                    facet_name="planner",
                    summary="stub",
                    metadata={"plan": plan},
                ),
                FacetResult(
                    facet_name="executor",
                    summary="exec",
                    metadata={
                        "execution_requested": True,
                        "execution_result": {
                            "overall_status": "success",
                            "halted": False,
                            "dry_run": True,
                            "records": [],
                        },
                    },
                ),
            ],
            decision=NexusDecision(action=DecisionAction.WAIT, reason="."),
            state=state,
        )
        res = ArtifactSchemaCheck().run(ctx)
        self.assertEqual(res.metadata["recommended_action"], "ask")

    def test_planner_blocked_executable_step_in_approved_plan(self) -> None:
        state = NexusState()
        state.facet_state["nexus"] = {"verifier_cycle_context": {}}
        plan = {
            "id": "p1",
            "status": "approved",
            "confidence": 0.5,
            "steps": [
                {
                    "id": "s1",
                    "description": "x",
                    "status": "blocked",
                    "risk_level": "medium",
                    "requires_approval": False,
                    "policy_status": "denied",
                    "target_type": "general",
                    "metadata": {"action_type": "update_goal"},
                }
            ],
        }
        ctx = make_context(
            facet_results=[
                FacetResult(
                    facet_name="planner",
                    summary="stub",
                    metadata={"plan": plan},
                ),
                FacetResult(
                    facet_name="executor",
                    summary="e",
                    metadata={
                        "execution_requested": True,
                        "execution_result": {
                            "overall_status": "failed",
                            "halted": False,
                            "dry_run": True,
                            "records": [
                                {
                                    "id": "x",
                                    "status": "failed",
                                    "dry_run": True,
                                    "metadata": {
                                        "reason": "blocked_by_policy",
                                    },
                                }
                            ],
                        },
                    },
                ),
            ],
            decision=NexusDecision(action=DecisionAction.WAIT, reason="."),
            state=state,
        )
        res = ArtifactSchemaCheck().run(ctx)
        self.assertEqual(res.metadata["recommended_action"], "ask")
        codes = [
            str(r["code"])
            for r in res.metadata.get("artifact_checks", [])
            if isinstance(r, dict)
        ]
        self.assertIn("blocked_step_in_approved_plan", codes)

    def test_executor_external_refusal_with_reason_is_valid(self) -> None:
        state = NexusState()
        state.facet_state["nexus"] = {"verifier_cycle_context": {}}
        er = {
            "overall_status": "failed",
            "halted": False,
            "dry_run": True,
            "records": [
                {
                    "id": "r1",
                    "status": "failed",
                    "dry_run": True,
                    "metadata": {
                        "reason": "unsupported_target_type",
                        "target_type": "shell",
                    },
                }
            ],
        }
        _, reco = verifier_artifacts.run_all_artifact_validators(
            facet_results=[
                FacetResult(
                    facet_name="executor",
                    summary="stub",
                    metadata={
                        "execution_requested": True,
                        "execution_result": er,
                    },
                ),
            ],
            decision_action=None,
            nexus_ctx={},
        )
        self.assertIsNone(reco)

    def test_executor_failed_record_missing_reason_recommends_escalation(self) -> None:
        state = NexusState()
        state.facet_state["nexus"] = {"verifier_cycle_context": {}}
        er = {
            "overall_status": "failed",
            "halted": True,
            "dry_run": True,
            "records": [
                {
                    "status": "failed",
                    "dry_run": True,
                    "metadata": {
                        "action_type": "noop",
                        "target_type": "shell",
                        "requested_action_type": "noop",
                    },
                }
            ],
        }
        rows, reco = verifier_artifacts.run_all_artifact_validators(
            facet_results=[
                FacetResult(
                    facet_name="executor",
                    summary="stub",
                    metadata={
                        "execution_requested": True,
                        "execution_result": er,
                    },
                ),
            ],
            decision_action=None,
            nexus_ctx={},
        )
        self.assertEqual(reco, DecisionAction.ASK)
        self.assertTrue(
            any(
                isinstance(r, dict) and r.get("code") == "missing_failure_reason_code"
                for r in rows
            ),
            rows,
        )

    def test_learning_applied_missing_provenance_fails(self) -> None:
        meta = {
            "learning_version": "v1",
            "consumed_learning_events": [],
            "adjustment_records": [],
            "proposed_adjustments": [],
                    "applied_adjustments": [
                {
                    "target": "memory_salience",
                    "target_id": None,
                    "target_facet": "memory",
                    "field": "salience",
                    "status": "applied",
                    "source_signal_id": "",
                    "old_value": 0.5,
                    "new_value": 0.55,
                    "delta": 0.05,
                    "reasons": [],
                }
            ],
            "skipped_adjustments": [],
            "cross_facet_routes": [],
        }
        _, reco = verifier_artifacts.validate_learning_v1_metadata(meta)
        self.assertEqual(reco, DecisionAction.RECORD)

    def test_learning_permission_mutation_is_critical(self) -> None:
        meta = {
            "learning_version": "v1",
            "consumed_learning_events": [],
            "adjustment_records": [],
            "proposed_adjustments": [],
            "applied_adjustments": [
                {
                    "target": "goal_priority",
                    "target_facet": "executor",
                    "field": "permission_scope",
                    "status": "applied",
                    "source_signal_id": "sig",
                    "old_value": None,
                    "new_value": None,
                    "delta": 0.0,
                    "reasons": [],
                }
            ],
            "skipped_adjustments": [],
            "cross_facet_routes": [],
        }
        rows, reco = verifier_artifacts.validate_learning_v1_metadata(meta)
        self.assertEqual(reco, DecisionAction.RECORD)
        self.assertTrue(
            any(
                isinstance(r, dict) and r["code"] == "forbidden_permission_learning_apply"
                for r in rows
            )
        )

    def test_context_load_malformed_ratio_warns(self) -> None:
        rows = verifier_artifacts.validate_context_load_metadata(
            {"item_count": 2, "max_items": 4, "load_ratio": 9.0, "overloaded": False}
        )
        self.assertTrue(
            any(
                isinstance(r, dict)
                and r.get("path") == "context_load.load_ratio"
                and r["code"] == "load_ratio_malformed"
                for r in rows
            )
        )

    def test_facet_result_includes_verifier_v1_summary_fields(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(SQLitePolicyStore(root / "policy.sqlite3"), state_dir=root),
                VerifierFacet(state_dir=root),
            ],
            store=InMemoryStateStore(),
        )
        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="remember this phrase")
        )
        vm = record.facet_results[-1].metadata
        self.assertEqual(vm["verifier_version"], "v1")
        self.assertIn("artifact_checks", vm)
        self.assertIn("schema_checks", vm)
        self.assertIn("validated_artifact_kinds", vm)

    def test_cli_verify_json_contains_verifier_v1(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--json",
                    "--behavior",
                    "--verify",
                    "--content",
                    "remember to pack lunch",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        vmeta = payload["facet_results"][-1]["metadata"]
        self.assertEqual(vmeta.get("verifier_version"), "v1")


if __name__ == "__main__":
    unittest.main()
