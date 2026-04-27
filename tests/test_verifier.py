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
    DecisionShapeCheck,
    FacetResultShapeCheck,
    PolicyComplianceCheck,
    VerificationContext,
    VerificationResult,
    VerificationSeverity,
    VerificationStatus,
)
from fullerene.world_model import Belief, SQLiteWorldModelStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-verifier-{uuid4().hex}"


def make_context(
    *,
    event: Event | None = None,
    facet_results: list[FacetResult] | None = None,
    decision: NexusDecision | None = None,
    state_dir: Path | None = None,
) -> VerificationContext:
    return VerificationContext(
        event=event or Event(event_type=EventType.USER_MESSAGE, content="hello"),
        state=NexusState(),
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


if __name__ == "__main__":
    unittest.main()
