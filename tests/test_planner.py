from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.context import ContextWindow
from fullerene.facets import (
    BehaviorFacet,
    ContextFacet,
    GoalsFacet,
    MemoryFacet,
    PlannerFacet,
    PolicyFacet,
    VerifierFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, FacetResult, NexusRuntime, NexusState
from fullerene.planner import (
    DeterministicPlanBuilder,
    Plan,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    RiskLevel,
)
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicyTargetType,
    SQLitePolicyStore,
)
from fullerene.state import FileStateStore, InMemoryStateStore
from fullerene.verifier import PlanSafetyCheck, VerificationStatus
from fullerene.world_model import Belief, SQLiteWorldModelStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-planner-{uuid4().hex}"


class PlannerModelTests(unittest.TestCase):
    def test_plan_step_round_trips_through_dict(self) -> None:
        step = PlanStep(
            id="step-1",
            description="Review goal context.",
            order=2,
            target_type="general",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            status=PlanStepStatus.PROPOSED,
            policy_status="allowed",
            metadata={"step_kind": "review_goal_context"},
        )

        round_tripped = PlanStep.from_dict(step.to_dict())

        self.assertEqual(round_tripped, step)

    def test_plan_round_trips_through_dict(self) -> None:
        plan = Plan(
            id="plan-1",
            source_event_id="event-1",
            goal_id="goal-1",
            title="Proposed plan for goal: stabilize state",
            steps=[
                PlanStep(
                    id="step-1",
                    description="Review goal context.",
                    order=1,
                )
            ],
            confidence=0.82,
            pressure=0.65,
            status=PlanStatus.PROPOSED,
            reasons=["explicit_plan_request"],
            metadata={"trigger_reason": "explicit_request"},
        )

        round_tripped = Plan.from_dict(plan.to_dict())

        self.assertEqual(round_tripped, plan)


class PlannerFacetExportTests(unittest.TestCase):
    def test_planner_facet_is_exported_from_fullerene_facets(self) -> None:
        from fullerene.facets import PlannerFacet as exported_planner_facet

        self.assertIs(exported_planner_facet, PlannerFacet)


class DeterministicPlanBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_no_trigger_returns_no_plan(self) -> None:
        builder = DeterministicPlanBuilder()

        plan = builder.build(
            Event(event_type=EventType.USER_MESSAGE, content="record this note"),
            NexusState(),
        )

        self.assertIsNone(plan)

    def test_explicit_request_creates_plan(self) -> None:
        builder = DeterministicPlanBuilder()

        plan = builder.build(
            Event(event_type=EventType.USER_MESSAGE, content="make a plan for this"),
            NexusState(),
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.status, PlanStatus.PROPOSED)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].description, "Clarify the objective.")
        self.assertEqual(plan.steps[-1].description, "Propose the next safe action.")

    def test_high_pressure_creates_fewer_direct_steps(self) -> None:
        builder = DeterministicPlanBuilder()

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"pressure": 0.8},
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.pressure, 0.8)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[-1].description, "Propose the next safe action.")

    def test_low_pressure_creates_more_exploratory_steps(self) -> None:
        builder = DeterministicPlanBuilder()

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"pressure": 0.2},
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.pressure, 0.2)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[1].description, "Identify constraints.")

    def test_confidence_increases_with_explicit_request(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-release",
                description="Prepare the release plan",
                priority=0.8,
                tags=["release"],
            )
        )
        builder = DeterministicPlanBuilder(goal_store=goal_store)

        goal_trigger_plan = builder.build(
            Event(event_type=EventType.USER_MESSAGE, content="what next for release"),
            NexusState(),
        )
        explicit_request_plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for the release",
            ),
            NexusState(),
        )

        assert goal_trigger_plan is not None
        assert explicit_request_plan is not None
        self.assertGreater(explicit_request_plan.confidence, goal_trigger_plan.confidence)

    def test_confidence_increases_with_matched_high_priority_goal(self) -> None:
        low_store = SQLiteGoalStore(self.root / "goals-low.sqlite3")
        low_store.add_goal(
            Goal(
                id="goal-low",
                description="Stabilize the state system",
                priority=0.2,
                tags=["state"],
            )
        )
        high_store = SQLiteGoalStore(self.root / "goals-high.sqlite3")
        high_store.add_goal(
            Goal(
                id="goal-high",
                description="Stabilize the state system",
                priority=0.9,
                tags=["state"],
            )
        )

        low_plan = DeterministicPlanBuilder(goal_store=low_store).build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for state stabilization",
                metadata={"tags": ["state"]},
            ),
            NexusState(),
        )
        high_plan = DeterministicPlanBuilder(goal_store=high_store).build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for state stabilization",
                metadata={"tags": ["state"]},
            ),
            NexusState(),
        )

        assert low_plan is not None
        assert high_plan is not None
        self.assertGreater(high_plan.confidence, low_plan.confidence)

    def test_confidence_increases_with_high_confidence_relevant_belief(self) -> None:
        no_belief_plan = DeterministicPlanBuilder().build(
            Event(event_type=EventType.USER_MESSAGE, content="make a plan for state work"),
            NexusState(),
        )

        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        world_store.add_belief(
            Belief(
                id="belief-state",
                claim="The state system needs careful review",
                confidence=0.95,
                tags=["state"],
            )
        )
        belief_plan = DeterministicPlanBuilder(world_model_store=world_store).build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for state work",
                metadata={"tags": ["state"]},
            ),
            NexusState(),
        )

        assert no_belief_plan is not None
        assert belief_plan is not None
        self.assertGreater(belief_plan.confidence, no_belief_plan.confidence)

    def test_confidence_is_clamped_to_unit_interval(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals-clamped.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Stabilize the state system",
                priority=1.0,
                tags=["state"],
            )
        )
        world_store = SQLiteWorldModelStore(self.root / "world-clamped.sqlite3")
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="The state system is safe for internal work",
                confidence=1.0,
                tags=["state"],
            )
        )
        policy_store = SQLitePolicyStore(self.root / "policy-clamped.sqlite3")
        builder = DeterministicPlanBuilder(
            goal_store=goal_store,
            world_model_store=world_store,
            policy_store=policy_store,
            state_dir=self.root,
        )

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for state stabilization",
                metadata={
                    "tags": ["state"],
                    "target_type": "internal_state",
                    "target": "state-dir",
                },
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.confidence, 1.0)

    def test_high_risk_step_requires_approval(self) -> None:
        builder = DeterministicPlanBuilder()

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"target_type": "shell"},
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.steps[-1].risk_level, RiskLevel.HIGH)
        self.assertTrue(plan.steps[-1].requires_approval)
        self.assertEqual(plan.steps[-1].status, PlanStepStatus.REQUIRES_APPROVAL)

    def test_denied_policy_marks_step_blocked(self) -> None:
        policy_store = SQLitePolicyStore(self.root / "policy-deny.sqlite3")
        policy_store.add_policy(
            PolicyRule(
                id="deny-shell",
                name="Deny shell",
                description="Deny shell planning steps.",
                rule_type=PolicyRuleType.DENY,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=10.0,
            )
        )
        builder = DeterministicPlanBuilder(
            policy_store=policy_store,
            state_dir=self.root,
        )

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"target_type": "shell"},
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.steps[-1].status, PlanStepStatus.BLOCKED)
        self.assertEqual(plan.steps[-1].policy_status, "denied")

    def test_require_approval_policy_marks_step_requires_approval(self) -> None:
        policy_store = SQLitePolicyStore(self.root / "policy-approval.sqlite3")
        policy_store.add_policy(
            PolicyRule(
                id="internal-approval",
                name="Approve internal planning",
                description="Require approval for internal-state planning steps.",
                rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                target_type=PolicyTargetType.INTERNAL_STATE,
                target="state-dir",
                priority=10.0,
            )
        )
        builder = DeterministicPlanBuilder(
            policy_store=policy_store,
            state_dir=self.root,
        )

        plan = builder.build(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"target_type": "internal_state", "target": "state-dir"},
            ),
            NexusState(),
        )

        assert plan is not None
        self.assertEqual(plan.steps[-1].status, PlanStepStatus.REQUIRES_APPROVAL)
        self.assertTrue(plan.steps[-1].requires_approval)
        self.assertEqual(plan.steps[-1].policy_status, "approval_required")


class PlannerFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_returns_wait_when_not_triggered(self) -> None:
        result = PlannerFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="record this"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertIsNone(result.metadata["plan"])

    def test_returns_record_with_plan_metadata_when_triggered(self) -> None:
        result = PlannerFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="make a plan for this"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertIsNotNone(result.metadata["plan"])
        self.assertEqual(result.metadata["trigger_reason"], "explicit_request")

    def test_works_without_goal_world_or_policy_stores(self) -> None:
        result = PlannerFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="make a plan for this"),
            NexusState(),
        )

        self.assertEqual(result.metadata["relevant_goal_ids"], [])
        self.assertEqual(result.metadata["relevant_belief_ids"], [])

    def test_works_with_goal_world_and_policy_stores(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-state",
                description="Keep the state system healthy",
                priority=0.9,
                tags=["state"],
            )
        )
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        world_store.add_belief(
            Belief(
                id="belief-state",
                claim="The state directory is safe for internal updates",
                confidence=0.9,
                tags=["state"],
            )
        )
        policy_store = SQLitePolicyStore(self.root / "policy.sqlite3")
        facet = PlannerFacet(
            goal_store=goal_store,
            world_model_store=world_store,
            policy_store=policy_store,
            state_dir=self.root,
        )

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for state maintenance",
                metadata={
                    "tags": ["state"],
                    "target_type": "internal_state",
                    "target": "state-dir",
                },
            ),
            NexusState(),
        )

        plan = result.metadata["plan"]
        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(plan["goal_id"], "goal-state")
        self.assertIn("goal-state", result.metadata["relevant_goal_ids"])
        self.assertIn("belief-state", result.metadata["relevant_belief_ids"])
        self.assertEqual(plan["steps"][-1]["policy_status"], "allowed")


class PlannerVerifierTests(unittest.TestCase):
    def test_plan_safety_check_requires_high_risk_steps_to_request_approval(self) -> None:
        planner_result = FacetResult(
            facet_name="planner",
            summary="Planner output.",
            metadata={
                "plan": {
                    "id": "plan-1",
                    "status": "proposed",
                    "steps": [
                        {
                            "id": "step-1",
                            "risk_level": "high",
                            "requires_approval": False,
                            "status": "proposed",
                        }
                    ],
                }
            },
        )

        result = PlanSafetyCheck().run(
            type(
                "VerificationContextStub",
                (),
                {
                    "event": Event(event_type=EventType.USER_MESSAGE, content="plan"),
                    "state": NexusState(),
                    "facet_results": [planner_result],
                    "decision": None,
                    "state_dir": None,
                },
            )()
        )

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("issues", result.metadata)


class PlannerRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_behavior_goals_world_policy_verifier_context_and_planner(
        self,
    ) -> None:
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
                claim="The internal state needs careful review before changes",
                confidence=0.95,
                tags=["state"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(memory_store, window_size=2),
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store, active_limit=5, relevant_limit=3),
                WorldModelFacet(world_store, active_limit=5, relevant_limit=3),
                BehaviorFacet(),
                PolicyFacet(policy_store, state_dir=root),
                PlannerFacet(
                    goal_store=goal_store,
                    world_model_store=world_store,
                    policy_store=policy_store,
                    state_dir=root,
                ),
                VerifierFacet(state_dir=root),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what are the next steps for state maintenance?",
                metadata={
                    "request_plan": True,
                    "target_type": "shell",
                    "tags": ["state"],
                },
            )
        )

        planner_result = next(
            result for result in record.facet_results if result.facet_name == "planner"
        )
        plan = planner_result.metadata["plan"]

        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            [
                "context",
                "memory",
                "goals",
                "world_model",
                "behavior",
                "policy",
                "planner",
                "verifier",
            ],
        )
        self.assertEqual(planner_result.proposed_decision, DecisionAction.RECORD)
        self.assertNotEqual(planner_result.proposed_decision, DecisionAction.ACT)
        self.assertEqual(plan["steps"][-1]["status"], "requires_approval")
        self.assertTrue(plan["steps"][-1]["requires_approval"])
        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(record.facet_results[-1].metadata["verification_status"], "passed")


class CLIPlannerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_cli_planner_runs_without_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--content",
                    "make a plan for this",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        planner_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "planner"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(planner_result["metadata"]["trigger_reason"], "explicit_request")
        self.assertEqual(planner_result["proposed_decision"], "record")

    def test_cli_pressure_is_reflected_in_planner_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--pressure",
                    "0.8",
                    "--content",
                    "what are the steps?",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        planner_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "planner"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(planner_result["metadata"]["pressure"], 0.8)
        self.assertEqual(len(planner_result["metadata"]["plan"]["steps"]), 2)

    def test_cli_explicit_request_outputs_plan_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--goals",
                    "--world",
                    "--policy",
                    "--content",
                    "make a plan for this",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        planner_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "planner"
        )

        self.assertEqual(exit_code, 0)
        self.assertIsNotNone(planner_result["metadata"]["plan"])
        self.assertEqual(planner_result["proposed_decision"], "record")

    def test_cli_planner_never_produces_act(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--content",
                    "make a plan for this",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        planner_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "planner"
        )

        self.assertEqual(exit_code, 0)
        self.assertNotEqual(planner_result["proposed_decision"], "act")


if __name__ == "__main__":
    unittest.main()
