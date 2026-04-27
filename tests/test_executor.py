from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.executor import (
    ActionType,
    ExecutionMode,
    ExecutionRecord,
    ExecutionResult,
    ExecutionStatus,
    InternalActionExecutor,
)
from fullerene.facets import ExecutorFacet, PlannerFacet
from fullerene.goals import Goal, GoalStatus, SQLiteGoalStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.planner import Plan, PlanStep, PlanStepStatus, RiskLevel
from fullerene.state import InMemoryStateStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-executor-{uuid4().hex}"


def make_plan(*steps: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        source_event_id="event-1",
        title="Executor test plan",
        steps=list(steps),
        reasons=["test-plan"],
    )


class ExecutorModelTests(unittest.TestCase):
    def test_execution_record_round_trips_through_dict(self) -> None:
        record = ExecutionRecord(
            id="exec-1",
            action_type=ActionType.NOOP,
            plan_id="plan-1",
            plan_step_id="step-1",
            status=ExecutionStatus.SUCCESS,
            dry_run=True,
            message="Dry-run validated noop action.",
            metadata={"reason": "execution_completed"},
        )

        round_tripped = ExecutionRecord.from_dict(record.to_dict())

        self.assertEqual(round_tripped, record)

    def test_execution_result_round_trips_through_dict(self) -> None:
        result = ExecutionResult(
            plan_id="plan-1",
            records=[
                ExecutionRecord(
                    id="exec-1",
                    action_type=ActionType.NOOP,
                    plan_id="plan-1",
                    plan_step_id="step-1",
                    status=ExecutionStatus.SUCCESS,
                    dry_run=True,
                    message="Dry-run validated noop action.",
                )
            ],
            overall_status=ExecutionStatus.SUCCESS,
            halted=False,
            dry_run=True,
            reasons=["execution_completed"],
            metadata={"mode": "dry_run"},
        )

        round_tripped = ExecutionResult.from_dict(result.to_dict())

        self.assertEqual(round_tripped, result)


class InternalActionExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_dry_run_goal_update_records_success_without_mutation(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Keep this active",
                status=GoalStatus.ACTIVE,
            )
        )
        executor = InternalActionExecutor(goal_store=goal_store, state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Complete the goal.",
                order=1,
                target_type="goal",
                metadata={
                    "action_type": "update_goal",
                    "goal_id": "goal-1",
                    "status": "completed",
                },
            )
        )

        result = executor.execute(plan, mode=ExecutionMode.DRY_RUN)

        self.assertEqual(result.overall_status, ExecutionStatus.SUCCESS)
        self.assertTrue(result.dry_run)
        self.assertEqual(goal_store.get_goal("goal-1").status, GoalStatus.ACTIVE)

    def test_live_noop_records_success(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Do nothing.",
                order=1,
                target_type="noop",
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan, mode=ExecutionMode.LIVE)

        self.assertEqual(result.overall_status, ExecutionStatus.SUCCESS)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.records[0].status, ExecutionStatus.SUCCESS)

    def test_requires_approval_step_is_skipped_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Approval required.",
                order=1,
                target_type="noop",
                requires_approval=True,
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.SKIPPED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["requires_approval"])
        self.assertEqual(result.records[0].metadata["reason"], "requires_approval")

    def test_blocked_step_is_skipped_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Blocked by policy.",
                order=1,
                target_type="noop",
                status=PlanStepStatus.BLOCKED,
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.SKIPPED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["blocked_by_policy"])
        self.assertEqual(result.records[0].metadata["reason"], "blocked_by_policy")

    def test_policy_denied_step_uses_blocked_by_policy_reason(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Denied by policy.",
                order=1,
                target_type="noop",
                policy_status="denied",
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.SKIPPED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["blocked_by_policy"])
        self.assertEqual(result.records[0].metadata["reason"], "blocked_by_policy")

    def test_high_risk_step_is_skipped_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Too risky.",
                order=1,
                target_type="noop",
                risk_level=RiskLevel.HIGH,
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.SKIPPED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["high_risk_not_allowed_v0"])
        self.assertEqual(result.records[0].metadata["reason"], "high_risk_not_allowed_v0")

    def test_unknown_action_type_fails_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Unknown action.",
                order=1,
                target_type="noop",
                metadata={"action_type": "invented_action"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["unsupported_action_type"])
        self.assertEqual(result.records[0].metadata["reason"], "unsupported_action_type")

    def test_unknown_target_type_fails_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Unknown target.",
                order=1,
                target_type="mystery",
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["unsupported_target_type"])
        self.assertEqual(result.records[0].metadata["reason"], "unsupported_target_type")

    def test_external_target_type_fails_and_halts(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="External shell action.",
                order=1,
                target_type="shell",
                metadata={"action_type": "noop"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["unsupported_target_type"])
        self.assertEqual(result.records[0].metadata["reason"], "unsupported_target_type")

    def test_unsupported_live_action_reason_is_distinct(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Update memory live.",
                order=1,
                target_type="memory",
                metadata={"action_type": "update_memory", "memory_id": "mem-1"},
            )
        )

        result = executor.execute(plan, mode=ExecutionMode.LIVE)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(result.reasons, ["unsupported_live_action"])
        self.assertEqual(result.records[0].metadata["reason"], "unsupported_live_action")

    def test_no_partial_execution_after_failure(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Do not complete me yet",
                status=GoalStatus.ACTIVE,
            )
        )
        executor = InternalActionExecutor(goal_store=goal_store, state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Would complete the goal.",
                order=1,
                target_type="goal",
                metadata={
                    "action_type": "update_goal",
                    "goal_id": "goal-1",
                    "status": "completed",
                },
            ),
            PlanStep(
                id="step-2",
                description="Needs approval.",
                order=2,
                target_type="noop",
                requires_approval=True,
                metadata={"action_type": "noop"},
            ),
        )

        result = executor.execute(plan, mode=ExecutionMode.LIVE)

        self.assertEqual(result.overall_status, ExecutionStatus.SKIPPED)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].plan_step_id, "step-2")
        self.assertEqual(goal_store.get_goal("goal-1").status, GoalStatus.ACTIVE)

    def test_second_step_does_not_execute_after_first_failure(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Unsupported action first.",
                order=1,
                target_type="noop",
                metadata={"action_type": "invented_action"},
            ),
            PlanStep(
                id="step-2",
                description="Would be a safe noop.",
                order=2,
                target_type="noop",
                metadata={"action_type": "noop"},
            ),
        )

        result = executor.execute(plan, mode=ExecutionMode.LIVE)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].plan_step_id, "step-1")
        self.assertEqual(result.reasons, ["unsupported_action_type"])

    def test_missing_action_type_does_not_infer_from_target_or_description(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Stay active",
                status=GoalStatus.ACTIVE,
            )
        )
        executor = InternalActionExecutor(goal_store=goal_store, state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Complete goal goal-1 now.",
                order=1,
                target_type="goal",
                metadata={"goal_id": "goal-1", "status": "completed"},
            )
        )

        result = executor.execute(plan, mode=ExecutionMode.LIVE)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertEqual(result.reasons, ["unsupported_action_type"])
        self.assertEqual(goal_store.get_goal("goal-1").status, GoalStatus.ACTIVE)

    def test_emit_event_requires_explicit_event_payload(self) -> None:
        executor = InternalActionExecutor(state_dir=self.root)
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Emit something based on this sentence.",
                order=1,
                target_type="event",
                metadata={"action_type": "emit_event"},
            )
        )

        result = executor.execute(plan)

        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertTrue(result.halted)
        self.assertEqual(result.records[0].metadata["reason"], "invalid_action_payload")


class ExecutorFacetTests(unittest.TestCase):
    def test_returns_wait_when_no_plan(self) -> None:
        result = ExecutorFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute",
                metadata={"execute_plan": True},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertEqual(result.metadata["execution_result"], None)

    def test_returns_wait_when_execute_plan_not_requested(self) -> None:
        state = NexusState(
            facet_state={"planner": {"last_plan": make_plan(PlanStep()).to_dict()}}
        )

        result = ExecutorFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="execute"),
            state,
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertEqual(result.metadata["reasons"], ["execution_not_requested"])

    def test_live_metadata_without_execute_plan_still_waits(self) -> None:
        state = NexusState(
            facet_state={"planner": {"last_plan": make_plan(PlanStep()).to_dict()}}
        )

        result = ExecutorFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute",
                metadata={"dry_run": False},
            ),
            state,
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertEqual(result.metadata["reasons"], ["execution_not_requested"])

    def test_dry_run_executes_proposed_safe_internal_plan_when_requested(self) -> None:
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Safe noop.",
                order=1,
                target_type="noop",
                metadata={"action_type": "noop"},
            )
        )
        state = NexusState(facet_state={"planner": {"last_plan": plan.to_dict()}})

        result = ExecutorFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute",
                metadata={"execute_plan": True},
            ),
            state,
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertTrue(result.metadata["dry_run"])
        self.assertEqual(
            result.metadata["execution_result"]["overall_status"],
            ExecutionStatus.SUCCESS.value,
        )

    def test_does_not_propose_act_and_includes_execution_result(self) -> None:
        plan = make_plan(
            PlanStep(
                id="step-1",
                description="Safe noop.",
                order=1,
                target_type="noop",
                metadata={"action_type": "noop"},
            )
        )
        state = NexusState(facet_state={"planner": {"last_plan": plan.to_dict()}})

        result = ExecutorFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute",
                metadata={"execute_plan": True},
            ),
            state,
        )

        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertIn("execution_result", result.metadata)


class ExecutorIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_planner_and_executor(self) -> None:
        runtime = NexusRuntime(
            facets=[PlannerFacet(), ExecutorFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"execute_plan": True},
            )
        )

        planner_result = next(
            result for result in record.facet_results if result.facet_name == "planner"
        )
        executor_result = next(
            result for result in record.facet_results if result.facet_name == "executor"
        )

        self.assertEqual([result.facet_name for result in record.facet_results], ["planner", "executor"])
        self.assertIsNotNone(planner_result.metadata["plan"])
        self.assertEqual(
            executor_result.metadata["execution_result"]["plan_id"],
            planner_result.metadata["plan"]["id"],
        )
        self.assertEqual(
            executor_result.metadata["execution_result"]["overall_status"],
            ExecutionStatus.SUCCESS.value,
        )

    def test_executor_respects_planner_step_risk_and_status_metadata(self) -> None:
        runtime = NexusRuntime(
            facets=[PlannerFacet(), ExecutorFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"execute_plan": True, "target_type": "shell"},
            )
        )
        executor_result = next(
            result for result in record.facet_results if result.facet_name == "executor"
        )

        self.assertEqual(
            executor_result.metadata["execution_result"]["overall_status"],
            ExecutionStatus.SKIPPED.value,
        )
        self.assertEqual(executor_result.metadata["reasons"], ["requires_approval"])
        self.assertTrue(executor_result.metadata["halted"])


class CLIExecutorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_executor_flag_runs_without_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--executor",
                    "--content",
                    "hello executor",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(
            any(result["facet_name"] == "executor" for result in payload["facet_results"])
        )

    def test_planner_executor_execute_plan_produces_dry_run_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--executor",
                    "--execute-plan",
                    "--content",
                    "make a plan for this",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        executor_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "executor"
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(executor_result["metadata"]["dry_run"])
        self.assertEqual(
            executor_result["metadata"]["execution_result"]["overall_status"],
            ExecutionStatus.SUCCESS.value,
        )

    def test_live_flag_is_accepted_but_risky_steps_are_refused(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--executor",
                    "--execute-plan",
                    "--live",
                    "--content",
                    "make a plan for this",
                    "--metadata",
                    '{"target_type": "shell"}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        executor_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "executor"
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(executor_result["metadata"]["dry_run"])
        self.assertEqual(
            executor_result["metadata"]["execution_result"]["overall_status"],
            ExecutionStatus.SKIPPED.value,
        )
        self.assertEqual(executor_result["metadata"]["reasons"], ["requires_approval"])

    def test_live_without_execute_plan_does_not_execute(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--planner",
                    "--executor",
                    "--live",
                    "--content",
                    "make a plan for this",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        executor_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "executor"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(executor_result["proposed_decision"], DecisionAction.WAIT.value)
        self.assertEqual(executor_result["metadata"]["reasons"], ["execution_not_requested"])


if __name__ == "__main__":
    unittest.main()
