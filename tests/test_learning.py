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
    ExecutorFacet,
    GoalsFacet,
    LearningFacet,
    MemoryFacet,
    PlannerFacet,
    PolicyFacet,
    VerifierFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.learning import (
    AdjustmentRecord,
    AdjustmentStatus,
    AdjustmentTarget,
    LearningResult,
    LearningSignal,
    SignalSource,
    SignalType,
    build_learning_result,
    classify_execution_result_signal,
    classify_goal_lifecycle_signal,
    classify_user_feedback_signal,
    collect_learning_signals,
    generate_adjustments,
)
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.nexus import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusRuntime,
    NexusState,
)
from fullerene.planner import Plan, PlanStep
from fullerene.policy import SQLitePolicyStore
from fullerene.state import InMemoryStateStore
from fullerene.workspace_state import workspace_state_root
from fullerene.world_model import SQLiteWorldModelStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-learning-{uuid4().hex}"


def make_plan(step: PlanStep) -> Plan:
    return Plan(
        id="plan-learning-1",
        source_event_id="event-1",
        title="Learning integration plan",
        steps=[step],
        reasons=["test-plan"],
    )


class BrokenPlannerFacet:
    name = "planner"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state
        plan = make_plan(
            PlanStep(
                id="bad-step-1",
                description="Use an unsupported action.",
                order=1,
                target_type="noop",
                metadata={"action_type": "invented_action"},
            )
        )
        return FacetResult(
            facet_name=self.name,
            summary="Injected a malformed plan for failure-path testing.",
            proposed_decision=DecisionAction.RECORD,
            state_updates={"last_plan": plan.to_dict(), "last_plan_id": plan.id},
            metadata={"plan": plan.to_dict()},
        )


class LearningModelTests(unittest.TestCase):
    def test_learning_signal_creation(self) -> None:
        signal = LearningSignal(
            signal_type=SignalType.POSITIVE,
            source=SignalSource.USER_FEEDBACK,
            magnitude=1.0,
            source_event_id="event-1",
            reasons=["explicit_feedback_positive"],
        )

        self.assertEqual(signal.signal_type, SignalType.POSITIVE)
        self.assertEqual(signal.source, SignalSource.USER_FEEDBACK)
        self.assertEqual(signal.magnitude, 1.0)
        self.assertEqual(signal.to_dict()["source_event_id"], "event-1")

    def test_adjustment_record_creation(self) -> None:
        record = AdjustmentRecord(
            target=AdjustmentTarget.GOAL_PRIORITY,
            target_id="goal-1",
            target_facet="goals",
            field="priority",
            old_value=0.5,
            new_value=0.55,
            delta=0.05,
            status=AdjustmentStatus.APPLIED,
            source_signal_id="signal-1",
            reasons=["applied_minor_nudge"],
        )

        self.assertEqual(record.target, AdjustmentTarget.GOAL_PRIORITY)
        self.assertEqual(record.status, AdjustmentStatus.APPLIED)
        self.assertEqual(record.new_value, 0.55)

    def test_learning_result_creation(self) -> None:
        signal = LearningSignal(
            signal_type=SignalType.POSITIVE,
            source=SignalSource.USER_FEEDBACK,
            magnitude=1.0,
        )
        adjustment = AdjustmentRecord(
            target=AdjustmentTarget.BEHAVIOR_CONFIDENCE,
            target_facet="behavior",
            field="confidence",
            delta=0.05,
            status=AdjustmentStatus.PROPOSED,
            source_signal_id=signal.id,
        )
        result = LearningResult(
            signals=[signal],
            adjustments=[adjustment],
            overall_status="proposed",
        )

        self.assertEqual(len(result.signals), 1)
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.overall_status, "proposed")


class LearningSignalClassificationTests(unittest.TestCase):
    def test_explicit_positive_feedback_metadata(self) -> None:
        signal = classify_user_feedback_signal(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="ignored",
                metadata={"feedback": "positive"},
            )
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.POSITIVE)

    def test_explicit_negative_feedback_metadata(self) -> None:
        signal = classify_user_feedback_signal(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="ignored",
                metadata={"feedback": "negative"},
            )
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.NEGATIVE)

    def test_positive_phrase_detection(self) -> None:
        signal = classify_user_feedback_signal(
            Event(event_type=EventType.USER_MESSAGE, content="that worked")
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.POSITIVE)

    def test_negative_phrase_detection(self) -> None:
        signal = classify_user_feedback_signal(
            Event(event_type=EventType.USER_MESSAGE, content="that was wrong")
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.NEGATIVE)

    def test_execution_success_produces_weak_positive_signal(self) -> None:
        event = Event(event_type=EventType.USER_MESSAGE, content="run")
        state = NexusState(
            facet_state={
                "executor": {
                    "last_execution_result": {
                        "overall_status": "success",
                        "dry_run": True,
                        "reasons": ["execution_completed"],
                        "records": [],
                    }
                }
            }
        )

        signal = classify_execution_result_signal(event, state)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.POSITIVE)
        self.assertEqual(signal.magnitude, 0.3)

    def test_execution_failure_produces_failure_signal(self) -> None:
        event = Event(event_type=EventType.USER_MESSAGE, content="run")
        state = NexusState(
            facet_state={
                "executor": {
                    "last_execution_result": {
                        "overall_status": "failed",
                        "dry_run": True,
                        "reasons": ["unsupported_action_type"],
                        "records": [{"id": "exec-1"}],
                    }
                }
            }
        )

        signal = classify_execution_result_signal(event, state)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.FAILURE)
        self.assertEqual(signal.magnitude, 0.7)
        self.assertEqual(signal.source_record_id, "exec-1")

    def test_goal_completed_produces_positive_signal(self) -> None:
        signal = classify_goal_lifecycle_signal(
            Event(
                event_type=EventType.SYSTEM_NOTE,
                metadata={"goal_status": "completed"},
            )
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.signal_type, SignalType.POSITIVE)

    def test_goal_stale_or_abandoned_produces_weak_negative_signal(self) -> None:
        for goal_status in ("stale", "abandoned"):
            signal = classify_goal_lifecycle_signal(
                Event(
                    event_type=EventType.SYSTEM_NOTE,
                    metadata={"goal_status": goal_status},
                )
            )
            self.assertIsNotNone(signal)
            assert signal is not None
            self.assertEqual(signal.signal_type, SignalType.NEGATIVE)


class LearningAdjustmentGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_positive_signal_creates_positive_nudge(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Keep learning", priority=0.5))
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={"feedback": "positive", "target_goal_id": "goal-1"},
        )
        signals = collect_learning_signals(event, NexusState())

        adjustments = generate_adjustments(
            signals,
            event=event,
            state=NexusState(),
            goal_store=goal_store,
        )

        self.assertEqual(adjustments[0].status, AdjustmentStatus.APPLIED)
        self.assertEqual(adjustments[0].delta, 0.05)
        self.assertEqual(adjustments[0].new_value, 0.55)

    def test_negative_signal_creates_negative_nudge(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Stay on track", priority=0.5))
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that was wrong",
            metadata={"feedback": "negative", "target_goal_id": "goal-1"},
        )
        signals = collect_learning_signals(event, NexusState())

        adjustments = generate_adjustments(
            signals,
            event=event,
            state=NexusState(),
            goal_store=goal_store,
        )

        self.assertEqual(adjustments[0].status, AdjustmentStatus.APPLIED)
        self.assertEqual(adjustments[0].delta, -0.05)
        self.assertEqual(adjustments[0].new_value, 0.45)

    def test_delta_is_conservative(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Stay on track", priority=0.5))
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={"feedback": "positive", "target_goal_id": "goal-1"},
        )
        signals = collect_learning_signals(event, NexusState())

        adjustment = generate_adjustments(
            signals,
            event=event,
            state=NexusState(),
            goal_store=goal_store,
        )[0]

        self.assertLessEqual(abs(adjustment.delta), 0.05)

    def test_large_delta_becomes_proposal_not_applied(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="New goal", priority=0.0))
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={"feedback": "positive", "target_goal_id": "goal-1"},
        )
        signals = collect_learning_signals(event, NexusState())

        adjustment = generate_adjustments(
            signals,
            event=event,
            state=NexusState(),
            goal_store=goal_store,
        )[0]

        self.assertEqual(adjustment.status, AdjustmentStatus.PROPOSED)
        self.assertEqual(adjustment.delta, 0.1)
        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.0)

    def test_missing_target_creates_skipped_adjustment_with_reason(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={"feedback": "positive"},
        )

        result = build_learning_result(event, NexusState())

        self.assertEqual(result.adjustments[0].status, AdjustmentStatus.SKIPPED)
        self.assertIn("no_target", result.adjustments[0].reasons)

    def test_behavior_threshold_adjustment_is_proposed_without_behavior_store(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that was wrong",
            metadata={
                "feedback": "negative",
                "target_behavior_field": "behavior_threshold",
                "current_behavior_threshold": 0.6,
            },
        )
        signals = collect_learning_signals(event, NexusState())

        adjustment = generate_adjustments(
            signals,
            event=event,
            state=NexusState(),
        )[0]

        self.assertEqual(adjustment.target, AdjustmentTarget.BEHAVIOR_THRESHOLD)
        self.assertEqual(adjustment.status, AdjustmentStatus.PROPOSED)


class LearningStoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_goal_priority_can_be_nudged_if_goal_store_supports_it(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Finish task", priority=0.5))
        event = Event(
            event_type=EventType.SYSTEM_NOTE,
            metadata={"goal_status": "completed", "target_goal_id": "goal-1"},
        )

        result = build_learning_result(event, NexusState(), goal_store=goal_store)

        self.assertEqual(result.applied[0].target, AdjustmentTarget.GOAL_PRIORITY)
        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.55)

    def test_memory_salience_can_be_nudged_if_memory_store_supports_it(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Useful memory",
                salience=0.5,
                confidence=1.0,
            )
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={"feedback": "positive", "target_memory_id": "mem-1"},
        )

        result = build_learning_result(event, NexusState(), memory_store=memory_store)

        self.assertEqual(result.applied[0].target, AdjustmentTarget.MEMORY_SALIENCE)
        self.assertEqual(memory_store.get_memory("mem-1").salience, 0.55)

    def test_memory_salience_becomes_proposal_without_memory_store(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={
                "feedback": "positive",
                "target_memory_id": "mem-1",
                "current_memory_salience": 0.5,
            },
        )

        result = build_learning_result(event, NexusState())

        self.assertEqual(result.proposals[0].target, AdjustmentTarget.MEMORY_SALIENCE)
        self.assertEqual(result.proposals[0].status, AdjustmentStatus.PROPOSED)

    def test_values_clamp_to_valid_ranges(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="that worked",
            metadata={
                "feedback": "positive",
                "target_behavior_field": "behavior_threshold",
                "current_behavior_threshold": 1.0,
            },
        )

        result = build_learning_result(event, NexusState())

        self.assertLessEqual(result.adjustments[0].new_value or 1.0, 1.0)


class LearningFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_returns_wait_when_no_signal(self) -> None:
        result = LearningFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="hello there"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)

    def test_returns_record_when_signal_exists(self) -> None:
        result = LearningFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that worked",
                metadata={"feedback": "positive"},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)

    def test_never_proposes_act(self) -> None:
        result = LearningFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that worked",
                metadata={"feedback": "positive"},
            ),
            NexusState(),
        )

        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)

    def test_metadata_includes_signals_and_adjustments(self) -> None:
        result = LearningFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that worked",
                metadata={"feedback": "positive"},
            ),
            NexusState(),
        )

        self.assertIn("signals", result.metadata)
        self.assertIn("adjustments", result.metadata)
        self.assertIn("proposals", result.metadata)

    def test_works_without_stores(self) -> None:
        result = LearningFacet().process(
            Event(
                event_type=EventType.SYSTEM_NOTE,
                metadata={"goal_status": "completed", "target_goal_id": "goal-1"},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertTrue(result.metadata["proposals"])

    def test_works_with_memory_and_goal_stores_if_provided(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Finish task", priority=0.5))
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Keep this",
                salience=0.5,
                confidence=1.0,
            )
        )
        facet = LearningFacet(memory_store=memory_store, goal_store=goal_store)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that worked",
                metadata={
                    "feedback": "positive",
                    "target_goal_id": "goal-1",
                    "target_memory_id": "mem-1",
                },
            ),
            NexusState(),
        )

        self.assertTrue(result.metadata["applied"])


class LearningIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_nexus_runs_with_full_stack_and_learning_after_executor(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        policy_store = SQLitePolicyStore(self.root / "policy.sqlite3")
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store),
                GoalsFacet(goal_store),
                WorldModelFacet(world_store),
                BehaviorFacet(),
                PolicyFacet(policy_store, state_dir=self.root),
                PlannerFacet(
                    goal_store=goal_store,
                    world_model_store=world_store,
                    policy_store=policy_store,
                    state_dir=self.root,
                ),
                ExecutorFacet(
                    goal_store=goal_store,
                    world_model_store=world_store,
                    memory_store=memory_store,
                    state_dir=self.root,
                ),
                LearningFacet(memory_store=memory_store, goal_store=goal_store),
                VerifierFacet(state_dir=self.root),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this",
                metadata={"execute_plan": True},
            )
        )

        facet_names = [result.facet_name for result in record.facet_results]
        self.assertEqual(
            facet_names,
            [
                "memory",
                "goals",
                "world_model",
                "behavior",
                "policy",
                "planner",
                "executor",
                "learning",
                "verifier",
            ],
        )
        learning_result = next(
            result for result in record.facet_results if result.facet_name == "learning"
        )
        self.assertTrue(
            any(
                signal["source"] == SignalSource.EXECUTION_RESULT.value
                for signal in learning_result.metadata["signals"]
            )
        )

    def test_execution_failure_can_be_observed_by_learning(self) -> None:
        runtime = NexusRuntime(
            facets=[
                BrokenPlannerFacet(),
                ExecutorFacet(state_dir=self.root),
                LearningFacet(),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute the broken plan",
                metadata={"execute_plan": True},
            )
        )
        learning_result = next(
            result for result in record.facet_results if result.facet_name == "learning"
        )

        self.assertTrue(
            any(
                signal["signal_type"] == SignalType.FAILURE.value
                for signal in learning_result.metadata["signals"]
            )
        )

    def test_no_major_state_change_occurs_without_traceable_adjustment_record(self) -> None:
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        goal_store.add_goal(Goal(id="goal-1", description="Keep moving", priority=0.5))
        runtime = NexusRuntime(
            facets=[GoalsFacet(goal_store), LearningFacet(goal_store=goal_store)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.SYSTEM_NOTE,
                metadata={"goal_status": "completed", "target_goal_id": "goal-1"},
            )
        )
        learning_result = next(
            result for result in record.facet_results if result.facet_name == "learning"
        )
        applied = learning_result.metadata["applied"][0]

        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.55)
        self.assertTrue(applied["source_signal_id"])


class CLILearningIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_learning_flag_runs_without_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--learning",
                    "--content",
                    "hello learning",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(
            any(result["facet_name"] == "learning" for result in payload["facet_results"])
        )

    def test_positive_feedback_smoke_command_creates_learning_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--learning",
                    "--content",
                    "that worked",
                    "--metadata",
                    '{"feedback": "positive"}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        learning_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "learning"
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(learning_result["metadata"]["signals"])

    def test_negative_feedback_smoke_command_creates_learning_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--learning",
                    "--goals",
                    "--content",
                    "that was wrong",
                    "--metadata",
                    '{"feedback": "negative"}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        learning_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "learning"
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(learning_result["metadata"]["signals"])


if __name__ == "__main__":
    unittest.main()
