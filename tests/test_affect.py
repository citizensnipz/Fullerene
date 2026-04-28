from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.affect import (
    AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0,
    AffectHistoryBuffer,
    AffectResult,
    AffectState,
    DeterministicAffectDeriver,
)
from fullerene.cli import main as cli_main
from fullerene.facets import (
    AffectFacet,
    AttentionFacet,
    BehaviorFacet,
    ContextFacet,
    ExecutorFacet,
    GoalsFacet,
    LearningFacet,
    MemoryFacet,
    PlannerFacet,
    PolicyFacet,
    VerifierFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, GoalSource, SQLiteGoalStore
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.nexus import (
    DecisionAction,
    Event,
    EventType,
    NexusRuntime,
    NexusState,
)
from fullerene.policy import SQLitePolicyStore
from fullerene.state import FileStateStore, InMemoryStateStore
from fullerene.workspace_state import workspace_state_root
from fullerene.world_model import Belief, BeliefSource, SQLiteWorldModelStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-affect-{uuid4().hex}"


class AffectModelTests(unittest.TestCase):
    def test_affect_state_creation_and_round_trip(self) -> None:
        state = AffectState(
            id="affect-1",
            valence=0.4,
            arousal=0.2,
            dominance=0.7,
            novelty=0.6,
            components={"valence": {"raw": 0.4}},
            metadata={"strategy": AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0},
        )

        round_tripped = AffectState.from_dict(state.to_dict())

        self.assertEqual(round_tripped, state)
        self.assertEqual(round_tripped.valence, 0.4)

    def test_affect_state_clamps_ranges(self) -> None:
        state = AffectState(
            valence=2.0,
            arousal=-1.0,
            dominance=1.7,
            novelty=3.0,
        )

        self.assertEqual(state.valence, 1.0)
        self.assertEqual(state.arousal, 0.0)
        self.assertEqual(state.dominance, 1.0)
        self.assertEqual(state.novelty, 1.0)

    def test_affect_result_creation_and_round_trip(self) -> None:
        current = AffectState(
            id="affect-1",
            valence=0.2,
            arousal=0.3,
            dominance=0.6,
            novelty=0.4,
        )
        history = [
            AffectState(
                id="affect-0",
                valence=0.1,
                arousal=0.2,
                dominance=0.5,
                novelty=0.5,
            )
        ]
        result = AffectResult(
            current_state=current,
            history=history,
            reasons=["novelty_default_moderate"],
            metadata={"history_count": 1},
        )

        round_tripped = AffectResult.from_dict(result.to_dict())

        self.assertEqual(round_tripped, result)
        self.assertEqual(
            round_tripped.strategy,
            AFFECT_STRATEGY_DETERMINISTIC_VAD_NOVELTY_V0,
        )


class AffectDerivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deriver = DeterministicAffectDeriver()

    def test_neutral_default_state_works_with_no_signals(self) -> None:
        result = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="hello affect"),
            NexusState(),
        )

        self.assertEqual(result.current_state.valence, 0.0)
        self.assertEqual(result.current_state.arousal, 0.0)
        self.assertEqual(result.current_state.dominance, 0.5)
        self.assertEqual(result.current_state.novelty, 0.5)

    def test_positive_feedback_increases_valence(self) -> None:
        result = self.deriver.derive(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that worked",
                metadata={"feedback": "positive"},
            ),
            NexusState(),
        )

        self.assertGreater(result.current_state.valence, 0.0)

    def test_negative_feedback_lowers_valence(self) -> None:
        result = self.deriver.derive(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="that was wrong",
                metadata={"feedback": "negative"},
            ),
            NexusState(),
        )

        self.assertLess(result.current_state.valence, 0.0)

    def test_pressure_metadata_increases_arousal(self) -> None:
        baseline = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="steady"),
            NexusState(),
        )
        pressured = self.deriver.derive(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="steady",
                metadata={"pressure": 0.8},
            ),
            NexusState(),
        )

        self.assertGreater(
            pressured.current_state.arousal,
            baseline.current_state.arousal,
        )

    def test_novelty_metadata_controls_novelty(self) -> None:
        result = self.deriver.derive(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="something new happened",
                metadata={"novelty": 0.9},
            ),
            NexusState(),
        )

        self.assertEqual(result.current_state.novelty, 0.9)

    def test_executor_success_increases_or_maintains_dominance(self) -> None:
        baseline = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="baseline"),
            NexusState(),
        )
        state = NexusState(
            facet_state={
                "executor": {
                    "last_execution_result": {
                        "overall_status": "success",
                        "dry_run": True,
                        "reasons": ["execution_completed"],
                        "records": [{"status": "success"}],
                    }
                }
            }
        )

        result = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="execution succeeded"),
            state,
        )

        self.assertGreaterEqual(
            result.current_state.dominance,
            baseline.current_state.dominance,
        )

    def test_executor_failure_lowers_dominance(self) -> None:
        state = NexusState(
            facet_state={
                "executor": {
                    "last_execution_result": {
                        "overall_status": "failed",
                        "dry_run": True,
                        "reasons": ["unsupported_action_type"],
                        "records": [{"status": "failed"}],
                    }
                }
            }
        )

        result = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="execution failed"),
            state,
        )

        self.assertLess(result.current_state.dominance, 0.5)

    def test_high_world_belief_confidence_increases_or_maintains_dominance(self) -> None:
        state = NexusState(
            facet_state={
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "This is likely true",
                            "confidence": 0.95,
                            "status": "active",
                        }
                    ]
                }
            }
        )

        result = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="stable world"),
            state,
        )

        self.assertGreater(result.current_state.dominance, 0.5)

    def test_low_or_contradicted_world_confidence_lowers_dominance(self) -> None:
        state = NexusState(
            facet_state={
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "This was contradicted",
                            "confidence": 0.1,
                            "status": "contradicted",
                        }
                    ]
                }
            }
        )

        result = self.deriver.derive(
            Event(event_type=EventType.USER_MESSAGE, content="unstable world"),
            state,
        )

        self.assertLess(result.current_state.dominance, 0.5)

    def test_all_values_clamp_correctly(self) -> None:
        state = NexusState(
            facet_state={
                "executor": {
                    "last_execution_result": {
                        "overall_status": "success",
                        "dry_run": False,
                        "reasons": ["execution_completed"],
                        "records": [{"status": "success"}],
                    }
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "Likely true",
                            "confidence": 2.0,
                            "status": "active",
                        }
                    ]
                },
            }
        )
        result = self.deriver.derive(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="very loud",
                metadata={"pressure": 2.0, "novelty": -1.0, "feedback": "positive"},
            ),
            state,
        )

        self.assertLessEqual(result.current_state.valence, 1.0)
        self.assertGreaterEqual(result.current_state.valence, -1.0)
        self.assertLessEqual(result.current_state.arousal, 1.0)
        self.assertGreaterEqual(result.current_state.arousal, 0.0)
        self.assertLessEqual(result.current_state.dominance, 1.0)
        self.assertGreaterEqual(result.current_state.dominance, 0.0)
        self.assertLessEqual(result.current_state.novelty, 1.0)
        self.assertGreaterEqual(result.current_state.novelty, 0.0)


class AffectHistoryTests(unittest.TestCase):
    def test_history_keeps_last_n_states(self) -> None:
        history = AffectHistoryBuffer(max_size=2)
        history.append(AffectState(id="affect-1"))
        history.append(AffectState(id="affect-2"))
        history.append(AffectState(id="affect-3"))

        self.assertEqual(
            [state.id for state in history.states],
            ["affect-2", "affect-3"],
        )

    def test_history_does_not_exceed_configured_size(self) -> None:
        payload = [
            AffectState(id="affect-1").to_dict(),
            AffectState(id="affect-2").to_dict(),
            AffectState(id="affect-3").to_dict(),
        ]

        history = AffectHistoryBuffer.from_payload(payload, max_size=2)

        self.assertEqual(len(history.states), 2)


class AffectFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_returns_record_with_affect_metadata(self) -> None:
        result = AffectFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="hello affect"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertIn("affect_result", result.metadata)
        self.assertIn("affect_state", result.metadata)

    def test_never_proposes_act(self) -> None:
        result = AffectFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="that worked"),
            NexusState(),
        )

        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)

    def test_works_with_event_metadata_only(self) -> None:
        result = AffectFacet().process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="something new happened",
                metadata={"feedback": "positive", "pressure": 0.7, "novelty": 0.9},
            ),
            NexusState(),
        )

        affect_state = result.metadata["affect_state"]
        self.assertGreater(affect_state["valence"], 0.0)
        self.assertGreater(affect_state["arousal"], 0.0)
        self.assertEqual(affect_state["novelty"], 0.9)

    def test_works_with_facet_metadata_if_available(self) -> None:
        state = NexusState(
            facet_state={
                "memory": {
                    "last_working_memory_ids": ["mem-1", "mem-2", "mem-3"],
                    "last_relevant_memory_ids": ["mem-1"],
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "Likely stable",
                            "confidence": 0.8,
                            "status": "active",
                        }
                    ]
                },
                "executor": {
                    "last_execution_result": {
                        "overall_status": "success",
                        "dry_run": True,
                        "reasons": ["execution_completed"],
                        "records": [{"status": "success"}],
                    }
                },
                "attention": {
                    "last_attention_result": {
                        "scores": {"event:event-1": 0.7, "goal:goal-1": 0.4}
                    }
                },
            }
        )

        result = AffectFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="follow up"),
            state,
        )

        affect_state = result.metadata["affect_state"]
        self.assertGreater(affect_state["arousal"], 0.0)
        self.assertGreater(affect_state["dominance"], 0.5)
        self.assertLess(affect_state["novelty"], 1.0)

    def test_does_not_mutate_other_stores(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Keep memory stable",
                salience=0.5,
                confidence=1.0,
            )
        )
        goal_store.add_goal(
            Goal(id="goal-1", description="Keep goal stable", priority=0.6)
        )
        world_store.add_belief(
            Belief(id="belief-1", claim="Keep belief stable", confidence=0.7)
        )
        state = NexusState(
            facet_state={
                "memory": {
                    "last_working_memory_ids": ["mem-1"],
                    "last_relevant_memory_ids": ["mem-1"],
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "Keep belief stable",
                            "confidence": 0.7,
                            "status": "active",
                        }
                    ]
                },
                "goals": {
                    "last_relevant_goals": [
                        {
                            "id": "goal-1",
                            "description": "Keep goal stable",
                            "priority": 0.6,
                        }
                    ]
                },
            }
        )

        result = AffectFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="stay stable"),
            state,
        )

        self.assertEqual(memory_store.get_memory("mem-1").salience, 0.5)
        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.6)
        self.assertEqual(world_store.get_belief("belief-1").confidence, 0.7)
        self.assertIn("affect_result", result.metadata)


class AffectIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_nexus_runs_with_affect_enabled(self) -> None:
        runtime = NexusRuntime(
            facets=[AffectFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="hello affect")
        )

        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["affect"],
        )
        self.assertIn("affect_result", record.facet_results[0].metadata)

    def test_affect_can_run_alongside_full_stack(self) -> None:
        state_store = FileStateStore(self.root)
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        policy_store = SQLitePolicyStore(self.root / "policy.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Previous task context",
                salience=0.7,
                confidence=1.0,
                tags=["task"],
            )
        )
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Plan the task",
                priority=0.8,
                tags=["task"],
                source=GoalSource.USER,
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="The task outcome is still uncertain",
                confidence=0.2,
                tags=["task"],
                source=BeliefSource.USER,
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(memory_store, window_size=3),
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
                AttentionFacet(memory_store=memory_store),
                AffectFacet(history_size=5),
                VerifierFacet(state_dir=self.root),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="make a plan for this task",
                metadata={
                    "feedback": "negative",
                    "pressure": 0.8,
                    "execute_plan": True,
                    "tags": ["task"],
                },
            )
        )

        facet_names = [result.facet_name for result in record.facet_results]
        self.assertEqual(
            facet_names,
            [
                "context",
                "memory",
                "goals",
                "world_model",
                "behavior",
                "policy",
                "planner",
                "executor",
                "learning",
                "attention",
                "affect",
                "verifier",
            ],
        )
        affect_result = next(
            result for result in record.facet_results if result.facet_name == "affect"
        )
        self.assertIn("affect_result", affect_result.metadata)
        self.assertTrue((self.root / "state.json").exists())
        self.assertTrue((self.root / "runtime-log.jsonl").exists())

        log_payload = json.loads(
            (self.root / "runtime-log.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )
        persisted_affect = next(
            result
            for result in log_payload["facet_results"]
            if result["facet_name"] == "affect"
        )
        self.assertIn("affect_result", persisted_affect["metadata"])


class CLIAffectIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_affect_flag_runs_without_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--affect",
                    "--content",
                    "that worked",
                    "--metadata",
                    '{"feedback": "positive"}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        affect_result = next(
            result
            for result in payload["facet_results"]
            if result["facet_name"] == "affect"
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("affect_result", affect_result["metadata"])

    def test_affect_history_size_is_accepted(self) -> None:
        first_stdout = io.StringIO()
        second_stdout = io.StringIO()

        with redirect_stdout(first_stdout):
            first_exit = cli_main(
                [
                    "--affect",
                    "--affect-history-size",
                    "2",
                    "--content",
                    "first",
                    "--state-dir",
                    str(self.root),
                ]
            )
        with redirect_stdout(second_stdout):
            second_exit = cli_main(
                [
                    "--affect",
                    "--affect-history-size",
                    "2",
                    "--content",
                    "second",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(second_stdout.getvalue())
        affect_result = next(
            result
            for result in payload["facet_results"]
            if result["facet_name"] == "affect"
        )

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        self.assertEqual(affect_result["metadata"]["history_count"], 2)

    def test_pressure_and_novelty_flags_influence_affect_output(self) -> None:
        baseline_stdout = io.StringIO()
        boosted_stdout = io.StringIO()

        with redirect_stdout(baseline_stdout):
            baseline_exit = cli_main(
                [
                    "--affect",
                    "--content",
                    "something happened",
                    "--state-dir",
                    str(self.root / "baseline"),
                ]
            )
        with redirect_stdout(boosted_stdout):
            boosted_exit = cli_main(
                [
                    "--affect",
                    "--pressure",
                    "0.8",
                    "--novelty",
                    "0.9",
                    "--content",
                    "something happened",
                    "--state-dir",
                    str(self.root / "boosted"),
                ]
            )

        baseline_payload = json.loads(baseline_stdout.getvalue())
        boosted_payload = json.loads(boosted_stdout.getvalue())
        baseline_affect = next(
            result
            for result in baseline_payload["facet_results"]
            if result["facet_name"] == "affect"
        )
        boosted_affect = next(
            result
            for result in boosted_payload["facet_results"]
            if result["facet_name"] == "affect"
        )

        self.assertEqual(baseline_exit, 0)
        self.assertEqual(boosted_exit, 0)
        self.assertLess(
            baseline_affect["metadata"]["affect_state"]["arousal"],
            boosted_affect["metadata"]["affect_state"]["arousal"],
        )
        self.assertLess(
            baseline_affect["metadata"]["affect_state"]["novelty"],
            boosted_affect["metadata"]["affect_state"]["novelty"],
        )


if __name__ == "__main__":
    unittest.main()
