from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.attention import (
    ATTENTION_COMPONENT_WEIGHTS,
    AttentionItem,
    AttentionResult,
    AttentionSource,
    FixedWeightAttentionScorer,
)
from fullerene.cli import main as cli_main
from fullerene.executor import ExecutionStatus
from fullerene.facets import (
    AttentionFacet,
    ExecutorFacet,
    GoalsFacet,
    MemoryFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, GoalSource, SQLiteGoalStore
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
from fullerene.state import InMemoryStateStore
from fullerene.workspace_state import workspace_state_root
from fullerene.world_model import Belief, BeliefSource, SQLiteWorldModelStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-attention-{uuid4().hex}"


class PlanCarrierFacet:
    name = "planner"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state
        plan = Plan(
            id="plan-attention-1",
            source_event_id=event.event_id,
            title="Attention integration plan",
            steps=[
                PlanStep(
                    id="step-1",
                    description="Safe noop for executor signal coverage.",
                    order=1,
                    target_type="noop",
                    metadata={"action_type": "noop"},
                )
            ],
            reasons=["attention-integration"],
        )
        return FacetResult(
            facet_name=self.name,
            summary="Injected a deterministic noop plan for attention integration.",
            proposed_decision=DecisionAction.RECORD,
            state_updates={"last_plan": plan.to_dict(), "last_plan_id": plan.id},
            metadata={"plan": plan.to_dict()},
        )


class AttentionModelTests(unittest.TestCase):
    def test_attention_item_creation_and_round_trip(self) -> None:
        item = AttentionItem(
            id="event:event-1",
            source=AttentionSource.EVENT,
            source_id="event-1",
            content="new urgent thing happened",
            score=0.61,
            components={
                "memory_salience": 0.25,
                "goal_priority": 0.0,
                "pressure": 0.16,
                "novelty": 0.09,
                "belief_uncertainty": 0.10,
                "execution_recency": 0.01,
            },
            dominant_component="memory_salience",
            metadata={"raw_signals": {"pressure": 0.8}},
        )

        round_tripped = AttentionItem.from_dict(item.to_dict())

        self.assertEqual(round_tripped, item)
        self.assertEqual(round_tripped.source, AttentionSource.EVENT)
        self.assertEqual(round_tripped.metadata["raw_signals"]["pressure"], 0.8)

    def test_attention_result_creation_and_round_trip(self) -> None:
        item = AttentionItem(
            id="goal:goal-1",
            source=AttentionSource.GOAL,
            source_id="goal-1",
            content="Finish the task",
            score=0.4,
            components={"goal_priority": 0.25, "pressure": 0.15},
            dominant_component="goal_priority",
        )
        result = AttentionResult(
            focus_items=[item],
            scores={"goal:goal-1": 0.4, "event:event-1": 0.2},
            dominant_source=AttentionSource.GOAL,
            metadata={"candidate_count": 2},
        )

        round_tripped = AttentionResult.from_dict(result.to_dict())

        self.assertEqual(round_tripped, result)
        self.assertEqual(round_tripped.dominant_source, AttentionSource.GOAL)
        self.assertEqual(round_tripped.metadata["candidate_count"], 2)


class AttentionScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = FixedWeightAttentionScorer()

    def test_missing_signals_are_treated_as_zero(self) -> None:
        scored = self.scorer.score_candidate(
            {
                "id": "memory:1",
                "source": "memory",
                "content": "stored memory",
                "memory_salience": 0.5,
            }
        )

        self.assertAlmostEqual(scored["score"], 0.125)
        self.assertAlmostEqual(scored["components"]["memory_salience"], 0.125)
        self.assertEqual(scored["components"]["goal_priority"], 0.0)
        self.assertEqual(scored["metadata"]["raw_signals"]["pressure"], 0.0)

    def test_weighted_score_uses_fixed_weights(self) -> None:
        scored = self.scorer.score_candidate(
            {
                "id": "event:1",
                "source": "event",
                "content": "focus now",
                "memory_salience": 1.0,
                "goal_priority": 0.8,
                "pressure": 0.5,
                "novelty": 0.4,
                "belief_uncertainty": 0.3,
                "execution_recency": 0.2,
            }
        )

        expected = (
            1.0 * ATTENTION_COMPONENT_WEIGHTS["memory_salience"]
            + 0.8 * ATTENTION_COMPONENT_WEIGHTS["goal_priority"]
            + 0.5 * ATTENTION_COMPONENT_WEIGHTS["pressure"]
            + 0.4 * ATTENTION_COMPONENT_WEIGHTS["novelty"]
            + 0.3 * ATTENTION_COMPONENT_WEIGHTS["belief_uncertainty"]
            + 0.2 * ATTENTION_COMPONENT_WEIGHTS["execution_recency"]
        )
        self.assertAlmostEqual(scored["score"], expected)

    def test_scores_are_clamped_to_unit_interval(self) -> None:
        scored = self.scorer.score_candidate(
            {
                "id": "event:1",
                "source": "event",
                "content": "overflow",
                "memory_salience": 2.0,
                "goal_priority": 2.0,
                "pressure": 2.0,
                "novelty": 2.0,
                "belief_uncertainty": 2.0,
                "execution_recency": 2.0,
            }
        )

        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["metadata"]["raw_signals"]["memory_salience"], 1.0)

    def test_dominant_component_is_computed_from_weighted_components(self) -> None:
        scored = self.scorer.score_candidate(
            {
                "id": "goal:1",
                "source": "goal",
                "content": "finish the task",
                "goal_priority": 1.0,
                "pressure": 0.8,
            }
        )

        self.assertEqual(scored["dominant_component"], "goal_priority")

    def test_top_n_selection_works(self) -> None:
        evaluated = self.scorer.evaluate(
            [
                {
                    "id": "event:1",
                    "source": "event",
                    "content": "focus",
                    "novelty": 0.8,
                },
                {
                    "id": "goal:1",
                    "source": "goal",
                    "content": "finish",
                    "goal_priority": 1.0,
                },
                {
                    "id": "belief:1",
                    "source": "belief",
                    "content": "uncertain belief",
                    "belief_uncertainty": 0.9,
                },
            ],
            top_n=2,
        )

        self.assertEqual(len(evaluated["selected"]), 2)
        self.assertEqual(evaluated["selected"][0]["id"], "goal:1")

    def test_higher_score_wins_competition(self) -> None:
        evaluated = self.scorer.evaluate(
            [
                {
                    "id": "memory:1",
                    "source": "memory",
                    "content": "low salience",
                    "memory_salience": 0.2,
                },
                {
                    "id": "memory:2",
                    "source": "memory",
                    "content": "high salience",
                    "memory_salience": 0.9,
                },
            ],
            top_n=1,
        )

        self.assertEqual(evaluated["selected"][0]["id"], "memory:2")


class AttentionFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_works_with_only_current_event(self) -> None:
        facet = AttentionFacet(top_n=3)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="new urgent thing happened",
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertTrue(result.metadata["focus_items"])
        self.assertEqual(result.metadata["focus_items"][0]["source"], "event")

    def test_returns_wait_when_no_meaningful_signals_exist(self) -> None:
        facet = AttentionFacet(top_n=3)

        result = facet.process(
            Event(event_type=EventType.SYSTEM_TICK),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertEqual(result.metadata["focus_items"], [])

    def test_returns_record_with_focus_items_when_candidates_exist(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Remember the urgent production bug",
                salience=0.9,
                confidence=1.0,
                tags=["urgent", "bug"],
            )
        )
        facet = AttentionFacet(memory_store=memory_store, top_n=3)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {
                            "id": "goal-1",
                            "description": "Fix the production bug",
                            "priority": 0.8,
                            "score": 1.4,
                            "shared_tags": ["bug"],
                        }
                    ]
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "The bug cause is still uncertain",
                            "confidence": 0.2,
                            "score": 1.1,
                        }
                    ]
                },
                "executor": {
                    "last_execution_result": {
                        "plan_id": "plan-1",
                        "overall_status": "success",
                        "dry_run": True,
                        "reasons": ["execution_completed"],
                        "records": [
                            {
                                "id": "exec-1",
                                "created_at": Event(
                                    event_type=EventType.SYSTEM_NOTE
                                ).timestamp.isoformat(),
                                "status": "success",
                            }
                        ],
                    }
                },
            }
        )

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I focus on?",
                metadata={"pressure": 0.8},
            ),
            state,
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertTrue(result.metadata["focus_items"])
        self.assertIn("memory:mem-1", result.metadata["scores"])
        self.assertIn("goal:goal-1", result.metadata["scores"])
        self.assertIn("belief:belief-1", result.metadata["scores"])

    def test_never_proposes_act(self) -> None:
        facet = AttentionFacet(top_n=3)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what matters here?",
            ),
            NexusState(),
        )

        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)

    def test_metadata_includes_weights_scores_dominant_source_and_strategy(self) -> None:
        facet = AttentionFacet(top_n=3)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what matters here?",
                metadata={"pressure": 0.7, "novelty": 0.6},
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["strategy"], "fixed_weight_competition_v0")
        self.assertEqual(result.metadata["top_n"], 3)
        self.assertEqual(result.metadata["weights"], ATTENTION_COMPONENT_WEIGHTS)
        self.assertIn("scores", result.metadata)
        self.assertIn("dominant_source", result.metadata)

    def test_top_n_is_respected(self) -> None:
        facet = AttentionFacet(top_n=2)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {
                            "id": "goal-1",
                            "description": "Finish task",
                            "priority": 0.9,
                            "score": 1.3,
                        }
                    ]
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "Outcome is uncertain",
                            "confidence": 0.1,
                            "score": 1.0,
                        }
                    ]
                },
            }
        )

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I focus on?",
            ),
            state,
        )

        self.assertLessEqual(len(result.metadata["focus_items"]), 2)


class AttentionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_nexus_runs_with_attention_enabled(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        runtime = NexusRuntime(
            facets=[MemoryFacet(memory_store), AttentionFacet(memory_store=memory_store)],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="new urgent thing happened")
        )

        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["memory", "attention"],
        )
        attention_result = record.facet_results[1]
        self.assertIn("attention_result", attention_result.metadata)

    def test_attention_reads_memory_goals_world_and_executor_metadata(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-prior-1",
                memory_type=MemoryType.EPISODIC,
                content="Previous focus note about the current task",
                salience=0.8,
                confidence=1.0,
                tags=["focus"],
            )
        )
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Track what to focus on",
                priority=0.8,
                tags=["focus"],
                source=GoalSource.USER,
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="The current focus is still uncertain",
                confidence=0.2,
                tags=["focus"],
                source=BeliefSource.USER,
            )
        )
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store),
                GoalsFacet(goal_store),
                WorldModelFacet(world_store),
                PlanCarrierFacet(),
                ExecutorFacet(state_dir=self.root),
                AttentionFacet(memory_store=memory_store, top_n=3),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I focus on?",
                metadata={"execute_plan": True, "pressure": 0.8},
            )
        )
        attention_result = next(
            result for result in record.facet_results if result.facet_name == "attention"
        )
        attention_payload = attention_result.metadata["attention_result"]

        self.assertIn("memory", attention_payload["metadata"]["available_sources"])
        self.assertIn("goal", attention_payload["metadata"]["available_sources"])
        self.assertIn("belief", attention_payload["metadata"]["available_sources"])
        self.assertIn("execution", attention_payload["metadata"]["available_sources"])
        self.assertTrue(
            any(item_id.startswith("execution:") for item_id in attention_payload["scores"])
        )

    def test_attention_does_not_mutate_other_stores(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        goal_store.add_goal(
            Goal(id="goal-1", description="Keep goal stable", priority=0.6)
        )
        world_store.add_belief(
            Belief(id="belief-1", claim="Keep belief stable", confidence=0.7)
        )
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Keep memory stable",
                salience=0.5,
                confidence=1.0,
            )
        )
        facet = AttentionFacet(memory_store=memory_store)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {
                            "id": "goal-1",
                            "description": "Keep goal stable",
                            "priority": 0.6,
                            "score": 1.1,
                        }
                    ]
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {
                            "id": "belief-1",
                            "claim": "Keep belief stable",
                            "confidence": 0.7,
                            "score": 0.9,
                        }
                    ]
                },
            }
        )

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="keep stable"),
            state,
        )

        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.6)
        self.assertEqual(world_store.get_belief("belief-1").confidence, 0.7)
        self.assertEqual(memory_store.get_memory("mem-1").salience, 0.5)
        self.assertIn("attention_result", result.metadata)

    def test_attention_does_not_broadcast_in_v0(self) -> None:
        facet = AttentionFacet()

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="focus on this"),
            NexusState(),
        )

        payload = json.dumps(result.metadata)
        self.assertNotIn("ignition", payload)
        self.assertNotIn("refractory", payload)


class CLIAttentionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_attention_flag_runs_without_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--attention",
                    "--content",
                    "new urgent thing happened",
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        attention_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "attention"
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("attention_result", attention_result["metadata"])

    def test_attention_top_n_controls_focus_item_count(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--world",
                    "--attention",
                    "--attention-top-n",
                    "2",
                    "--content",
                    "what should I focus on?",
                    "--metadata",
                    '{"create_goal": true, "create_belief": true, "tags": ["focus"]}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        attention_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "attention"
        )

        self.assertEqual(exit_code, 0)
        self.assertLessEqual(len(attention_result["metadata"]["focus_items"]), 2)

    def test_pressure_and_novelty_flags_affect_score(self) -> None:
        baseline_stdout = io.StringIO()
        boosted_stdout = io.StringIO()

        with redirect_stdout(baseline_stdout):
            baseline_exit = cli_main(
                [
                    "--json",
                    "--attention",
                    "--content",
                    "new urgent thing happened",
                    "--state-dir",
                    str(self.root / "baseline"),
                ]
            )
        with redirect_stdout(boosted_stdout):
            boosted_exit = cli_main(
                [
                    "--json",
                    "--attention",
                    "--pressure",
                    "0.8",
                    "--novelty",
                    "0.6",
                    "--content",
                    "new urgent thing happened",
                    "--state-dir",
                    str(self.root / "boosted"),
                ]
            )

        baseline_payload = json.loads(baseline_stdout.getvalue())
        boosted_payload = json.loads(boosted_stdout.getvalue())
        baseline_attention = next(
            result for result in baseline_payload["facet_results"] if result["facet_name"] == "attention"
        )
        boosted_attention = next(
            result for result in boosted_payload["facet_results"] if result["facet_name"] == "attention"
        )
        baseline_score = baseline_attention["metadata"]["focus_items"][0]["score"]
        boosted_score = boosted_attention["metadata"]["focus_items"][0]["score"]

        self.assertEqual(baseline_exit, 0)
        self.assertEqual(boosted_exit, 0)
        self.assertGreater(boosted_score, baseline_score)

    def test_smoke_command_with_memory_goals_world_produces_attention_metadata(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--memory",
                    "--goals",
                    "--world",
                    "--attention",
                    "--content",
                    "what should I focus on?",
                    "--metadata",
                    '{"create_goal": true, "create_belief": true, "tags": ["focus"]}',
                    "--state-dir",
                    str(self.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        attention_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "attention"
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("attention_result", attention_result["metadata"])
        self.assertTrue(attention_result["metadata"]["scores"])


if __name__ == "__main__":
    unittest.main()
