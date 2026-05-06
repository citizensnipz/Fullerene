from __future__ import annotations

import io
import json
import shutil
import sqlite3
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.facets import BehaviorFacet, EchoFacet, GoalsFacet, MemoryFacet
from fullerene.workspace_state import workspace_state_root
from fullerene.goals import (
    Goal,
    GoalSource,
    GoalStatus,
    SQLiteGoalStore,
    goal_keyword_overlap,
    normalize_goal_description,
)
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-goals-{uuid4().hex}"


class GoalModelTests(unittest.TestCase):
    def test_goal_round_trips_through_dict(self) -> None:
        goal = Goal(
            id="goal-1",
            description="Track my tasks",
            priority=0.8,
            status=GoalStatus.ACTIVE,
            tags=["Tasks", "tasks", " Work "],
            source=GoalSource.USER,
            metadata={"origin": "manual"},
        )

        round_tripped = Goal.from_dict(goal.to_dict())

        self.assertEqual(round_tripped, goal)
        self.assertEqual(round_tripped.tags, ["tasks", "work"])


class GoalNormalizationTests(unittest.TestCase):
    def test_equivalent_intent_phrases_normalize_to_same_key(self) -> None:
        variants = (
            "I should remember to finish Fullerene",
            "remember to finish Fullerene",
            "finish Fullerene",
        )

        normalized = {normalize_goal_description(value) for value in variants}

        self.assertEqual(normalized, {"finish fullerene"})

    def test_punctuation_case_and_spacing_are_normalized(self) -> None:
        variants = (
            "  MAKE SURE WE finish   Fullerene!!!  ",
            "make sure we finish fullerene",
            "Finish, Fullerene.",
        )

        normalized = [normalize_goal_description(value) for value in variants]

        self.assertEqual(normalized[0], "finish fullerene")
        self.assertEqual(normalized[1], "finish fullerene")
        self.assertEqual(normalized[2], "finish fullerene")

    def test_keyword_overlap_can_match_conservative_near_duplicates(self) -> None:
        overlap = goal_keyword_overlap("finish Fullerene", "finishing Fullerene")

        self.assertGreaterEqual(overlap, 0.85)


class SQLiteGoalStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.db_path = self.root / "goals.sqlite3"
        self.store = SQLiteGoalStore(self.db_path)

    def test_initializes_schema(self) -> None:
        self.assertTrue(self.db_path.exists())

        with sqlite3.connect(self.db_path) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

        self.assertIn("goals", table_names)

    def test_add_get_list_and_update(self) -> None:
        active = Goal(
            id="goal-active",
            description="Track my tasks",
            priority=0.4,
            status=GoalStatus.ACTIVE,
            tags=["tasks"],
        )
        paused = Goal(
            id="goal-paused",
            description="Plan vacation",
            priority=0.9,
            status=GoalStatus.PAUSED,
            tags=["travel"],
        )
        paused.updated_at = paused.updated_at - timedelta(days=1)

        self.store.add_goal(active)
        self.store.add_goal(paused)

        fetched = self.store.get_goal("goal-active")
        active_goals = self.store.list_active_goals(limit=5)
        paused_goals = self.store.list_goals(limit=5, status=GoalStatus.PAUSED)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "goal-active")
        self.assertEqual([goal.id for goal in active_goals], ["goal-active"])
        self.assertEqual([goal.id for goal in paused_goals], ["goal-paused"])

        original_updated_at = paused.updated_at
        paused.status = GoalStatus.ACTIVE
        paused.priority = 0.95
        self.store.update_goal(paused)

        updated = self.store.get_goal("goal-paused")
        active_goals = self.store.list_active_goals(limit=5)
        paused_goals = self.store.list_goals(limit=5, status=GoalStatus.PAUSED)

        self.assertIsNotNone(updated)
        self.assertGreater(updated.updated_at, original_updated_at)
        self.assertEqual([goal.id for goal in active_goals], ["goal-paused", "goal-active"])
        self.assertEqual(paused_goals, [])

    def test_pause_resume_complete_helpers_apply_lifecycle_transitions(self) -> None:
        goal = Goal(id="goal-lifecycle", description="Ship v1", priority=0.8, tags=["ship"])
        self.store.add_goal(goal)

        paused = self.store.pause_goal(goal.id, reason="waiting")
        self.assertTrue(paused["ok"])
        self.assertEqual(paused["previous_status"], GoalStatus.ACTIVE.value)
        self.assertEqual(paused["new_status"], GoalStatus.PAUSED.value)
        paused_goal = self.store.get_goal(goal.id)
        self.assertIsNotNone(paused_goal)
        self.assertEqual(paused_goal.status, GoalStatus.PAUSED)
        self.assertEqual(paused_goal.paused_reason, "waiting")

        resumed = self.store.resume_goal(goal.id, reason="unblocked")
        self.assertTrue(resumed["ok"])
        resumed_goal = self.store.get_goal(goal.id)
        self.assertIsNotNone(resumed_goal)
        self.assertEqual(resumed_goal.status, GoalStatus.ACTIVE)

        completed = self.store.complete_goal(
            goal.id, reason="done", evidence_event_id="evt-1"
        )
        self.assertTrue(completed["ok"])
        completed_goal = self.store.get_goal(goal.id)
        self.assertIsNotNone(completed_goal)
        self.assertEqual(completed_goal.status, GoalStatus.COMPLETED)
        self.assertEqual(completed_goal.completion_score, 1.0)
        self.assertIsNotNone(completed_goal.completed_at)
        self.assertIn("evt-1", completed_goal.evidence_event_ids)

    def test_list_active_goals_excludes_inactive_unless_explicitly_requested(self) -> None:
        self.store.add_goal(Goal(id="g-active", description="A", priority=0.5))
        paused = Goal(id="g-paused", description="B", priority=0.5, status=GoalStatus.PAUSED)
        completed = Goal(
            id="g-completed", description="C", priority=0.5, status=GoalStatus.COMPLETED
        )
        self.store.add_goal(paused)
        self.store.add_goal(completed)

        active_only = self.store.list_active_goals(limit=10)
        with_inactive = self.store.list_active_goals(limit=10, include_inactive=True)

        self.assertEqual([goal.id for goal in active_only], ["g-active"])
        self.assertEqual({goal.id for goal in with_inactive}, {"g-active", "g-paused", "g-completed"})


class GoalsFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.store = SQLiteGoalStore(self.root / "goals.sqlite3")
        self.facet = GoalsFacet(self.store, active_limit=5, relevant_limit=3)

    def test_returns_empty_when_no_goals(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="work on my tasks"),
            NexusState(),
        )

        self.assertEqual(result.metadata["relevant_goals"], [])
        self.assertEqual(result.metadata["relevance_score"], 0.0)
        self.assertIn("no active goals", result.summary.lower())

    def test_returns_relevant_goals_when_tags_match(self) -> None:
        self.store.add_goal(
            Goal(
                id="goal-1",
                description="Track my tasks",
                priority=0.4,
                tags=["tasks"],
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="work on my tasks",
                metadata={"tags": ["tasks"]},
            ),
            NexusState(),
        )

        self.assertEqual(len(result.metadata["relevant_goals"]), 1)
        self.assertEqual(result.metadata["relevant_goals"][0]["id"], "goal-1")
        self.assertIn("tasks", result.metadata["relevant_goals"][0]["shared_tags"])
        self.assertGreater(result.metadata["relevance_score"], 0.0)

    def test_scoring_favors_higher_priority_goals(self) -> None:
        self.store.add_goal(
            Goal(
                id="goal-low",
                description="Track my tasks",
                priority=0.2,
                tags=["tasks"],
            )
        )
        self.store.add_goal(
            Goal(
                id="goal-high",
                description="Track my tasks",
                priority=0.9,
                tags=["tasks"],
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="track my tasks",
                metadata={"tags": ["tasks"]},
            ),
            NexusState(),
        )

        relevant_goals = result.metadata["relevant_goals"]
        self.assertEqual(relevant_goals[0]["id"], "goal-high")
        self.assertGreater(relevant_goals[0]["score"], relevant_goals[1]["score"])

    def test_high_salience_related_events_reinforce_active_goals(self) -> None:
        self.store.add_goal(
            Goal(
                id="goal-reinforce",
                description="Track my tasks",
                priority=0.7,
                tags=["tasks"],
            )
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="track tasks now",
            metadata={"tags": ["tasks"], "salience": 0.9},
        )

        result = self.facet.process(event, NexusState())
        updated = self.store.get_goal("goal-reinforce")

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertGreater(updated.reinforcement_score, 0.0)
        self.assertEqual(updated.activation_count, 1)
        self.assertEqual(updated.last_activated_event_id, event.event_id)
        self.assertTrue(result.metadata["goal_reinforcement_updates"])

    def test_goal_ranking_includes_score_breakdown_metadata(self) -> None:
        self.store.add_goal(
            Goal(id="goal-rank", description="Do tasks", priority=0.8, tags=["tasks"])
        )
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="tasks",
                metadata={"tags": ["tasks"], "salience": 0.8},
            ),
            NexusState(),
        )
        ranked = result.metadata["relevant_goals"][0]
        self.assertIn("priority_component", ranked)
        self.assertIn("reinforcement_component", ranked)
        self.assertIn("recency_component", ranked)
        self.assertIn("final_score", ranked)


class GoalsBehaviorIntegrationTests(unittest.TestCase):
    def test_behavior_confidence_increases_when_goal_signal_is_available(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Track my tasks",
                priority=0.9,
                tags=["tasks"],
            )
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="work on my tasks",
            metadata={"tags": ["tasks"]},
        )
        baseline = BehaviorFacet().process(event, NexusState())
        runtime = NexusRuntime(
            facets=[GoalsFacet(goal_store), BehaviorFacet(), EchoFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(event)
        behavior_result = next(
            result for result in record.facet_results if result.facet_name == "behavior"
        )

        self.assertTrue(behavior_result.metadata["goal_signal_available"])
        self.assertGreater(behavior_result.metadata["goal_alignment_score"], 0.0)
        self.assertIn("goal_alignment_signal", behavior_result.metadata["confidence_breakdown"])
        self.assertGreater(
            behavior_result.metadata["confidence"],
            baseline.metadata["confidence"],
        )


class CLIGoalsIntegrationTests(unittest.TestCase):
    def test_cli_with_goals_creates_goal_sqlite_under_state_dir_by_default(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--content",
                    "track my tasks",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue((root / "goals.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_cli_goals_db_flag_overrides_default_path(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        custom_db = root / "custom" / "goals.sqlite3"

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--content",
                    "track my tasks",
                    "--state-dir",
                    str(root),
                    "--goals-db",
                    str(custom_db),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(custom_db.exists())
        self.assertFalse((root / "goals.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_create_goal_metadata_creates_a_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--content",
                    "track my tasks",
                    "--metadata",
                    '{"create_goal": true}',
                    "--state-dir",
                    str(root),
                ]
            )

        store = SQLiteGoalStore(root / "goals.sqlite3")
        goals = store.list_active_goals(limit=5)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].description, "track my tasks")
        self.assertEqual(goals[0].priority, 0.5)

    def test_goal_intent_creates_active_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--content",
                    "Remember that finishing Fullerene is important",
                    "--state-dir",
                    str(root),
                ]
            )

        store = SQLiteGoalStore(root / "goals.sqlite3")
        goals = store.list_active_goals(limit=5)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].description, "finishing Fullerene")
        self.assertEqual(goals[0].priority, 0.8)
        self.assertEqual(goals[0].source, GoalSource.USER)
        self.assertEqual(goals[0].status, GoalStatus.ACTIVE)

    def test_repeated_equivalent_goal_intent_updates_existing_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        for content in (
            "I should remember to finish Fullerene",
            "remember to finish Fullerene",
        ):
            with redirect_stdout(io.StringIO()):
                exit_code = cli_main(
                    [
                        "--goals",
                        "--content",
                        content,
                        "--state-dir",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)

        goals = SQLiteGoalStore(root / "goals.sqlite3").list_active_goals(limit=5)

        self.assertEqual(len(goals), 1)
        self.assertEqual(normalize_goal_description(goals[0].description), "finish fullerene")
        self.assertTrue(goals[0].metadata.get("merged_from_duplicate_intent"))

    def test_goal_intent_survives_across_cli_runs_and_guides_next_focus(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        with redirect_stdout(io.StringIO()):
            first_exit = cli_main(
                [
                    "--full",
                    "--content",
                    "Remember that finishing Fullerene is important",
                    "--state-dir",
                    str(root),
                ]
            )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            second_exit = cli_main(
                [
                    "--full",
                    "--content",
                    "What should I focus on next?",
                    "--state-dir",
                    str(root),
                ]
            )

        store = SQLiteGoalStore(root / "goals.sqlite3")
        goals = store.list_active_goals(limit=5)
        output = stdout.getvalue()

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        self.assertEqual(len(goals), 1)
        self.assertIn("decision: ACT", output)
        self.assertIn("finishing Fullerene", output)

    def test_duplicate_goal_intent_updates_existing_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        with redirect_stdout(io.StringIO()):
            first_exit = cli_main(
                [
                    "--goals",
                    "--content",
                    "I want to finish Fullerene",
                    "--metadata",
                    '{"tags": ["urgent"]}',
                    "--state-dir",
                    str(root),
                ]
            )
        with redirect_stdout(io.StringIO()):
            second_exit = cli_main(
                [
                    "--goals",
                    "--content",
                    "remember to finish Fullerene",
                    "--metadata",
                    '{"tags": ["release"]}',
                    "--state-dir",
                    str(root),
                ]
            )

        goals = SQLiteGoalStore(root / "goals.sqlite3").list_active_goals(limit=5)

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].priority, 0.8)
        self.assertIn("urgent", goals[0].tags)
        self.assertIn("release", goals[0].tags)
        self.assertTrue(goals[0].metadata.get("merged_from_duplicate_intent"))

    def test_ordinary_note_records_without_creating_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--goals",
                    "--content",
                    "This is an ordinary note for later.",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        goals = SQLiteGoalStore(root / "goals.sqlite3").list_active_goals(limit=5)

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["decision"]["action"], "record")
        self.assertEqual(goals, [])


class GoalsRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_goals_behavior_and_echo_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Track my tasks",
                priority=0.9,
                tags=["tasks"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store, active_limit=5, relevant_limit=3),
                BehaviorFacet(),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="work on my tasks",
                metadata={"tags": ["tasks"]},
            )
        )
        behavior_result = next(
            result for result in record.facet_results if result.facet_name == "behavior"
        )
        goals_result = next(
            result for result in record.facet_results if result.facet_name == "goals"
        )

        self.assertEqual(len(record.facet_results), 4)
        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["memory", "goals", "behavior", "echo"],
        )
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertTrue((root / "memory.sqlite3").exists())
        self.assertTrue((root / "goals.sqlite3").exists())
        self.assertGreater(goals_result.metadata["relevance_score"], 0.0)
        self.assertGreater(behavior_result.metadata["goal_alignment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
