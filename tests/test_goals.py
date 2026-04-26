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
from fullerene.goals import Goal, GoalSource, GoalStatus, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore


def make_tempdir_path() -> Path:
    return Path.cwd() / "goals_storage" / f".test-goals-{uuid4().hex}"


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
    def test_cli_with_goals_creates_goal_sqlite_under_state_dir(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
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

    def test_create_goal_metadata_creates_a_goal(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
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
