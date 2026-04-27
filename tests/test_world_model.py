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
from fullerene.workspace_state import workspace_state_root
from fullerene.facets import (
    BehaviorFacet,
    EchoFacet,
    GoalsFacet,
    MemoryFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore
from fullerene.world_model import (
    Belief,
    BeliefSource,
    BeliefStatus,
    SQLiteWorldModelStore,
)
from fullerene.world_model.models import utcnow


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-world-model-{uuid4().hex}"


class BeliefModelTests(unittest.TestCase):
    def test_belief_round_trips_through_dict(self) -> None:
        created_at = utcnow() - timedelta(hours=2)
        updated_at = utcnow() - timedelta(hours=1)
        belief = Belief(
            id="belief-1",
            claim="SQLite is the canonical memory store.",
            confidence=0.9,
            status=BeliefStatus.STALE,
            tags=["SQLite", "Memory", "sqlite"],
            source=BeliefSource.SYSTEM,
            source_event_id="event-1",
            source_memory_id="memory-1",
            created_at=created_at,
            updated_at=updated_at,
            metadata={"origin": "manual", "score": 3},
        )

        round_tripped = Belief.from_dict(belief.to_dict())

        self.assertEqual(round_tripped, belief)
        self.assertEqual(round_tripped.tags, ["sqlite", "memory"])


class SQLiteWorldModelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.db_path = self.root / "world.sqlite3"
        self.store = SQLiteWorldModelStore(self.db_path)

    def test_initializes_schema(self) -> None:
        self.assertTrue(self.db_path.exists())

        with sqlite3.connect(self.db_path) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

        self.assertIn("beliefs", table_names)

    def test_add_get_list_and_update(self) -> None:
        active = Belief(
            id="belief-active",
            claim="SQLite is canonical memory storage",
            confidence=0.4,
            status=BeliefStatus.ACTIVE,
            tags=["memory", "sqlite"],
        )
        stale = Belief(
            id="belief-stale",
            claim="The old API endpoint is still current",
            confidence=0.9,
            status=BeliefStatus.STALE,
            tags=["api"],
        )
        stale.updated_at = stale.updated_at - timedelta(days=1)

        self.store.add_belief(active)
        self.store.add_belief(stale)

        fetched = self.store.get_belief("belief-active")
        active_beliefs = self.store.list_active_beliefs(limit=5)
        stale_beliefs = self.store.list_beliefs(limit=5, status=BeliefStatus.STALE)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "belief-active")
        self.assertEqual([belief.id for belief in active_beliefs], ["belief-active"])
        self.assertEqual([belief.id for belief in stale_beliefs], ["belief-stale"])

        original_updated_at = stale.updated_at
        stale.status = BeliefStatus.ACTIVE
        stale.confidence = 0.95
        self.store.update_belief(stale)

        updated = self.store.get_belief("belief-stale")
        active_beliefs = self.store.list_active_beliefs(limit=5)
        stale_beliefs = self.store.list_beliefs(limit=5, status=BeliefStatus.STALE)

        self.assertIsNotNone(updated)
        self.assertGreater(updated.updated_at, original_updated_at)
        self.assertEqual(
            [belief.id for belief in active_beliefs],
            ["belief-stale", "belief-active"],
        )
        self.assertEqual(stale_beliefs, [])


class WorldModelFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        self.facet = WorldModelFacet(self.store, active_limit=5, relevant_limit=3)

    def test_returns_empty_when_no_beliefs(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Should we change memory storage?",
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["relevant_beliefs"], [])
        self.assertEqual(result.metadata["relevance_score"], 0.0)
        self.assertIn("no active beliefs", result.summary.lower())

    def test_returns_relevant_beliefs_when_matching(self) -> None:
        self.store.add_belief(
            Belief(
                id="belief-1",
                claim="SQLite is the canonical memory store",
                confidence=0.7,
                tags=["memory", "sqlite"],
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Should we change memory storage in sqlite?",
                metadata={"tags": ["memory"]},
            ),
            NexusState(),
        )

        self.assertEqual(len(result.metadata["relevant_beliefs"]), 1)
        self.assertEqual(result.metadata["relevant_beliefs"][0]["id"], "belief-1")
        self.assertIn("memory", result.metadata["relevant_beliefs"][0]["shared_tags"])
        self.assertIn(
            "sqlite",
            result.metadata["relevant_beliefs"][0]["shared_keywords"],
        )
        self.assertGreater(result.metadata["relevance_score"], 0.0)

    def test_scoring_favors_higher_confidence_beliefs(self) -> None:
        self.store.add_belief(
            Belief(
                id="belief-low",
                claim="SQLite is the canonical memory store",
                confidence=0.2,
                tags=["memory", "sqlite"],
            )
        )
        self.store.add_belief(
            Belief(
                id="belief-high",
                claim="SQLite is the canonical memory store",
                confidence=0.9,
                tags=["memory", "sqlite"],
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Should we change memory storage in sqlite?",
                metadata={"tags": ["memory"]},
            ),
            NexusState(),
        )

        relevant_beliefs = result.metadata["relevant_beliefs"]
        self.assertEqual(relevant_beliefs[0]["id"], "belief-high")
        self.assertGreater(relevant_beliefs[0]["score"], relevant_beliefs[1]["score"])


class WorldBehaviorIntegrationTests(unittest.TestCase):
    def test_behavior_confidence_increases_when_world_signal_is_available(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        world_store = SQLiteWorldModelStore(root / "world.sqlite3")
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="SQLite is the canonical memory store",
                confidence=0.95,
                tags=["memory", "sqlite"],
            )
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="Should we change memory storage in sqlite?",
            metadata={"tags": ["memory"]},
        )
        baseline = BehaviorFacet().process(event, NexusState())
        runtime = NexusRuntime(
            facets=[WorldModelFacet(world_store), BehaviorFacet(), EchoFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(event)
        behavior_result = next(
            result for result in record.facet_results if result.facet_name == "behavior"
        )

        self.assertTrue(behavior_result.metadata["world_signal_available"])
        self.assertGreater(behavior_result.metadata["world_alignment_score"], 0.0)
        self.assertIn(
            "world_alignment_signal",
            behavior_result.metadata["confidence_breakdown"],
        )
        self.assertGreater(
            behavior_result.metadata["confidence"],
            baseline.metadata["confidence"],
        )


class CLIWorldModelIntegrationTests(unittest.TestCase):
    def test_cli_with_world_creates_world_sqlite_under_state_dir_by_default(
        self,
    ) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--world",
                    "--content",
                    "Should we change memory storage?",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue((root / "world.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_cli_world_db_flag_overrides_default_path(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        custom_db = root / "custom" / "world.sqlite3"

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--world",
                    "--content",
                    "Should we change memory storage?",
                    "--state-dir",
                    str(root),
                    "--world-db",
                    str(custom_db),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(custom_db.exists())
        self.assertFalse((root / "world.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_create_belief_metadata_creates_a_belief(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--world",
                    "--content",
                    "SQLite is the canonical memory store",
                    "--metadata",
                    '{"create_belief": true}',
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        store = SQLiteWorldModelStore(root / "world.sqlite3")
        beliefs = store.list_active_beliefs(limit=5)
        world_result = next(
            result
            for result in payload["facet_results"]
            if result["facet_name"] == "world_model"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0].claim, "SQLite is the canonical memory store")
        self.assertEqual(beliefs[0].confidence, 0.7)
        self.assertEqual(beliefs[0].source, BeliefSource.USER)
        self.assertEqual(beliefs[0].source_event_id, payload["event"]["event_id"])
        self.assertEqual(world_result["metadata"]["relevant_beliefs"][0]["id"], beliefs[0].id)


class WorldRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_goals_world_behavior_and_echo_facets(
        self,
    ) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(root / "world.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Keep memory storage reliable",
                priority=0.9,
                tags=["memory"],
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="SQLite is the canonical memory store",
                confidence=0.95,
                tags=["memory", "sqlite"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store, active_limit=5, relevant_limit=3),
                WorldModelFacet(world_store, active_limit=5, relevant_limit=3),
                BehaviorFacet(),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Should we change memory storage in sqlite?",
                metadata={"tags": ["memory"]},
            )
        )
        behavior_result = next(
            result for result in record.facet_results if result.facet_name == "behavior"
        )
        goals_result = next(
            result for result in record.facet_results if result.facet_name == "goals"
        )
        world_result = next(
            result
            for result in record.facet_results
            if result.facet_name == "world_model"
        )

        self.assertEqual(len(record.facet_results), 5)
        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["memory", "goals", "world_model", "behavior", "echo"],
        )
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertTrue((root / "memory.sqlite3").exists())
        self.assertTrue((root / "goals.sqlite3").exists())
        self.assertTrue((root / "world.sqlite3").exists())
        self.assertGreater(goals_result.metadata["relevance_score"], 0.0)
        self.assertGreater(world_result.metadata["relevance_score"], 0.0)
        self.assertGreater(behavior_result.metadata["goal_alignment_score"], 0.0)
        self.assertGreater(behavior_result.metadata["world_alignment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
