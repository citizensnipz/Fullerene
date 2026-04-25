from __future__ import annotations

import io
import sqlite3
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.facets import EchoFacet, MemoryFacet
from fullerene.memory import (
    MemoryRecord,
    MemoryType,
    SQLiteMemoryStore,
    compute_salience,
    explain_salience,
    infer_tags,
    merge_tags,
)
from fullerene.memory.models import utcnow
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore


def make_tempdir_path() -> Path:
    return Path.cwd() / "mem_storage" / f".test-memory-{uuid4().hex}"


class TrackingMemoryStore:
    def __init__(self) -> None:
        self.added: list[MemoryRecord] = []
        self.list_recent_limits: list[int] = []
        self.retrieve_limits: list[int] = []

    def add_memory(self, record: MemoryRecord) -> None:
        self.added.append(record)

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        for record in self.added:
            if record.id == memory_id:
                return record
        return None

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        self.list_recent_limits.append(limit)
        records = list(reversed(self.added))
        if memory_type is not None:
            records = [
                record for record in records if record.memory_type == memory_type
            ]
        return records[:limit]

    def search_keyword(self, query: str, limit: int) -> list[MemoryRecord]:
        del query, limit
        return []

    def retrieve_relevant(self, event: Event, limit: int) -> list[MemoryRecord]:
        del event
        self.retrieve_limits.append(limit)
        return list(reversed(self.added))[:limit]


class MemoryRecordTests(unittest.TestCase):
    def test_round_trips_through_dict(self) -> None:
        created_at = utcnow() - timedelta(hours=2)
        record = MemoryRecord(
            id="memory-1",
            created_at=created_at,
            memory_type=MemoryType.SEMANTIC,
            content="The repo uses SQLite for canonical memory.",
            source_event_id="event-1",
            salience=0.8,
            confidence=0.9,
            tags=["Memory", "sqlite", "memory"],
            metadata={"kind": "decision", "score": 3},
        )

        round_tripped = MemoryRecord.from_dict(record.to_dict())

        self.assertEqual(round_tripped, record)
        self.assertEqual(round_tripped.tags, ["memory", "sqlite"])


class SQLiteMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.db_path = self.root / "memory.sqlite3"
        self.store = SQLiteMemoryStore(self.db_path)

    def test_initializes_schema(self) -> None:
        self.assertTrue(self.db_path.exists())

        with sqlite3.connect(self.db_path) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

        self.assertIn("memories", table_names)

    def test_add_get_and_list_recent(self) -> None:
        older = MemoryRecord(
            id="older",
            created_at=utcnow() - timedelta(days=2),
            memory_type=MemoryType.EPISODIC,
            content="Older episodic memory",
            salience=0.4,
            confidence=1.0,
            tags=["history"],
        )
        newer = MemoryRecord(
            id="newer",
            created_at=utcnow(),
            memory_type=MemoryType.SEMANTIC,
            content="Newer semantic memory",
            salience=0.9,
            confidence=0.8,
            tags=["fact"],
        )

        self.store.add_memory(older)
        self.store.add_memory(newer)

        fetched = self.store.get_memory("newer")
        recent = self.store.list_recent(limit=2)
        episodic_only = self.store.list_recent(limit=2, memory_type=MemoryType.EPISODIC)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "newer")
        self.assertEqual([record.id for record in recent], ["newer", "older"])
        self.assertEqual([record.id for record in episodic_only], ["older"])

    def test_search_keyword_and_retrieve_relevant(self) -> None:
        matching = MemoryRecord(
            id="matching",
            created_at=utcnow() - timedelta(hours=1),
            memory_type=MemoryType.EPISODIC,
            content="Remember that hello memory should stay persistent in sqlite",
            salience=0.9,
            confidence=1.0,
            tags=["greeting", "memory"],
        )
        unrelated = MemoryRecord(
            id="unrelated",
            created_at=utcnow(),
            memory_type=MemoryType.EPISODIC,
            content="The weather is sunny today",
            salience=0.3,
            confidence=1.0,
            tags=["weather"],
        )

        self.store.add_memory(unrelated)
        self.store.add_memory(matching)

        keyword_results = self.store.search_keyword("hello memory", limit=5)
        relevant = self.store.retrieve_relevant(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="hello memory in sqlite",
                metadata={"tags": ["memory"]},
            ),
            limit=2,
        )

        self.assertEqual(keyword_results[0].id, "matching")
        self.assertEqual(relevant[0].id, "matching")


class MemoryFacetTests(unittest.TestCase):
    def test_stores_user_message_as_episodic_memory(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        facet = MemoryFacet(store, retrieve_limit=2, working_limit=2)
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="hello memory",
            metadata={"tags": ["Greeting", "Memory"]},
        )

        result = facet.process(event, NexusState())
        memories = store.list_recent(limit=5, memory_type=MemoryType.EPISODIC)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source_event_id, event.event_id)
        self.assertEqual(memories[0].memory_type, MemoryType.EPISODIC)
        self.assertEqual(memories[0].tags, ["greeting", "memory"])
        self.assertIn("stored episodic memory", result.summary)
        self.assertEqual(result.metadata["stored_memory"]["id"], memories[0].id)

    def test_does_not_load_all_memories(self) -> None:
        store = TrackingMemoryStore()
        facet = MemoryFacet(store, retrieve_limit=3, working_limit=2)

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="bounded retrieval"),
            NexusState(),
        )

        self.assertEqual(store.list_recent_limits, [2])
        self.assertEqual(store.retrieve_limits, [4])
        self.assertEqual(len(result.metadata["working_memories"]), 1)
        self.assertEqual(len(result.metadata["relevant_memories"]), 0)


class MemoryRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_and_echo_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="hello memory runtime",
                metadata={"tags": ["memory"]},
            )
        )

        self.assertEqual(len(record.facet_results), 2)
        self.assertEqual(record.facet_results[0].facet_name, "memory")
        self.assertEqual(record.facet_results[1].facet_name, "echo")
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertTrue((root / "memory.sqlite3").exists())


class CLIMemoryIntegrationTests(unittest.TestCase):
    def test_cli_with_memory_creates_memory_sqlite_under_state_dir_by_default(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--content",
                    "hello memory",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue((root / "memory.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())

    def test_cli_memory_db_flag_overrides_default_path(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        custom_db = root / "custom" / "memory.sqlite3"

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--content",
                    "hello memory",
                    "--state-dir",
                    str(root),
                    "--memory-db",
                    str(custom_db),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(custom_db.exists())
        self.assertFalse((root / "memory.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())

    def test_cli_without_memory_does_not_create_memory_sqlite(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(["--content", "hello echo", "--state-dir", str(root)])

        self.assertEqual(exit_code, 0)
        self.assertFalse((root / "memory.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())


class TagInferenceTests(unittest.TestCase):
    def test_dont_ever_skip_my_boss_emails_extracts_requested_tags(self) -> None:
        # Smart-quote apostrophe should also work after normalization.
        smart = infer_tags("don\u2019t ever skip my boss emails")
        straight = infer_tags("don't ever skip my boss emails")

        self.assertIn("communication", smart)
        self.assertIn("authority", smart)
        self.assertIn("hard-rule-candidate", smart)
        self.assertIn("correction", smart)
        self.assertEqual(smart, straight)

    def test_extracts_communication_authority_and_urgent_tags(self) -> None:
        tags = infer_tags("Send the email to the boss now")

        self.assertIn("communication", tags)
        self.assertIn("authority", tags)
        self.assertIn("urgent", tags)

    def test_extracts_bug_and_verification_tags(self) -> None:
        tags = infer_tags("a failing test is broken; please verify")

        self.assertIn("bug", tags)
        self.assertIn("verification", tags)

    def test_extracts_memory_goals_and_policy_tags(self) -> None:
        memory_tags = infer_tags("remember the goal and the policy")

        self.assertIn("memory", memory_tags)
        self.assertIn("goals", memory_tags)
        self.assertIn("policy", memory_tags)

    def test_token_boundaries_avoid_false_positives(self) -> None:
        # "leader" must not trigger "lead"; "embossed" must not trigger "boss".
        self.assertEqual(infer_tags("leader embossed nowadays"), [])

    def test_ignores_empty_content(self) -> None:
        self.assertEqual(infer_tags(""), [])

    def test_merge_tags_normalizes_and_dedupes(self) -> None:
        merged = merge_tags(["Memory", " greeting "], ["memory", "communication"])

        self.assertEqual(merged, ["memory", "greeting", "communication"])


class SalienceScoringTests(unittest.TestCase):
    def test_base_salience_for_neutral_content(self) -> None:
        score = compute_salience(
            content="weather looks fine",
            tags=[],
            is_user_message=False,
        )
        self.assertAlmostEqual(score, 0.3)

    def test_user_message_increases_salience(self) -> None:
        baseline = compute_salience(
            content="finish the task",
            tags=[],
            is_user_message=False,
        )
        user_message = compute_salience(
            content="finish the task",
            tags=[],
            is_user_message=True,
        )
        self.assertGreater(user_message, baseline)
        self.assertAlmostEqual(user_message - baseline, 0.2)

    def test_correction_language_increases_salience(self) -> None:
        baseline = compute_salience(
            content="this output looks fine",
            tags=[],
            is_user_message=True,
        )
        corrected = compute_salience(
            content="this output is wrong; do it the other way instead",
            tags=[],
            is_user_message=True,
        )
        self.assertGreater(corrected, baseline)
        self.assertAlmostEqual(corrected - baseline, 0.2)

    def test_hard_rule_urgent_authority_and_communication_tags_increase_salience(
        self,
    ) -> None:
        baseline = compute_salience(
            content="generic note",
            tags=[],
            is_user_message=False,
        )
        hard_rule = compute_salience(
            content="generic note",
            tags=["hard-rule-candidate"],
            is_user_message=False,
        )
        urgent = compute_salience(
            content="generic note",
            tags=["urgent"],
            is_user_message=False,
        )
        authority = compute_salience(
            content="generic note",
            tags=["authority"],
            is_user_message=False,
        )
        communication = compute_salience(
            content="generic note",
            tags=["communication"],
            is_user_message=False,
        )
        self.assertAlmostEqual(hard_rule - baseline, 0.2)
        self.assertAlmostEqual(urgent - baseline, 0.1)
        self.assertAlmostEqual(authority - baseline, 0.1)
        self.assertAlmostEqual(communication - baseline, 0.05)

    def test_salience_is_clamped_to_unit_interval(self) -> None:
        # Stack every signal so the raw total exceeds 1.0; clamp must hold.
        loud = compute_salience(
            content="don't ever fail my boss email now",
            tags=[],
            is_user_message=True,
        )
        self.assertEqual(loud, 1.0)

        # Low base also clamps at 0.
        quiet = compute_salience(
            content="",
            tags=[],
            is_user_message=False,
            base=-1.0,
        )
        self.assertEqual(quiet, 0.0)

    def test_explain_salience_reports_components(self) -> None:
        breakdown = explain_salience(
            content="don't ever skip my boss emails now",
            tags=[],
            is_user_message=True,
        )
        self.assertEqual(breakdown["base"], 0.3)
        self.assertEqual(breakdown["user_message"], 0.2)
        self.assertEqual(breakdown["hard_rule_candidate_tag"], 0.2)
        self.assertEqual(breakdown["urgent_tag"], 0.1)
        self.assertEqual(breakdown["correction_tag"], 0.2)
        self.assertEqual(breakdown["authority_tag"], 0.1)
        self.assertEqual(breakdown["communication_tag"], 0.05)
        self.assertEqual(breakdown["total"], 1.0)


class MemoryFacetInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        self.facet = MemoryFacet(self.store, retrieve_limit=2, working_limit=2)

    def test_facet_stores_inferred_tags_alongside_metadata_tags(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="don't ever skip my boss emails",
            metadata={"tags": ["personal"]},
        )

        self.facet.process(event, NexusState())
        memories = self.store.list_recent(limit=1)

        self.assertEqual(len(memories), 1)
        stored = memories[0]
        self.assertIn("personal", stored.tags)
        self.assertIn("communication", stored.tags)
        self.assertIn("authority", stored.tags)
        self.assertIn("hard-rule-candidate", stored.tags)
        self.assertIn("correction", stored.tags)
        # Explicit metadata tag retains priority (appears before inferred ones).
        self.assertEqual(stored.tags[0], "personal")
        self.assertEqual(stored.metadata["metadata_tags"], ["personal"])
        self.assertIn("communication", stored.metadata["inferred_tags"])

    def test_facet_stores_computed_salience(self) -> None:
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="don't ever skip my boss emails",
        )

        self.facet.process(event, NexusState())
        memories = self.store.list_recent(limit=1)

        self.assertEqual(len(memories), 1)
        stored = memories[0]
        self.assertAlmostEqual(stored.salience, 1.0, places=6)
        self.assertIn("salience_breakdown", stored.metadata)
        self.assertEqual(stored.metadata["salience_breakdown"]["base"], 0.3)
        self.assertEqual(stored.metadata["salience_breakdown"]["user_message"], 0.2)


class MemoryRetrievalTagPreferenceTests(unittest.TestCase):
    def test_retrieve_relevant_favors_matching_tags_and_salience(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")

        # High-salience, tag-matching record but older.
        high = MemoryRecord(
            id="high",
            created_at=utcnow() - timedelta(days=2),
            memory_type=MemoryType.EPISODIC,
            content="don't ever skip my boss emails",
            salience=0.9,
            confidence=1.0,
            tags=["communication", "authority", "hard-rule-candidate", "correction"],
        )
        # Newer and partly matching, but materially less salient.
        medium = MemoryRecord(
            id="medium",
            created_at=utcnow(),
            memory_type=MemoryType.EPISODIC,
            content="boss emails are archived for later review",
            salience=0.2,
            confidence=1.0,
            tags=["communication", "authority"],
        )
        low = MemoryRecord(
            id="low",
            created_at=utcnow(),
            memory_type=MemoryType.EPISODIC,
            content="random unrelated note about lunch",
            salience=0.1,
            confidence=1.0,
            tags=["food"],
        )
        store.add_memory(high)
        store.add_memory(medium)
        store.add_memory(low)

        relevant = store.retrieve_relevant(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="boss email follow up",
            ),
            limit=3,
        )

        self.assertEqual([memory.id for memory in relevant][:1], ["high"])


if __name__ == "__main__":
    unittest.main()
