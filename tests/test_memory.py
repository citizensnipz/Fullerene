from __future__ import annotations

import sqlite3
import shutil
import unittest
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fullerene.facets import EchoFacet, MemoryFacet
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.memory.models import utcnow
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore


def make_tempdir_path() -> Path:
    return Path.cwd() / f".test-memory-{uuid4().hex}"


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


if __name__ == "__main__":
    unittest.main()
