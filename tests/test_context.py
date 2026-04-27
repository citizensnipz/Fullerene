from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.context import (
    STATIC_RECENT_EPISODIC_V0,
    ContextItem,
    ContextItemType,
    ContextWindow,
    StaticContextAssembler,
)
from fullerene.facets import ContextFacet, EchoFacet, MemoryFacet
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.memory.models import utcnow
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-context-{uuid4().hex}"


class TrackingMemoryStore:
    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self.records = list(records or [])
        self.list_recent_calls: list[tuple[int, MemoryType | None]] = []

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        self.list_recent_calls.append((limit, memory_type))
        records = sorted(
            self.records,
            key=lambda record: (record.created_at, record.id),
            reverse=True,
        )
        if memory_type is not None:
            records = [
                record for record in records if record.memory_type == memory_type
            ]
        return records[:limit]


class ContextModelTests(unittest.TestCase):
    def test_context_item_round_trips_through_dict(self) -> None:
        created_at = utcnow() - timedelta(minutes=30)
        item = ContextItem(
            id="context-item-1",
            item_type=ContextItemType.MEMORY,
            content="Remember the latest user instruction.",
            source_id="event-1",
            created_at=created_at,
            metadata={"tags": ["memory"], "salience": 0.8},
        )

        round_tripped = ContextItem.from_dict(item.to_dict())

        self.assertEqual(round_tripped, item)

    def test_context_window_round_trips_through_dict(self) -> None:
        window = ContextWindow(
            id="context-window-1",
            created_at=utcnow(),
            items=[
                ContextItem(
                    id="item-1",
                    item_type=ContextItemType.MEMORY,
                    content="Latest episodic note",
                )
            ],
            max_items=5,
            strategy=STATIC_RECENT_EPISODIC_V0,
            metadata={"source_types": ["episodic_memory"]},
        )

        round_tripped = ContextWindow.from_dict(window.to_dict())

        self.assertEqual(round_tripped, window)


class StaticContextAssemblerTests(unittest.TestCase):
    def test_returns_empty_context_when_no_memory_records_exist(self) -> None:
        store = TrackingMemoryStore()
        assembler = StaticContextAssembler(store, max_items=5)

        window = assembler.assemble()

        self.assertEqual(window.items, [])
        self.assertEqual(window.max_items, 5)
        self.assertEqual(window.strategy, STATIC_RECENT_EPISODIC_V0)
        self.assertEqual(store.list_recent_calls, [(5, MemoryType.EPISODIC)])

    def test_includes_only_recent_episodic_memories(self) -> None:
        records = [
            MemoryRecord(
                id="episodic-old",
                created_at=utcnow() - timedelta(days=2),
                memory_type=MemoryType.EPISODIC,
                content="older episodic memory",
                tags=["memory"],
            ),
            MemoryRecord(
                id="semantic-new",
                created_at=utcnow() - timedelta(hours=1),
                memory_type=MemoryType.SEMANTIC,
                content="semantic memory that must be ignored",
                tags=["fact"],
            ),
            MemoryRecord(
                id="episodic-new",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="newer episodic memory",
                tags=["memory"],
            ),
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=5)

        window = assembler.assemble()

        self.assertEqual([item.id for item in window.items], ["episodic-new", "episodic-old"])
        self.assertEqual(
            [item.item_type for item in window.items],
            [ContextItemType.MEMORY, ContextItemType.MEMORY],
        )

    def test_respects_max_items_window_size(self) -> None:
        records = [
            MemoryRecord(
                id=f"episodic-{index}",
                created_at=utcnow() - timedelta(minutes=index),
                memory_type=MemoryType.EPISODIC,
                content=f"episodic memory {index}",
                tags=["memory"],
            )
            for index in range(4)
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=2)

        window = assembler.assemble()

        self.assertEqual(len(window.items), 2)
        self.assertEqual(store.list_recent_calls, [(2, MemoryType.EPISODIC)])

    def test_does_not_load_all_memory(self) -> None:
        records = [
            MemoryRecord(
                id="episodic-1",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="episodic memory 1",
            )
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=3)

        assembler.assemble()

        self.assertEqual(store.list_recent_calls, [(3, MemoryType.EPISODIC)])


class ContextFacetTests(unittest.TestCase):
    def test_returns_empty_context_without_memory_store(self) -> None:
        facet = ContextFacet(None, window_size=4)

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="show context"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision.value, "wait")
        self.assertEqual(result.metadata["item_count"], 0)
        self.assertEqual(result.metadata["strategy"], STATIC_RECENT_EPISODIC_V0)
        self.assertEqual(result.metadata["max_items"], 4)
        self.assertEqual(result.metadata["source_types"], [])
        self.assertEqual(result.metadata["context_window"]["items"], [])
        self.assertIn("no memory store", result.summary.lower())

    def test_returns_recent_episodic_items_with_memory_store(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        store.add_memory(
            MemoryRecord(
                id="episodic-1",
                created_at=utcnow() - timedelta(hours=2),
                memory_type=MemoryType.EPISODIC,
                content="first episodic memory",
                source_event_id="event-1",
                tags=["memory"],
            )
        )
        store.add_memory(
            MemoryRecord(
                id="semantic-1",
                created_at=utcnow() - timedelta(hours=1),
                memory_type=MemoryType.SEMANTIC,
                content="semantic memory",
                source_event_id="event-2",
                tags=["fact"],
            )
        )
        store.add_memory(
            MemoryRecord(
                id="episodic-2",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="second episodic memory",
                source_event_id="event-3",
                tags=["memory"],
            )
        )
        facet = ContextFacet(store, window_size=5)

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="show context"),
            NexusState(),
        )

        items = result.metadata["context_window"]["items"]
        self.assertEqual([item["id"] for item in items], ["episodic-2", "episodic-1"])
        self.assertEqual(result.metadata["item_count"], 2)
        self.assertEqual(result.metadata["source_types"], ["episodic_memory"])
        self.assertEqual(result.metadata["strategy"], STATIC_RECENT_EPISODIC_V0)


class ContextRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_context_and_echo_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="prior-memory",
                created_at=utcnow() - timedelta(minutes=5),
                memory_type=MemoryType.EPISODIC,
                content="prior episodic memory",
                source_event_id="event-prior",
                tags=["memory"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(memory_store, window_size=2),
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="current message")
        )

        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["context", "memory", "echo"],
        )
        context_result = record.facet_results[0]
        self.assertEqual(context_result.metadata["item_count"], 1)
        self.assertEqual(
            context_result.metadata["context_window"]["items"][0]["content"],
            "prior episodic memory",
        )
        self.assertTrue((root / "memory.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())


class CLIContextIntegrationTests(unittest.TestCase):
    def test_cli_with_context_runs_without_error(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--context",
                    "--content",
                    "show context",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        context_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "context"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(context_result["metadata"]["strategy"], STATIC_RECENT_EPISODIC_V0)
        self.assertTrue((root / "memory.sqlite3").exists())

    def test_cli_memory_and_context_can_load_recent_episodic_records_on_later_run(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        for content in ("first context memory", "second context memory"):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--memory",
                        "--context",
                        "--content",
                        content,
                        "--state-dir",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)

        final_stdout = io.StringIO()
        with redirect_stdout(final_stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--context",
                    "--context-window-size",
                    "2",
                    "--content",
                    "show recent context",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(final_stdout.getvalue())
        context_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "context"
        )
        context_items = context_result["metadata"]["context_window"]["items"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(context_result["metadata"]["item_count"], 2)
        self.assertEqual(
            [item["content"] for item in context_items],
            ["second context memory", "first context memory"],
        )
        self.assertEqual(context_result["metadata"]["max_items"], 2)
        self.assertEqual(context_result["metadata"]["source_types"], ["episodic_memory"])


if __name__ == "__main__":
    unittest.main()
