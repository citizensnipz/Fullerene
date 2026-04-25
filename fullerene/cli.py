"""Minimal CLI for exercising the Nexus runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fullerene.facets import BehaviorFacet, EchoFacet, MemoryFacet
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.state import FileStateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a single event through the Fullerene Nexus runtime."
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable the SQLite-backed MemoryFacet for this run.",
    )
    parser.add_argument(
        "--behavior",
        action="store_true",
        help="Enable the deterministic BehaviorFacet for this run.",
    )
    parser.add_argument(
        "--event-type",
        choices=[event_type.value for event_type in EventType],
        default=EventType.USER_MESSAGE.value,
        help="The kind of event to process.",
    )
    parser.add_argument(
        "--content",
        default="",
        help="Event content for user message or system note events.",
    )
    parser.add_argument(
        "--state-dir",
        default=".fullerene-state",
        help="Local directory for the runtime snapshot and append-only log.",
    )
    parser.add_argument(
        "--memory-db",
        default=None,
        help=(
            "SQLite path used by --memory runs. "
            "Defaults to <state-dir>/memory.sqlite3 when omitted."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir)
    store = FileStateStore(state_dir)
    facets = [EchoFacet()]
    if args.behavior:
        facets = [BehaviorFacet(), *facets]
    if args.memory:
        memory_db_path = Path(args.memory_db) if args.memory_db else state_dir / "memory.sqlite3"
        memory_store = SQLiteMemoryStore(memory_db_path)
        facets = [MemoryFacet(memory_store), *facets]

    runtime = NexusRuntime(facets=facets, store=store)
    event = Event(event_type=EventType(args.event_type), content=args.content)
    record = runtime.process_event(event)

    print(json.dumps(record.to_dict(), indent=2))
    return 0
