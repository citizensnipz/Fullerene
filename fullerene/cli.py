"""Minimal CLI for exercising the Nexus runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fullerene.facets import BehaviorFacet, EchoFacet, GoalsFacet, MemoryFacet
from fullerene.goals import Goal, GoalSource, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore, infer_tags, merge_tags, normalize_tags
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
        "--goals",
        action="store_true",
        help="Enable the SQLite-backed GoalsFacet for this run.",
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
        "--metadata",
        default=None,
        help="Optional JSON object attached to the event metadata.",
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
    parser.add_argument(
        "--goals-db",
        default=None,
        help=(
            "SQLite path used by --goals runs. "
            "Defaults to <state-dir>/goals.sqlite3 when omitted."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata = _parse_metadata(parser, args.metadata)

    state_dir = Path(args.state_dir)
    store = FileStateStore(state_dir)
    facets = []
    if args.memory:
        memory_db_path = (
            Path(args.memory_db) if args.memory_db else state_dir / "memory.sqlite3"
        )
        memory_store = SQLiteMemoryStore(memory_db_path)
        facets.append(MemoryFacet(memory_store))
    if args.goals:
        goals_db_path = (
            Path(args.goals_db) if args.goals_db else state_dir / "goals.sqlite3"
        )
        goal_store = SQLiteGoalStore(goals_db_path)
        _create_goal_from_metadata(goal_store, content=args.content, metadata=metadata)
        facets.append(GoalsFacet(goal_store))
    if args.behavior:
        facets.append(BehaviorFacet())
    facets.append(EchoFacet())

    runtime = NexusRuntime(facets=facets, store=store)
    event = Event(
        event_type=EventType(args.event_type),
        content=args.content,
        metadata=metadata,
    )
    record = runtime.process_event(event)

    print(json.dumps(record.to_dict(), indent=2))
    return 0


def _parse_metadata(
    parser: argparse.ArgumentParser,
    raw_metadata: str | None,
) -> dict[str, Any]:
    if raw_metadata is None:
        return {}

    try:
        payload = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        parser.error(f"--metadata must be valid JSON: {exc}")

    if not isinstance(payload, dict):
        parser.error("--metadata must decode to a JSON object.")
    return payload


def _create_goal_from_metadata(
    store: SQLiteGoalStore,
    *,
    content: str,
    metadata: dict[str, Any],
) -> Goal | None:
    if not _metadata_flag(metadata, "create_goal"):
        return None
    if not content.strip():
        return None

    explicit_tags: list[str] = []
    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        explicit_tags = normalize_tags(raw_tags)

    goal = Goal(
        description=content,
        priority=0.5,
        tags=merge_tags(explicit_tags, infer_tags(content)),
        source=GoalSource.USER,
        metadata={
            key: value
            for key, value in metadata.items()
            if key != "create_goal"
        },
    )
    store.add_goal(goal)
    return goal


def _metadata_flag(metadata: dict[str, Any], key: str) -> bool:
    raw_value = metadata.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return False
