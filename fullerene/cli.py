"""Minimal CLI for exercising the Nexus runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fullerene.facets import (
    BehaviorFacet,
    EchoFacet,
    GoalsFacet,
    MemoryFacet,
    PolicyFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, GoalSource, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore, infer_tags, merge_tags, normalize_tags
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.policy import (
    PolicyRule,
    PolicySource,
    SQLitePolicyStore,
    coerce_policy_rule_type,
    coerce_policy_source,
    coerce_policy_target_type,
)
from fullerene.workspace_state import DEFAULT_STATE_DIR
from fullerene.state import FileStateStore
from fullerene.world_model import Belief, BeliefSource, SQLiteWorldModelStore


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
        "--world",
        action="store_true",
        help="Enable the SQLite-backed WorldModelFacet for this run.",
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Enable the SQLite-backed PolicyFacet for this run.",
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
        default=DEFAULT_STATE_DIR,
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
    parser.add_argument(
        "--world-db",
        default=None,
        help=(
            "SQLite path used by --world runs. "
            "Defaults to <state-dir>/world.sqlite3 when omitted."
        ),
    )
    parser.add_argument(
        "--policy-db",
        default=None,
        help=(
            "SQLite path used by --policy runs. "
            "Defaults to <state-dir>/policy.sqlite3 when omitted."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata = _parse_metadata(parser, args.metadata)

    state_dir = Path(args.state_dir)
    store = FileStateStore(state_dir)
    event = Event(
        event_type=EventType(args.event_type),
        content=args.content,
        metadata=metadata,
    )
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
    if args.world:
        world_db_path = (
            Path(args.world_db) if args.world_db else state_dir / "world.sqlite3"
        )
        world_store = SQLiteWorldModelStore(world_db_path)
        _create_belief_from_metadata(world_store, event=event)
        facets.append(WorldModelFacet(world_store))
    if args.behavior:
        facets.append(BehaviorFacet())
    if args.policy:
        policy_db_path = (
            Path(args.policy_db) if args.policy_db else state_dir / "policy.sqlite3"
        )
        policy_store = SQLitePolicyStore(policy_db_path)
        try:
            _create_policy_from_metadata(
                policy_store,
                content=args.content,
                metadata=metadata,
            )
        except ValueError as exc:
            parser.error(str(exc))
        facets.append(PolicyFacet(policy_store, state_dir=state_dir))
    facets.append(EchoFacet())

    runtime = NexusRuntime(facets=facets, store=store)
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


def _create_belief_from_metadata(
    store: SQLiteWorldModelStore,
    *,
    event: Event,
) -> Belief | None:
    if not _metadata_flag(event.metadata, "create_belief"):
        return None
    if not event.content.strip():
        return None

    explicit_tags: list[str] = []
    raw_tags = event.metadata.get("tags", [])
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        explicit_tags = normalize_tags(raw_tags)

    belief = Belief(
        claim=event.content,
        confidence=0.7,
        tags=merge_tags(explicit_tags, infer_tags(event.content)),
        source=_belief_source_for_event(event.event_type),
        source_event_id=event.event_id,
        metadata={
            key: value
            for key, value in event.metadata.items()
            if key != "create_belief"
        },
    )
    store.add_belief(belief)
    return belief


def _belief_source_for_event(event_type: EventType) -> BeliefSource:
    if event_type == EventType.USER_MESSAGE:
        return BeliefSource.USER
    return BeliefSource.SYSTEM


def _create_policy_from_metadata(
    store: SQLitePolicyStore,
    *,
    content: str,
    metadata: dict[str, Any],
) -> PolicyRule | None:
    if not _metadata_flag(metadata, "create_policy"):
        return None

    rule_type = coerce_policy_rule_type(metadata.get("rule_type"))
    if rule_type is None:
        raise ValueError(
            "create_policy metadata requires a valid rule_type "
            "(allow, deny, require_approval, prefer)."
        )

    target_type = coerce_policy_target_type(metadata.get("target_type"))
    if target_type is None:
        raise ValueError(
            "create_policy metadata requires a valid target_type "
            "(internal_state, file_write, file_delete, shell, network, "
            "message, git, tool, decision, tag, general)."
        )

    conditions = metadata.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError("create_policy metadata field 'conditions' must be a JSON object.")

    priority = metadata.get("priority", 0.0)
    if not isinstance(priority, (int, float)):
        raise ValueError("create_policy metadata field 'priority' must be numeric.")

    rule = PolicyRule(
        name=_policy_name_from_metadata(content=content, metadata=metadata),
        description=_coerce_metadata_string(metadata, "description") or content.strip(),
        rule_type=rule_type,
        target_type=target_type,
        target=_coerce_metadata_string(metadata, "target") or "*",
        conditions=conditions,
        priority=float(priority),
        enabled=_metadata_enabled(metadata),
        source=coerce_policy_source(metadata.get("source")) or PolicySource.USER,
        metadata=_policy_metadata_payload(metadata),
    )
    store.add_policy(rule)
    return rule


def _policy_name_from_metadata(
    *,
    content: str,
    metadata: dict[str, Any],
) -> str:
    for key in ("policy_name", "name"):
        value = _coerce_metadata_string(metadata, key)
        if value:
            return value
    if content.strip():
        return content.strip()
    return "policy-rule"


def _coerce_metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    raw_value = metadata.get(key)
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip()
    return cleaned or None


def _metadata_enabled(metadata: dict[str, Any]) -> bool:
    if "enabled" not in metadata:
        return True
    return _metadata_flag(metadata, "enabled")


def _policy_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw_policy_metadata = metadata.get("policy_metadata")
    if isinstance(raw_policy_metadata, dict):
        payload.update(raw_policy_metadata)

    control_keys = {
        "create_policy",
        "rule_type",
        "target_type",
        "target",
        "priority",
        "enabled",
        "source",
        "conditions",
        "policy_name",
        "name",
        "description",
        "policy_metadata",
    }
    for key, value in metadata.items():
        if key in control_keys:
            continue
        payload[key] = value
    return payload
