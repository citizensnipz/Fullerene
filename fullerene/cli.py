"""Minimal CLI for exercising the Nexus runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fullerene.facets import EchoFacet
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.state import FileStateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a single event through the Fullerene Nexus runtime."
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    store = FileStateStore(Path(args.state_dir))
    runtime = NexusRuntime(facets=[EchoFacet()], store=store)
    event = Event(event_type=EventType(args.event_type), content=args.content)
    record = runtime.process_event(event)

    print(json.dumps(record.to_dict(), indent=2))
    return 0
