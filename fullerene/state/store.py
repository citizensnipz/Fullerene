"""State store implementations for the Nexus runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from fullerene.nexus.models import NexusRecord, NexusState


class StateStore(Protocol):
    """Persistence contract for runtime snapshots and logs."""

    def load_state(self) -> NexusState | None:
        """Load a previously persisted state snapshot if one exists."""

    def save_state(self, state: NexusState) -> None:
        """Persist the current state snapshot."""

    def append_record(self, record: NexusRecord) -> None:
        """Append a processed event record to durable storage."""


class InMemoryStateStore:
    """Simple in-memory store for tests or embedded callers."""

    def __init__(self) -> None:
        self.state: NexusState | None = None
        self.records: list[NexusRecord] = []

    def load_state(self) -> NexusState | None:
        return self.state

    def save_state(self, state: NexusState) -> None:
        self.state = state

    def append_record(self, record: NexusRecord) -> None:
        self.records.append(record)


class FileStateStore:
    """File-backed store that keeps writes inside an explicit state directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.log_path = self.root / "runtime-log.jsonl"

    def load_state(self) -> NexusState | None:
        if not self.state_path.exists():
            return None
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return NexusState.from_dict(payload)

    def save_state(self, state: NexusState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def append_record(self, record: NexusRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
