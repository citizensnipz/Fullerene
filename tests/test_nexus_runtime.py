from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

from fullerene.facets import EchoFacet
from fullerene.nexus import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusRuntime,
    NexusState,
)
from fullerene.state import FileStateStore, InMemoryStateStore


class AskFacet:
    name = "asker"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Escalated the event to a follow-up question.",
            proposed_decision=DecisionAction.ASK,
        )


class SilentFacet:
    name = "silent"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Observed the event without proposing anything.",
        )


class NexusRuntimeTests(unittest.TestCase):
    def test_process_event_persists_state_and_log(self) -> None:
        store_root = Path.cwd() / f".test-nexus-store-{uuid4().hex}"
        store = FileStateStore(store_root)
        runtime = NexusRuntime(facets=[EchoFacet()], store=store)

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="hello nexus")
        )

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertTrue(store.state_path.exists())
        self.assertTrue(store.log_path.exists())

        state_payload = json.loads(store.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state_payload["event_count"], 1)
        self.assertEqual(
            state_payload["facet_state"]["echo"]["last_user_message"],
            "hello nexus",
        )

        log_lines = store.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(log_lines), 1)
        record_payload = json.loads(log_lines[0])
        self.assertEqual(record_payload["event"]["event_type"], "user_message")
        self.assertEqual(record_payload["decision"]["action"], "record")

    def test_explicit_facet_proposal_wins_default_decision(self) -> None:
        runtime = NexusRuntime(
            facets=[EchoFacet(), AskFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="hello")
        )

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(record.decision.source_facets, ["asker"])

    def test_system_tick_defaults_to_wait_without_updates(self) -> None:
        runtime = NexusRuntime(
            facets=[SilentFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(Event(event_type=EventType.SYSTEM_TICK))

        self.assertEqual(record.decision.action, DecisionAction.WAIT)


if __name__ == "__main__":
    unittest.main()
