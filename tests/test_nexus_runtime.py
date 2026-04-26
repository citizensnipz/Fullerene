from __future__ import annotations

import json
import shutil
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
from fullerene.scratch import scratch_root
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


class ExplodingFacet:
    name = "exploder"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        raise ValueError("facet exploded")


class ActFacet:
    name = "actor"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Proposed a future action for the event.",
            proposed_decision=DecisionAction.ACT,
        )


class NexusRuntimeTests(unittest.TestCase):
    def make_file_store(self) -> FileStateStore:
        store_root = scratch_root() / f".test-nexus-store-{uuid4().hex}"
        self.addCleanup(lambda: shutil.rmtree(store_root, ignore_errors=True))
        return FileStateStore(store_root)

    def test_process_event_persists_state_and_log(self) -> None:
        store = self.make_file_store()
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

    def test_priority_order_prefers_act_over_ask_and_record(self) -> None:
        runtime = NexusRuntime(
            facets=[EchoFacet(), AskFacet(), ActFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="hello")
        )

        self.assertEqual(record.decision.action, DecisionAction.ACT)
        self.assertEqual(record.decision.source_facets, ["actor"])

    def test_facet_errors_are_isolated_and_persisted(self) -> None:
        store = self.make_file_store()
        runtime = NexusRuntime(
            facets=[ExplodingFacet(), EchoFacet()],
            store=store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="hello nexus")
        )

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertEqual(len(record.facet_results), 2)

        error_result = record.facet_results[0]
        self.assertEqual(error_result.facet_name, "exploder")
        self.assertEqual(error_result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(error_result.metadata["error_type"], "ValueError")
        self.assertEqual(error_result.metadata["error_message"], "facet exploded")
        self.assertIn("failed while processing the event", error_result.summary)
        self.assertNotIn("Traceback", error_result.summary)

        echo_result = record.facet_results[1]
        self.assertEqual(echo_result.facet_name, "echo")
        self.assertEqual(
            runtime.state.facet_state["echo"]["last_user_message"],
            "hello nexus",
        )

        state_payload = json.loads(store.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state_payload["event_count"], 1)
        self.assertEqual(
            state_payload["facet_state"]["echo"]["last_user_message"],
            "hello nexus",
        )
        self.assertNotIn("exploder", state_payload["facet_state"])

        record_payload = json.loads(
            store.log_path.read_text(encoding="utf-8").strip().splitlines()[0]
        )
        persisted_error_result = record_payload["facet_results"][0]
        self.assertEqual(persisted_error_result["facet_name"], "exploder")
        self.assertEqual(
            persisted_error_result["metadata"]["error_type"],
            "ValueError",
        )
        self.assertEqual(
            persisted_error_result["metadata"]["error_message"],
            "facet exploded",
        )
        self.assertNotIn("Traceback", json.dumps(persisted_error_result))


if __name__ == "__main__":
    unittest.main()
