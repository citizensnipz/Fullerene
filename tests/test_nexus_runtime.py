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
from fullerene.workspace_state import workspace_state_root
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


class CountingFacet:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def process(self, event: Event, state: NexusState) -> FacetResult:
        self.calls.append(f"{self.name}:{event.event_type.value}")
        return FacetResult(
            facet_name=self.name,
            summary=f"{self.name} observed the event.",
            proposed_decision=DecisionAction.WAIT,
        )


class AttentionPressureFacet:
    name = "attention"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Emitted a deterministic attention peak.",
            metadata={"scores": {"event": 0.6}},
        )


class AffectPressureFacet:
    name = "affect"

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="Emitted deterministic affect arousal.",
            metadata={"affect_state": {"arousal": 0.3}},
        )


class InternalEmitterFacet:
    name = "behavior"

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def process(self, event: Event, state: NexusState) -> FacetResult:
        self.calls.append(event.event_type.value)
        return FacetResult(
            facet_name=self.name,
            summary="Emitted internal events for bounded processing.",
            metadata={
                "internal_events": [
                    Event(
                        event_type=EventType.INTERNAL,
                        content="first internal",
                        metadata={"source": "test"},
                    ),
                    Event(
                        event_type=EventType.INTERNAL,
                        content="second internal",
                        metadata={"source": "test"},
                    ),
                ]
            },
        )


class NexusRuntimeTests(unittest.TestCase):
    def make_file_store(self) -> FileStateStore:
        store_root = workspace_state_root() / f".test-nexus-store-{uuid4().hex}"
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

    def test_system_pressure_is_computed_and_clamped(self) -> None:
        runtime = NexusRuntime(facets=[SilentFacet()], store=InMemoryStateStore())

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="urgent",
                metadata={"pressure": 4.0},
            )
        )

        self.assertEqual(record.metadata["system_pressure"], 1.0)
        self.assertEqual(runtime.state.system_pressure, 1.0)

    def test_system_pressure_aggregates_attention_and_affect_outputs(self) -> None:
        runtime = NexusRuntime(
            facets=[AttentionPressureFacet(), AffectPressureFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="pressure signals",
                metadata={"pressure": 0.9},
            )
        )

        self.assertEqual(record.metadata["system_pressure"], 0.6)
        self.assertEqual(runtime.state.system_pressure, 0.6)

    def test_phases_execute_in_declared_order_and_facets_run_once(self) -> None:
        calls: list[str] = []
        runtime = NexusRuntime(
            facets=[
                CountingFacet("planner", calls),
                CountingFacet("memory", calls),
                CountingFacet("behavior", calls),
                CountingFacet("context", calls),
                CountingFacet("executor", calls),
                CountingFacet("learning", calls),
                CountingFacet("attention", calls),
                CountingFacet("affect", calls),
                CountingFacet("echo", calls),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="phase order")
        )

        self.assertEqual(
            calls,
            [
                "context:user_message",
                "memory:user_message",
                "behavior:user_message",
                "planner:user_message",
                "executor:user_message",
                "learning:user_message",
                "attention:user_message",
                "affect:user_message",
                "echo:user_message",
            ],
        )
        phase_trace = record.metadata["phase_execution_order"]
        self.assertEqual(
            [phase["phase"] for phase in phase_trace],
            [
                "INPUT / CONTEXT",
                "STATE",
                "DECISION",
                "PLANNING / EXECUTION",
                "LEARNING / SIGNAL",
                "VERIFICATION / OUTPUT",
            ],
        )
        self.assertEqual(phase_trace[0]["facets"], ["context", "memory"])
        self.assertEqual(
            record.metadata["facet_outputs_by_phase"]["PLANNING / EXECUTION"][0][
                "facet_name"
            ],
            "planner",
        )

    def test_internal_event_is_processed_at_most_once(self) -> None:
        calls: list[str] = []
        store = InMemoryStateStore()
        runtime = NexusRuntime(
            facets=[InternalEmitterFacet(calls)],
            store=store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="emit internal")
        )

        self.assertEqual(calls, ["user_message", "internal"])
        self.assertEqual(len(store.records), 2)
        self.assertEqual(store.records[1].event.event_type, EventType.INTERNAL)
        self.assertEqual(len(record.metadata["internal_events_processed"]), 1)
        self.assertEqual(record.metadata["internal_events_dropped"], 1)

    def test_low_pressure_behavior_decision_remains_unchanged(self) -> None:
        runtime = NexusRuntime(
            facets=[EchoFacet(), SilentFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="ordinary low pressure note",
                metadata={"pressure": 0.0},
            )
        )

        self.assertEqual(record.decision.action, DecisionAction.RECORD)
        self.assertEqual(record.metadata["system_pressure"], 0.0)


if __name__ == "__main__":
    unittest.main()
