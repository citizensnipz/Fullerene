from __future__ import annotations

import unittest

from fullerene.facets import BehaviorFacet
from fullerene.nexus import Event, EventType, FacetResult, NexusRuntime, NexusState
from fullerene.signals.latent_pressure import (
    LatentPressureEntry,
    LatentPressureResult,
    update_latent_pressure,
)
from fullerene.state import InMemoryStateStore


class LatentPressureModelTests(unittest.TestCase):
    def test_entry_round_trip_and_clamping(self) -> None:
        entry = LatentPressureEntry(
            source="behavior",
            entry_type="contradiction",
            description="x",
            intensity=2.0,
            decay_rate=-1.0,
            escalation_rate=2.0,
            retrigger_count=-5,
        )
        payload = entry.to_dict()
        self.assertEqual(payload["intensity"], 1.0)
        self.assertEqual(payload["decay_rate"], 0.0)
        self.assertEqual(payload["escalation_rate"], 1.0)
        self.assertEqual(payload["retrigger_count"], 0)
        self.assertEqual(LatentPressureEntry.from_dict(payload).to_dict(), payload)

    def test_result_round_trip_and_clamping(self) -> None:
        result = LatentPressureResult(latent_pressure_total=2.0, reasons=["a"])
        payload = result.to_dict()
        self.assertEqual(payload["latent_pressure_total"], 1.0)
        self.assertEqual(LatentPressureResult.from_dict(payload).to_dict(), payload)


class LatentPressureBufferTests(unittest.TestCase):
    def _state(self) -> NexusState:
        return NexusState()

    def test_creates_contradiction_from_behavior_trace(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        result = FacetResult(
            facet_name="behavior",
            summary="x",
            metadata={"decision_trace": {"contradiction_flag": True, "event": {"id": "e1"}}},
        )
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[result])
        self.assertTrue(any(e["entry_type"] == "contradiction" for e in lp.active_entries))

    def test_creates_policy_block_from_behavior(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        result = FacetResult(
            facet_name="behavior",
            summary="x",
            metadata={"decision_trace": {"policy_result": "approval_required"}},
        )
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[result])
        self.assertTrue(any(e["entry_type"] == "policy_block" for e in lp.active_entries))

    def test_creates_context_overload_from_signal_map(self) -> None:
        state = self._state()
        state.facet_state["nexus"] = {"current_cycle_signal_map": {"context_overloaded": True}}
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[])
        self.assertTrue(any(e["entry_type"] == "context_overload" for e in lp.active_entries))

    def test_creates_attention_conflict(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        att = FacetResult(
            facet_name="attention", summary="x", metadata={"attention_conflict": True}
        )
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[att])
        self.assertTrue(any(e["entry_type"] == "attention_conflict" for e in lp.active_entries))

    def test_creates_verifier_failure(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        ver = FacetResult(
            facet_name="verifier",
            summary="x",
            metadata={"artifact_checks": [{"code": "bad", "severity": "error", "status": "failed"}]},
        )
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[ver])
        self.assertTrue(any(e["entry_type"] == "verifier_failure" for e in lp.active_entries))

    def test_dedup_updates_existing_entry(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        res = FacetResult(
            facet_name="behavior", summary="x", metadata={"decision_trace": {"contradiction_flag": True}}
        )
        state_blob, _ = update_latent_pressure(event=event, state=state, facet_results=[res])
        state.facet_state["signals"] = {"latent_pressure": state_blob}
        _, second = update_latent_pressure(event=event, state=state, facet_results=[res])
        self.assertEqual(len(second.active_entries), 1)
        self.assertGreaterEqual(second.active_entries[0]["retrigger_count"], 2)

    def test_decay_and_resolution(self) -> None:
        state = self._state()
        now_event = Event(event_type=EventType.USER_MESSAGE, content="x")
        res = FacetResult(
            facet_name="behavior", summary="x", metadata={"decision_trace": {"contradiction_flag": True}}
        )
        state_blob, _ = update_latent_pressure(event=now_event, state=state, facet_results=[res])
        state.facet_state["signals"] = {"latent_pressure": state_blob}
        # explicit resolution
        resolve_event = Event(
            event_type=EventType.USER_MESSAGE,
            content="resolve",
            metadata={"resolve_latent_pressure": ["contradiction"]},
        )
        _, resolved = update_latent_pressure(event=resolve_event, state=state, facet_results=[])
        self.assertTrue(resolved.resolved_entries)

    def test_total_weighted_and_bounded(self) -> None:
        entries = [
            {
                "id": f"e{i}",
                "source": "unknown",
                "entry_type": "unknown",
                "description": str(i),
                "intensity": 1.0,
                "decay_rate": 0.05,
                "escalation_rate": 0.08,
                "retrigger_count": 1,
                "created_at": Event(event_type=EventType.SYSTEM_NOTE).timestamp.isoformat(),
                "last_activated_at": Event(event_type=EventType.SYSTEM_NOTE).timestamp.isoformat(),
                "last_decayed_at": Event(event_type=EventType.SYSTEM_NOTE).timestamp.isoformat(),
                "status": "active",
                "metadata": {},
            }
            for i in range(7)
        ]
        state = self._state()
        state.facet_state["signals"] = {"latent_pressure": {"entries": entries}}
        _, lp = update_latent_pressure(
            event=Event(event_type=EventType.USER_MESSAGE, content="none"),
            state=state,
            facet_results=[],
        )
        self.assertLessEqual(lp.latent_pressure_total, 1.0)

    def test_ignition_recommended_thresholds(self) -> None:
        state = self._state()
        event = Event(event_type=EventType.USER_MESSAGE, content="x")
        ver = FacetResult(
            facet_name="verifier",
            summary="x",
            metadata={"artifact_checks": [{"code": "bad", "severity": "critical", "status": "failed"}]},
        )
        state_blob, _ = update_latent_pressure(event=event, state=state, facet_results=[ver])
        state.facet_state["signals"] = {"latent_pressure": state_blob}
        _, lp = update_latent_pressure(event=event, state=state, facet_results=[ver])
        self.assertTrue(lp.ignition_recommended)


class LatentPressureRuntimeTests(unittest.TestCase):
    def test_not_registered_facet_and_nexus_persists_state(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet()], store=InMemoryStateStore())
        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do?",
                metadata={"latent_pressure": 0.8},
            )
        )
        facet_names = [result.facet_name for result in record.facet_results]
        self.assertNotIn("latent_pressure", facet_names)
        self.assertIn("signals", runtime.state.facet_state)
        self.assertIn("latent_pressure", runtime.state.facet_state["signals"])
        self.assertIn("latent_pressure_result", record.metadata)

    def test_behavior_consumes_following_cycle_latent_pressure(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet()], store=InMemoryStateStore())
        runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do?",
                metadata={"latent_pressure": 0.8},
            )
        )
        second = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="what should I do next?")
        )
        behavior = next(r for r in second.facet_results if r.facet_name == "behavior")
        self.assertGreaterEqual(float(behavior.metadata.get("latent_pressure", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()

