from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.facets import BehaviorFacet, EchoFacet, MemoryFacet
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore


def make_tempdir_path() -> Path:
    return Path.cwd() / f".test-behavior-{uuid4().hex}"


class BehaviorFacetRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.facet = BehaviorFacet()

    def test_empty_content_waits(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content=""),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.WAIT)
        self.assertEqual(result.metadata["selected_decision"], "wait")
        self.assertIn("empty_content_wait", result.metadata["reasons"])

    def test_normal_user_message_records(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="hello there"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(result.metadata["selected_decision"], "record")
        self.assertIn("user_message_default_record", result.metadata["reasons"])

    def test_hard_rule_style_content_is_recorded_with_high_priority_metadata(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="don't ever skip my boss emails now",
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertTrue(result.metadata["high_priority"])
        self.assertEqual(result.metadata["priority_level"], "high")
        self.assertIn("authority", result.metadata["tags_considered"])
        self.assertIn("urgent", result.metadata["tags_considered"])
        self.assertIn("hard-rule-candidate", result.metadata["tags_considered"])
        self.assertIn("correction", result.metadata["tags_considered"])
        self.assertIn("high_priority_tags", result.metadata["reasons"])

    def test_requires_response_metadata_asks(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="status update",
                metadata={"requires_response": True},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertTrue(result.metadata["response_needed"])
        self.assertIn("requires_response_metadata", result.metadata["reasons"])

    def test_explicit_action_with_low_risk_acts(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the note",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertEqual(result.metadata["selected_decision"], "act")
        self.assertIn("explicit_action_low_risk", result.metadata["reasons"])

    def test_explicit_action_without_low_risk_asks(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="delete the database",
                metadata={"explicit_action": True},
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertIn(
            "explicit_action_without_low_risk",
            result.metadata["reasons"],
        )

    def test_behavior_facet_works_without_memory_facet(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="how do I fix this?"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertFalse(result.metadata["memory_signal_available"])
        self.assertGreaterEqual(result.metadata["confidence"], 0.0)


class BehaviorRuntimeIntegrationTests(unittest.TestCase):
    def test_empty_content_waits_with_behavior_and_echo(self) -> None:
        runtime = NexusRuntime(
            facets=[BehaviorFacet(), EchoFacet()],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="")
        )

        self.assertEqual(record.decision.action, DecisionAction.WAIT)
        self.assertEqual(record.decision.source_facets, ["behavior"])

    def test_nexus_runs_with_memory_behavior_and_echo_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                BehaviorFacet(),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="how do I handle this next?",
                metadata={"tags": ["memory"]},
            )
        )

        self.assertEqual(len(record.facet_results), 3)
        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["memory", "behavior", "echo"],
        )
        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertTrue((root / "memory.sqlite3").exists())


class CLIBehaviorIntegrationTests(unittest.TestCase):
    def test_cli_with_behavior_flag_uses_behavior_facet(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--behavior",
                    "--content",
                    "what should I do next?",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["decision"]["action"], "ask")
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
