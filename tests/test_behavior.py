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
from fullerene.workspace_state import workspace_state_root
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-behavior-{uuid4().hex}"


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

    def test_high_pressure_increases_act_and_ask_likelihood(self) -> None:
        low_pressure_act = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the note",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(),
        )
        high_pressure_act = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the note",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "pressure": 0.9,
                },
            ),
            NexusState(),
        )
        low_pressure_ask = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="what should I do?"),
            NexusState(),
        )
        high_pressure_ask = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do?",
                metadata={"pressure": 0.9},
            ),
            NexusState(),
        )

        self.assertGreater(
            high_pressure_act.metadata["decision_scores"]["act"],
            low_pressure_act.metadata["decision_scores"]["act"],
        )
        self.assertGreater(
            high_pressure_ask.metadata["decision_scores"]["ask"],
            low_pressure_ask.metadata["decision_scores"]["ask"],
        )
        self.assertLess(
            high_pressure_ask.metadata["decision_scores"]["wait"],
            low_pressure_ask.metadata["decision_scores"]["wait"],
        )

    def test_high_goal_priority_boosts_act(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="continue the task",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(
                facet_state={
                    "goals": {
                        "last_relevant_goals": [
                            {"id": "goal-1", "priority": 0.9, "score": 1.2}
                        ],
                        "last_relevance_score": 1.2,
                    }
                }
            ),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertEqual(result.metadata["goal_relevance"], 0.9)
        self.assertIn("goal priority boosted ACT score", result.metadata["reasons"])

    def test_low_memory_retrieval_boosts_ask_for_relevant_goal(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="continue the task"),
            NexusState(
                facet_state={
                    "goals": {
                        "last_relevant_goals": [
                            {"id": "goal-1", "priority": 0.9, "score": 1.2}
                        ],
                        "last_relevance_score": 1.2,
                    }
                }
            ),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertEqual(result.metadata["retrieval_strength"], 0.0)
        self.assertIn(
            "goal relevant but insufficient context",
            result.metadata["reasons"],
        )
        self.assertIn("low retrieval caused ASK preference", result.metadata["reasons"])

    def test_no_signals_defaults_to_record_or_wait(self) -> None:
        record_result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="ok"),
            NexusState(),
        )
        wait_result = self.facet.process(
            Event(event_type=EventType.SYSTEM_TICK, content=""),
            NexusState(),
        )

        self.assertEqual(record_result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(wait_result.proposed_decision, DecisionAction.WAIT)

    def test_confidence_is_clamped(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the urgent task now",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "pressure": 2.0,
                    "retrieval_strength": 2.0,
                    "salience": 2.0,
                },
            ),
            NexusState(
                facet_state={
                    "goals": {
                        "last_relevant_goals": [
                            {"id": "goal-1", "priority": 1.0, "score": 2.0}
                        ],
                        "last_relevance_score": 2.0,
                    }
                }
            ),
        )

        self.assertEqual(result.metadata["confidence"], 1.0)
        self.assertEqual(result.metadata["confidence_breakdown"]["total"], 1.0)

    def test_reasons_include_behavior_v1_contributing_factors(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do?",
                metadata={"pressure": 0.8},
            ),
            NexusState(),
        )
        reasons_text = "\n".join(result.metadata["reasons"])

        self.assertIn("pressure contribution", reasons_text)
        self.assertIn("goal relevance contribution", reasons_text)
        self.assertIn("memory contribution", reasons_text)
        self.assertIn("final confidence breakdown", reasons_text)

    def test_behavior_never_crashes_if_signal_stores_are_missing(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="continue",
                metadata={
                    "goals": {"active_goals": "not-a-list"},
                    "memory": {"relevant_memories": "not-a-list"},
                },
            ),
            NexusState(facet_state={"context": {}, "attention": {}, "goals": {}}),
        )

        self.assertIn(result.proposed_decision, set(DecisionAction))
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
        self.assertFalse((root / "memory.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
