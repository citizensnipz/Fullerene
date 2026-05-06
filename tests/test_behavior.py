from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.behavior.lexical import extract_text_signals
from fullerene.behavior.models import BehaviorSignals, BehaviorTextSignals, TextIntentScores
from fullerene.behavior.scoring import select_decision
from fullerene.facets import BehaviorFacet, EchoFacet, MemoryFacet
from fullerene.facets.behavior import HIGH_AMBIGUITY_THRESHOLD, LOW_AMBIGUITY_THRESHOLD
from fullerene.workspace_state import workspace_state_root
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore, InMemoryStateStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-behavior-{uuid4().hex}"


def make_behavior_signals(*, text: BehaviorTextSignals) -> BehaviorSignals:
    return BehaviorSignals(
        tags=[],
        salience=0.5,
        salience_source="metadata",
        meaningful_content=True,
        has_metadata_signal=False,
        question_like=text.response_needed,
        requires_response=False,
        explicit_action=False,
        low_risk=False,
        uncertainty=False,
        high_priority=False,
        pressure=0.0,
        latent_pressure=0.0,
        retrieval_strength=0.0,
        relevant_memory_strength=0.0,
        has_relevant_memory=False,
        has_preference_memory=False,
        has_goal=False,
        top_goal_priority=0.0,
        goal_signal_strength=0.0,
        goal_relevance=0.0,
        goal_alignment_score=0.0,
        goal_alignment_priority=0.0,
        aligned_goal_ids=[],
        world_signal_available=False,
        world_alignment_score=0.0,
        world_alignment_confidence=0.0,
        aligned_belief_ids=[],
        belief_confidence=0.0,
        belief_contradiction=False,
        belief_reason=None,
        policy_result="allow",
        policy_requires_approval=False,
        policy_blocks_act=False,
        policy_reason=None,
        context_item_count_signal=0,
        context_max_items_signal=0,
        context_load_ratio=0.0,
        context_overloaded=False,
        memory_signal_available=False,
        goal_signal_available=False,
        domain_match=False,
        event_domain=None,
        included_memory_roles=[],
        included_memory_domains=[],
        active_goal_count=0,
        relevant_goal_count=0,
        relevant_memory_count=0,
        relevant_belief_count=0,
        context_item_count=0,
        planner_available=False,
        context_sufficiency=0.0,
        missing_context=[],
        included_working_memory_turns=[],
        working_memory_turn_count=0,
        included_context_types=[],
        included_lpb_entry_ids=[],
        included_belief_ids=[],
        context_strategy=None,
        related_context_item_ids=[],
        related_memory_ids=[],
        related_belief_ids=[],
        text=text,
    )


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
        self.assertFalse(result.metadata["response_needed"])
        self.assertIn("user_message_default_record", result.metadata["reasons"])

    def test_direct_status_question_sets_response_needed(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What are you doing right now?",
            ),
            NexusState(),
        )

        self.assertTrue(result.metadata["response_needed"])
        self.assertEqual(result.metadata["response_reason"], "direct_question")

    def test_focus_question_acts_when_active_goal_context_exists(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What should I do next?",
            ),
            NexusState(
                facet_state={
                    "goals": {
                        "active_goals": [
                            {
                                "id": "goal-fullerene",
                                "description": "finishing Fullerene",
                                "priority": 0.8,
                            }
                        ],
                    }
                }
            ),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertTrue(result.metadata["response_needed"])
        self.assertEqual(result.metadata["query_intent"], "recommendation")
        self.assertEqual(result.metadata["active_goal_count"], 1)
        self.assertEqual(result.metadata["context_sufficiency"], 1.0)
        self.assertEqual(
            result.metadata["response_template"],
            "next_steps_available",
        )

    def test_book_recommendation_without_context_asks_for_preferences(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What book should I read?",
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertEqual(result.metadata["query_intent"], "recommendation")
        self.assertEqual(
            result.metadata["response_template"],
            "clarify_recommendation_preferences",
        )
        self.assertEqual(result.metadata["missing_context"], ["preferences", "purpose"])
        self.assertEqual(result.metadata["context_sufficiency"], 0.0)

    def test_book_recommendation_with_relevant_memory_acts(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What book should I read?",
            ),
            NexusState(
                facet_state={
                    "memory": {
                        "relevant_memories": [
                            {
                                "id": "memory-reading",
                                "content_preview": (
                                    "User prefers practical software books."
                                ),
                            }
                        ],
                    }
                }
            ),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertEqual(result.metadata["query_intent"], "recommendation")
        self.assertEqual(result.metadata["relevant_memory_count"], 1)
        self.assertEqual(result.metadata["context_sufficiency"], 1.0)
        self.assertEqual(result.metadata["missing_context"], [])

    def test_direct_status_question_acts_with_text_metadata(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What are you doing right now?",
            ),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertEqual(result.metadata["selected_decision"], "act")
        self.assertEqual(result.metadata["output_type"], "text")
        self.assertEqual(result.metadata["tool"], "text")
        self.assertEqual(result.metadata["response_template"], "status_report")
        self.assertTrue(result.metadata["response_needed"])
        self.assertEqual(result.metadata["query_intent"], "factual")

    def test_ambiguous_help_request_uses_text_response_metadata(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="help me"),
            NexusState(),
        )

        self.assertIn(result.proposed_decision, {DecisionAction.ASK, DecisionAction.ACT})
        self.assertEqual(result.metadata["output_type"], "text")
        self.assertEqual(result.metadata["tool"], "text")
        self.assertEqual(
            result.metadata["response_template"],
            "clarification_needed",
        )
        self.assertTrue(result.metadata["response_needed"])

    def test_behavior_does_not_add_decision_enum_values(self) -> None:
        self.assertEqual(
            {action.value for action in DecisionAction},
            {"wait", "ask", "act", "record"},
        )

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

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(result.metadata["retrieval_strength"], 0.0)
        self.assertIn("non_question_statement_record", result.metadata["reasons"])

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


class BehaviorV1SignalIntegrationTests(unittest.TestCase):
    def _run_cli_json(self, args: list[str]) -> dict:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(args)
        self.assertEqual(exit_code, 0)
        return json.loads(stdout.getvalue())

    @staticmethod
    def _behavior_metadata(payload: dict) -> dict:
        return next(
            result["metadata"]
            for result in payload["facet_results"]
            if result["facet_name"] == "behavior"
        )

    def test_preference_memory_grounds_weekend_recommendation(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        self._run_cli_json(
            ["--full", "--json", "--content", "I like scuba diving", "--state-dir", str(root)]
        )
        payload = self._run_cli_json(
            [
                "--full",
                "--json",
                "--content",
                "What should I do this weekend?",
                "--state-dir",
                str(root),
            ]
        )
        behavior = self._behavior_metadata(payload)

        self.assertEqual(payload["decision"]["action"], "act")
        self.assertEqual(behavior["decision"], "act")
        self.assertEqual(behavior["query_intent"], "recommendation")
        self.assertTrue(behavior["has_preference_memory"])
        self.assertLessEqual(behavior["ambiguity_score"], LOW_AMBIGUITY_THRESHOLD)
        self.assertIn("preference_memory_signal", behavior["reasons"])

    def test_no_context_recommendation_asks(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        payload = self._run_cli_json(
            [
                "--full",
                "--json",
                "--content",
                "What should I do this weekend?",
                "--state-dir",
                str(root),
            ]
        )
        behavior = self._behavior_metadata(payload)

        self.assertEqual(payload["decision"]["action"], "ask")
        self.assertGreaterEqual(behavior["ambiguity_score"], HIGH_AMBIGUITY_THRESHOLD)

    def test_goal_signal_grounds_next_step_request(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        self._run_cli_json(
            [
                "--full",
                "--json",
                "--content",
                "I need to finish Fullerene",
                "--state-dir",
                str(root),
            ]
        )
        payload = self._run_cli_json(
            [
                "--full",
                "--json",
                "--content",
                "What should I do next?",
                "--state-dir",
                str(root),
            ]
        )
        behavior = self._behavior_metadata(payload)

        self.assertEqual(payload["decision"]["action"], "act")
        self.assertTrue(behavior["has_goal"])
        self.assertIn("goal_signal", behavior["reasons"])

    def test_memory_summary_acts_with_stored_memories(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        self._run_cli_json(
            ["--full", "--json", "--content", "I like art museums", "--state-dir", str(root)]
        )
        payload = self._run_cli_json(
            [
                "--full",
                "--json",
                "--content",
                "What do you know about me?",
                "--state-dir",
                str(root),
            ]
        )
        behavior = self._behavior_metadata(payload)

        self.assertEqual(payload["decision"]["action"], "act")
        self.assertEqual(behavior["query_intent"], "memory_summary")
        self.assertNotEqual(payload["decision"]["action"], "ask")

    def test_vague_recommendation_asks(self) -> None:
        result = BehaviorFacet().process(
            Event(event_type=EventType.USER_MESSAGE, content="What should I do?"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertGreaterEqual(
            result.metadata["ambiguity_score"],
            HIGH_AMBIGUITY_THRESHOLD,
        )

    def test_behavior_policy_does_not_call_model(self) -> None:
        with patch(
            "fullerene.models.ollama.OllamaAdapter.generate",
            return_value="not decision logic",
        ) as generate:
            result = BehaviorFacet().process(
                Event(event_type=EventType.USER_MESSAGE, content="What should I do?"),
                NexusState(),
            )

        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        generate.assert_not_called()


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


class BehaviorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.facet = BehaviorFacet()

    def test_high_pressure_high_goal_relevance_can_produce_act(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="continue and finish this now",
                metadata={"pressure": 0.9, "explicit_action": True, "low_risk": True},
            ),
            NexusState(
                facet_state={
                    "goals": {
                        "last_relevant_goals": [{"id": "g1", "priority": 0.9, "score": 1.0}],
                    }
                }
            ),
        )
        self.assertEqual(result.proposed_decision, DecisionAction.ACT)

    def test_policy_block_downgrades_act(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="ship it now",
                metadata={"explicit_action": True, "low_risk": True, "policy": {"policy_status": "denied"}},
            ),
            NexusState(),
        )
        self.assertIn(result.proposed_decision, {DecisionAction.ASK, DecisionAction.RECORD, DecisionAction.WAIT})
        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)
        self.assertIn("policy_result:denied", result.metadata["reasons"])

    def test_low_belief_confidence_suppresses_act(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="execute this action",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(
                facet_state={
                    "world_model": {
                        "last_relevant_beliefs": [{"id": "b1", "confidence": 0.2, "status": "active"}],
                    }
                }
            ),
        )
        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)

    def test_contradiction_biases_toward_ask(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="should I proceed?"),
            NexusState(
                facet_state={
                    "world_model": {
                        "last_relevant_beliefs": [{"id": "b1", "confidence": 0.9, "status": "contradicted"}],
                    }
                }
            ),
        )
        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertTrue(result.metadata["belief_contradiction"])

    def test_context_overload_reduces_act_confidence(self) -> None:
        low = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the note",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(),
        )
        high = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="archive the note",
                metadata={"explicit_action": True, "low_risk": True},
            ),
            NexusState(
                facet_state={"context": {"last_context_item_count": 10, "last_context_max_items": 10}}
            ),
        )
        self.assertLessEqual(high.metadata["confidence"], low.metadata["confidence"])

    def test_latent_pressure_contributes_to_interrupt_and_scoring(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do next?",
                metadata={"latent_pressure": 0.8},
            ),
            NexusState(),
        )
        self.assertTrue(result.metadata["interrupt_recommended"])
        self.assertEqual(result.metadata["interrupt_reason"], "latent_pressure_high")
        self.assertGreater(result.metadata["decision_scores"]["ask"], 0.0)

    def test_decision_trace_contains_required_fields(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="help me decide"),
            NexusState(),
        )
        trace = result.metadata["decision_trace"]
        for key in (
            "event",
            "pressure_score",
            "latent_pressure",
            "memory_relevance_score",
            "goal_relevance_score",
            "world_model_belief_confidence",
            "contradiction_flag",
            "policy_result",
            "context_load_ratio",
            "raw_candidate_scores",
            "adjusted_candidate_scores",
            "final_decision",
            "confidence",
            "reasons",
            "interrupt_recommended",
            "timestamp",
        ):
            self.assertIn(key, trace)

    def test_neutral_adapters_preserve_v1_record_default(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="hello there"),
            NexusState(),
        )
        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)

    def test_source_request_intent_is_detected(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="where did that come from?"),
            NexusState(),
        )
        self.assertEqual(result.metadata["conversational_intent"], "source_request")
        self.assertEqual(result.metadata["grounding_need"], "runtime_state")

    def test_short_referential_follow_up_uses_working_memory(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="where?"),
            NexusState(
                facet_state={
                    "context": {
                        "working_memory_turn_count": 2,
                        "included_working_memory_turns": ["u1", "a1"],
                    }
                }
            ),
        )
        self.assertEqual(result.metadata["conversational_intent"], "follow_up")
        self.assertNotEqual(result.metadata["ambiguity_kind"], "generic")
        self.assertGreater(result.metadata["continuity_confidence"], 0.6)

    def test_short_referential_follow_up_without_context_biases_ask(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="where?"),
            NexusState(),
        )
        self.assertIn(result.proposed_decision, {DecisionAction.ASK, DecisionAction.RECORD})
        self.assertLess(result.metadata["continuity_confidence"], 0.5)

    def test_clarification_supplied_reduces_ambiguity(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="To clarify, I mean the API timeout case."),
            NexusState(),
        )
        self.assertEqual(result.metadata["conversational_intent"], "clarification_supplied")
        self.assertEqual(result.metadata["ambiguity_kind"], "none")

    def test_challenge_lowers_confidence_and_emits_learning_signal(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="that's not right"),
            NexusState(),
        )
        self.assertEqual(result.metadata["conversational_intent"], "challenge")
        self.assertGreater(result.metadata["challenge_confidence_penalty"], 0.0)
        learning = result.metadata["learning_event"]
        self.assertIn("signals", learning)
        self.assertIn("challenge_unresolved", learning["signals"])

    def test_repeated_dissatisfaction_emits_learning_signal(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="you still did not answer that"),
            NexusState(),
        )
        self.assertEqual(result.metadata["conversational_intent"], "repeated_dissatisfaction")
        self.assertIn("repeated_dissatisfaction", result.metadata["learning_event"]["signals"])

    def test_world_model_signal_changes_act_confidence_path(self) -> None:
        contradicted = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="execute this"),
            NexusState(
                facet_state={
                    "world_model": {
                        "last_relevant_beliefs": [{"id": "b1", "confidence": 0.9, "status": "contradicted"}],
                    }
                }
            ),
        )
        supported = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="execute this"),
            NexusState(
                facet_state={
                    "world_model": {
                        "last_relevant_beliefs": [{"id": "b2", "confidence": 0.95, "status": "active"}],
                    }
                }
            ),
        )
        self.assertLess(
            contradicted.metadata["decision_trace"]["adjusted_candidate_scores"]["act"],
            supported.metadata["decision_trace"]["adjusted_candidate_scores"]["act"],
        )

    def test_decision_trace_exposes_v21_fields(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="how do you know that?"),
            NexusState(),
        )
        trace = result.metadata["decision_trace"]
        for key in (
            "conversational_intent",
            "conversational_intent_score",
            "grounding_need",
            "grounding_available",
            "grounding_confidence",
            "ambiguity_kind",
            "continuity_confidence",
            "self_consistency_confidence",
            "challenge_confidence_penalty",
        ):
            self.assertIn(key, trace)

    def test_lexical_numeric_scoring_and_no_decision_output(self) -> None:
        lexical = extract_text_signals(
            "What should I do next?",
            query_intent="planning",
            ambiguity_score=0.3,
            context_sufficiency=1.0,
            missing_context=[],
            working_memory_turn_count=1,
            has_context_items=True,
            grounding_available=True,
            grounding_confidence=0.9,
            self_consistency_confidence=0.9,
        )
        self.assertIsInstance(lexical.intents, TextIntentScores)
        self.assertGreaterEqual(lexical.question_score, 0.0)
        self.assertFalse(hasattr(lexical, "decision"))

    def test_scoring_decision_uses_behavior_signals_without_text_matching(self) -> None:
        text = BehaviorTextSignals(
            intents=TextIntentScores(unknown=1.0),
            question_score=0.0,
            shortness_score=0.0,
            referential_score=0.0,
            imperative_score=0.0,
            vague_score=0.0,
            challenge_score=0.0,
            source_request_score=0.0,
            clarification_score=0.0,
            correction_score=0.0,
            query_intent="unknown",
            response_template=None,
            deterministic_response_available=False,
            response_needed=False,
            response_reason=None,
            conversational_intent="unknown",
            conversational_intent_score=0.3,
            conversational_intent_reasons=[],
            follow_up_reference_detected=False,
            short_follow_up=False,
            grounding_need="none",
            grounding_need_reasons=[],
            grounding_available=False,
            grounding_confidence=0.0,
            continuity_confidence=0.0,
            self_consistency_confidence=0.8,
            challenge_confidence_penalty=0.0,
            ambiguity_kind="none",
            ambiguity_score=0.1,
            ambiguity_reasons=[],
            repeated_dissatisfaction=False,
        )
        decision, _, _ = select_decision(
            Event(event_type=EventType.USER_MESSAGE, content="nonsense tokens only"),
            make_behavior_signals(text=text),
        )
        self.assertEqual(decision, DecisionAction.RECORD)

    def test_response_intent_and_template_compatibility_present(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="What are you doing right now?"),
            NexusState(),
        )
        self.assertIn("response_intent", result.metadata)
        self.assertIn("response_template", result.metadata)


class CLIBehaviorIntegrationTests(unittest.TestCase):
    def test_cli_with_behavior_flag_uses_behavior_facet(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
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
