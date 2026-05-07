from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fullerene.cli import _build_model_prompt, main as cli_main
from fullerene.context import (
    ConversationContinuity,
    DYNAMIC_ACTIVE_FACETS_V1,
    PRESSURE_RELEVANCE_V2,
    SELF_EDITING_V3,
    ReferenceAnchor,
    STATIC_RECENT_EPISODIC_V0,
    ContextAssemblyConfig,
    ContextItem,
    ContextItemType,
    ContextWindow,
    DynamicContextAssembler,
    StaticContextAssembler,
    derive_reference_anchors,
)
from fullerene.attention import AttentionBroadcast, AttentionMode, AttentionSource
from fullerene.facets import ContextFacet, EchoFacet, GoalsFacet, MemoryFacet, WorldModelFacet
from fullerene.goals import (
    Goal,
    GoalSource,
    GoalStatus,
    SQLiteGoalStore,
    normalize_goal_description,
)
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.memory.models import utcnow
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicySource,
    PolicyTargetType,
    SQLitePolicyStore,
)
from fullerene.state import FileStateStore
from fullerene.world_model import Belief, BeliefSource, BeliefStatus, SQLiteWorldModelStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-context-{uuid4().hex}"


class TrackingMemoryStore:
    def __init__(
        self,
        records: list[MemoryRecord] | None = None,
        *,
        relevant_records: list[MemoryRecord] | None = None,
    ) -> None:
        self.records = list(records or [])
        self.relevant_records = list(relevant_records or records or [])
        self.list_recent_calls: list[tuple[int, MemoryType | None]] = []
        self.retrieve_relevant_calls: list[tuple[str, int]] = []

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        self.list_recent_calls.append((limit, memory_type))
        records = sorted(
            self.records,
            key=lambda record: (record.created_at, record.id),
            reverse=True,
        )
        if memory_type is not None:
            records = [
                record for record in records if record.memory_type == memory_type
            ]
        return records[:limit]

    def retrieve_relevant(self, event: Event, limit: int) -> list[MemoryRecord]:
        self.retrieve_relevant_calls.append((event.event_id, limit))
        records = sorted(
            self.relevant_records,
            key=lambda record: (record.salience, record.created_at, record.id),
            reverse=True,
        )
        return records[:limit]


class TrackingGoalStore:
    def __init__(self, goals: list[Goal] | None = None) -> None:
        self.goals = list(goals or [])
        self.calls: list[int] = []

    def list_active_goals(self, limit: int) -> list[Goal]:
        self.calls.append(limit)
        active_goals = [
            goal for goal in self.goals if goal.status == GoalStatus.ACTIVE
        ]
        active_goals.sort(
            key=lambda goal: (goal.priority, goal.updated_at, goal.id),
            reverse=True,
        )
        return active_goals[:limit]


class TrackingWorldModelStore:
    def __init__(self, beliefs: list[Belief] | None = None) -> None:
        self.beliefs = list(beliefs or [])
        self.calls: list[int] = []

    def list_active_beliefs(self, limit: int) -> list[Belief]:
        self.calls.append(limit)
        beliefs = sorted(
            self.beliefs,
            key=lambda belief: (belief.confidence, belief.updated_at, belief.id),
            reverse=True,
        )
        return beliefs[:limit]

    def list_beliefs(self, limit: int, status: BeliefStatus | None = None) -> list[Belief]:
        beliefs = self.list_active_beliefs(limit)
        if status is None:
            return beliefs
        return [belief for belief in beliefs if belief.status == status]


class TrackingPolicyStore:
    def __init__(self, policies: list[PolicyRule] | None = None) -> None:
        self.policies = list(policies or [])
        self.list_policies_calls: list[tuple[int, bool]] = []
        self.count_calls = 0

    def list_policies(
        self,
        limit: int,
        enabled_only: bool = False,
    ) -> list[PolicyRule]:
        self.list_policies_calls.append((limit, enabled_only))
        policies = [policy for policy in self.policies if policy.enabled or not enabled_only]
        policies.sort(
            key=lambda policy: (policy.priority, policy.updated_at, policy.id),
            reverse=True,
        )
        return policies[:limit]

    def list_enabled_policies(self, limit: int | None = None) -> list[PolicyRule]:
        policies = [policy for policy in self.policies if policy.enabled]
        if limit is None:
            return list(policies)
        return policies[:limit]

    def count_enabled_policies(self) -> int:
        self.count_calls += 1
        return len([policy for policy in self.policies if policy.enabled])


class HybridTrackingMemoryStore(TrackingMemoryStore):
    def __init__(
        self,
        records: list[MemoryRecord] | None = None,
        *,
        hybrid_pairs: list[tuple[MemoryRecord, dict[str, float]]] | None = None,
    ) -> None:
        super().__init__(records=records or [])
        self.hybrid_pairs = list(hybrid_pairs or [])
        self.hybrid_calls: list[tuple[str, int, str | None]] = []

    def hybrid_retrieve_relevant(
        self,
        event: Event,
        *,
        limit: int,
        domain_hint: str | None = None,
    ) -> list[tuple[MemoryRecord, dict[str, float]]]:
        self.hybrid_calls.append((event.event_id, limit, domain_hint))
        return self.hybrid_pairs[:limit]


class ContextModelTests(unittest.TestCase):
    def test_context_assembly_config_defaults_are_correct(self) -> None:
        config = ContextAssemblyConfig()

        self.assertEqual(config.max_goals, 3)
        self.assertEqual(config.max_memories, 5)
        self.assertEqual(config.max_beliefs, 5)
        self.assertEqual(config.salience_threshold, 0.0)
        self.assertTrue(config.include_policy_summary)
        self.assertTrue(config.include_signal_summaries)
        self.assertTrue(config.include_belief_consistency)
        self.assertEqual(config.strategy, DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(config.max_items, 21)

    def test_context_v2_defaults_and_clamping(self) -> None:
        config = ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, max_items_total=-1)
        self.assertEqual(config.strategy, PRESSURE_RELEVANCE_V2)
        self.assertEqual(config.max_items_total, 1)
        self.assertEqual(config.max_items, 1)
        self.assertEqual(config.min_relevance_score, 0.15)
        self.assertEqual(config.min_pressure_score, 0.20)

    def test_context_v3_defaults_and_clamping(self) -> None:
        config = ContextAssemblyConfig(
            strategy=SELF_EDITING_V3,
            low_pressure_threshold=-1,
            overload_pressure_threshold=2,
            predictive_min_score=4,
            consolidation_min_items=1,
        )
        self.assertEqual(config.strategy, SELF_EDITING_V3)
        self.assertEqual(config.low_pressure_threshold, 0.0)
        self.assertEqual(config.overload_pressure_threshold, 1.0)
        self.assertEqual(config.predictive_min_score, 1.0)
        self.assertGreaterEqual(config.consolidation_min_items, 2)

    def test_context_item_round_trips_through_dict(self) -> None:
        created_at = utcnow() - timedelta(minutes=30)
        item = ContextItem(
            id="context-item-1",
            item_type=ContextItemType.MEMORY,
            content="Remember the latest user instruction.",
            source_id="event-1",
            created_at=created_at,
            metadata={"tags": ["memory"], "salience": 0.8},
        )

        round_tripped = ContextItem.from_dict(item.to_dict())

        self.assertEqual(round_tripped, item)

    def test_context_window_round_trips_through_dict(self) -> None:
        window = ContextWindow(
            id="context-window-1",
            created_at=utcnow(),
            items=[
                ContextItem(
                    id="item-1",
                    item_type=ContextItemType.MEMORY,
                    content="Latest episodic note",
                )
            ],
            max_items=5,
            strategy=STATIC_RECENT_EPISODIC_V0,
            metadata={"source_types": ["episodic_memory"]},
        )

        round_tripped = ContextWindow.from_dict(window.to_dict())

        self.assertEqual(round_tripped, window)

    def test_context_window_can_use_dynamic_strategy(self) -> None:
        window = ContextWindow(
            items=[
                ContextItem(
                    id="event-1",
                    item_type=ContextItemType.EVENT,
                    content="What should I do next?",
                )
            ],
            max_items=20,
            strategy=DYNAMIC_ACTIVE_FACETS_V1,
        )

        round_tripped = ContextWindow.from_dict(window.to_dict())

        self.assertEqual(round_tripped.strategy, DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(round_tripped.items[0].item_type, ContextItemType.EVENT)

    def test_reference_anchor_round_trip(self) -> None:
        anchor = ReferenceAnchor(
            anchor_id="a1",
            surface_form="that",
            referent_text="release note",
            referent_source_turn_id="t2",
            referent_source_role="assistant",
            confidence=0.72,
            reason="recent_noun_phrase",
            current_message_fragment="use that one",
            metadata={"source": "working_memory"},
        )
        round_tripped = ReferenceAnchor.from_dict(anchor.to_dict())
        self.assertEqual(round_tripped.surface_form, "that")
        self.assertEqual(round_tripped.referent_text, "release note")
        self.assertAlmostEqual(round_tripped.confidence, 0.72, places=2)

    def test_conversation_continuity_round_trip(self) -> None:
        continuity = ConversationContinuity(
            current_topic_hint="recent discussion about release/note",
            topic_terms=["release", "note"],
            reference_anchors=[
                ReferenceAnchor(
                    anchor_id="a1",
                    surface_form="that",
                    referent_text="release note",
                    confidence=0.7,
                )
            ],
            unresolved_references=["it"],
            continuity_confidence=0.65,
            working_memory_turn_count=3,
            source="working_memory",
        )
        parsed = ConversationContinuity.from_dict(continuity.to_dict())
        self.assertEqual(parsed.current_topic_hint, continuity.current_topic_hint)
        self.assertEqual(parsed.topic_terms, continuity.topic_terms)
        self.assertEqual(len(parsed.reference_anchors), 1)
        self.assertEqual(parsed.unresolved_references, ["it"])


class ContextFacetExportTests(unittest.TestCase):
    def test_context_facet_is_exported_from_fullerene_facets(self) -> None:
        self.assertEqual(ContextFacet.__name__, "ContextFacet")


class StaticContextAssemblerTests(unittest.TestCase):
    def test_returns_empty_context_when_no_memory_records_exist(self) -> None:
        store = TrackingMemoryStore()
        assembler = StaticContextAssembler(store, max_items=5)

        window = assembler.assemble()

        self.assertEqual(window.items, [])
        self.assertEqual(window.max_items, 5)
        self.assertEqual(window.strategy, STATIC_RECENT_EPISODIC_V0)
        self.assertEqual(store.list_recent_calls, [(5, MemoryType.EPISODIC)])

    def test_includes_only_recent_episodic_memories(self) -> None:
        records = [
            MemoryRecord(
                id="episodic-old",
                created_at=utcnow() - timedelta(days=2),
                memory_type=MemoryType.EPISODIC,
                content="older episodic memory",
                tags=["memory"],
            ),
            MemoryRecord(
                id="semantic-new",
                created_at=utcnow() - timedelta(hours=1),
                memory_type=MemoryType.SEMANTIC,
                content="semantic memory that must be ignored",
                tags=["fact"],
            ),
            MemoryRecord(
                id="episodic-new",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="newer episodic memory",
                tags=["memory"],
            ),
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=5)

        window = assembler.assemble()

        self.assertEqual([item.id for item in window.items], ["episodic-new", "episodic-old"])
        self.assertEqual(
            [item.item_type for item in window.items],
            [ContextItemType.MEMORY, ContextItemType.MEMORY],
        )

    def test_respects_max_items_window_size(self) -> None:
        records = [
            MemoryRecord(
                id=f"episodic-{index}",
                created_at=utcnow() - timedelta(minutes=index),
                memory_type=MemoryType.EPISODIC,
                content=f"episodic memory {index}",
                tags=["memory"],
            )
            for index in range(4)
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=2)

        window = assembler.assemble()

        self.assertEqual(len(window.items), 2)
        self.assertEqual(store.list_recent_calls, [(2, MemoryType.EPISODIC)])

    def test_does_not_load_all_memory(self) -> None:
        records = [
            MemoryRecord(
                id="episodic-1",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="episodic memory 1",
            )
        ]
        store = TrackingMemoryStore(records)
        assembler = StaticContextAssembler(store, max_items=3)

        assembler.assemble()

        self.assertEqual(store.list_recent_calls, [(3, MemoryType.EPISODIC)])


class DynamicContextAssemblerTests(unittest.TestCase):
    def make_event(self, content: str = "What should I do next?") -> Event:
        return Event(event_type=EventType.USER_MESSAGE, content=content)

    def test_includes_current_event_and_active_state(self) -> None:
        event = self.make_event("What should I do next about Fullerene?")
        memory_store = TrackingMemoryStore(
            [
                MemoryRecord(
                    id="recent-memory",
                    created_at=utcnow() - timedelta(minutes=5),
                    memory_type=MemoryType.EPISODIC,
                    content="User mentioned finishing Fullerene soon.",
                    salience=0.7,
                    tags=["goals"],
                )
            ],
            relevant_records=[
                MemoryRecord(
                    id="relevant-memory",
                    created_at=utcnow() - timedelta(minutes=1),
                    memory_type=MemoryType.EPISODIC,
                    content="User said finishing Fullerene is important.",
                    salience=0.9,
                    tags=["goals", "memory"],
                )
            ],
        )
        goal_store = TrackingGoalStore(
            [
                Goal(
                    id="goal-1",
                    description="finish Fullerene",
                    priority=0.8,
                    status=GoalStatus.ACTIVE,
                    tags=["goals"],
                    source=GoalSource.USER,
                )
            ]
        )
        world_store = TrackingWorldModelStore(
            [
                Belief(
                    id="belief-1",
                    claim="SQLite is the canonical memory store.",
                    confidence=0.9,
                    tags=["memory"],
                    source=BeliefSource.SYSTEM,
                )
            ]
        )
        policy_store = TrackingPolicyStore(
            [
                PolicyRule(
                    id="policy-shell",
                    name="Require approval for shell",
                    rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                    target_type=PolicyTargetType.SHELL,
                    target="*",
                    source=PolicySource.SYSTEM,
                )
            ]
        )
        state = NexusState(
            facet_state={
                "planner": {
                    "last_plan": {"steps": [{"description": "Finish Context v1"}]},
                    "last_plan_confidence": 0.75,
                    "last_trigger_reason": "high_priority_goal_next_steps",
                },
                "attention": {
                    "last_dominant_source": "goal",
                    "last_focus_item_ids": ["goal:goal-1"],
                },
                "affect": {
                    "last_affect_state": {
                        "valence": 0.1,
                        "arousal": 0.4,
                        "dominance": 0.6,
                        "novelty": 0.2,
                    }
                },
                "learning": {
                    "last_signal_count": 1,
                    "last_adjustment_count": 1,
                    "last_proposal_count": 0,
                },
            }
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            goal_store=goal_store,
            world_model_store=world_store,
            policy_store=policy_store,
            config=ContextAssemblyConfig(),
        )

        window = assembler.assemble(event=event, state=state)

        self.assertEqual(window.strategy, DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(window.items[0].item_type, ContextItemType.EVENT)
        self.assertEqual(window.items[0].content, event.content)
        self.assertIn("goal-1", window.metadata["included_goal_ids"])
        self.assertIn("relevant-memory", window.metadata["included_memory_ids"])
        self.assertIn("belief-1", window.metadata["included_belief_ids"])
        self.assertEqual(memory_store.list_recent_calls, [(5, MemoryType.EPISODIC)])
        self.assertEqual(memory_store.retrieve_relevant_calls, [(event.event_id, 5)])
        self.assertEqual(goal_store.calls, [15])
        self.assertEqual(world_store.calls, [10])
        self.assertEqual(policy_store.list_policies_calls, [(20, True)])
        self.assertEqual(policy_store.count_calls, 1)

    def test_includes_belief_consistency_from_prior_world_model_facet_state(self) -> None:
        event = self.make_event()
        memory_store = TrackingMemoryStore([])
        state = NexusState(
            facet_state={
                "world_model": {
                    "wm_v2_requires_approval_due_to_contradiction": True,
                    "wm_v2_top_belief_cluster_pressure": 0.5,
                    "wm_v2_top_contradiction_score": 0.1,
                    "wm_v2_belief_graph_confidence": 0.3,
                    "wm_v2_contradiction_cluster_sample": [
                        {"cluster_id": "cc-test-1", "pressure_score": 0.5, "member_count": 3},
                    ],
                },
            },
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                max_goals=0,
                max_memories=0,
                max_beliefs=0,
                include_goals=False,
                include_world_model=False,
                include_signal_summaries=False,
                include_policy_summary=False,
            ),
        )
        window = assembler.assemble(event=event, state=state)

        consistency_items = [
            item
            for item in window.items
            if item.item_type == ContextItemType.BELIEF_CONSISTENCY
        ]
        self.assertEqual(len(consistency_items), 1)
        self.assertIn("Belief consistency", consistency_items[0].content)
        self.assertEqual(
            window.metadata.get("included_wm_v2_contradiction_cluster_ids"),
            ["cc-test-1"],
        )
        self.assertTrue(window.metadata.get("belief_consistency_prior_wm"))

    def test_filters_memories_by_salience_threshold(self) -> None:
        event = self.make_event()
        memory_store = TrackingMemoryStore(
            [
                MemoryRecord(
                    id="recent-low",
                    created_at=utcnow(),
                    memory_type=MemoryType.EPISODIC,
                    content="low salience recent memory",
                    salience=0.2,
                ),
                MemoryRecord(
                    id="recent-high",
                    created_at=utcnow() - timedelta(minutes=1),
                    memory_type=MemoryType.EPISODIC,
                    content="high salience recent memory",
                    salience=0.8,
                ),
            ],
            relevant_records=[
                MemoryRecord(
                    id="relevant-low",
                    created_at=utcnow(),
                    memory_type=MemoryType.EPISODIC,
                    content="low salience relevant memory",
                    salience=0.1,
                ),
                MemoryRecord(
                    id="relevant-high",
                    created_at=utcnow() - timedelta(minutes=2),
                    memory_type=MemoryType.EPISODIC,
                    content="high salience relevant memory",
                    salience=0.9,
                ),
            ],
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                max_memories=5,
                salience_threshold=0.5,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )

        window = assembler.assemble(event=event, state=NexusState())

        memory_ids = [
            item.id for item in window.items if item.item_type == ContextItemType.MEMORY
        ]
        self.assertEqual(memory_ids, ["relevant-high", "recent-high"])

    def test_respects_maximums_and_does_not_duplicate_memories(self) -> None:
        event = self.make_event()
        duplicate_memory = MemoryRecord(
            id="memory-1",
            created_at=utcnow(),
            memory_type=MemoryType.EPISODIC,
            content="duplicate memory",
            salience=0.9,
        )
        assembler = DynamicContextAssembler(
            memory_store=TrackingMemoryStore(
                [
                    duplicate_memory,
                    MemoryRecord(
                        id="memory-2",
                        created_at=utcnow() - timedelta(minutes=1),
                        memory_type=MemoryType.EPISODIC,
                        content="second memory",
                        salience=0.8,
                    ),
                ],
                relevant_records=[
                    duplicate_memory,
                    MemoryRecord(
                        id="memory-3",
                        created_at=utcnow() - timedelta(minutes=2),
                        memory_type=MemoryType.EPISODIC,
                        content="third memory",
                        salience=0.7,
                    ),
                ],
            ),
            goal_store=TrackingGoalStore(
                [
                    Goal(
                        id=f"goal-{index}",
                        description=f"goal {index}",
                        priority=1.0 - (index * 0.1),
                        status=GoalStatus.ACTIVE,
                        source=GoalSource.USER,
                    )
                    for index in range(5)
                ]
            ),
            world_model_store=TrackingWorldModelStore(
                [
                    Belief(
                        id=f"belief-{index}",
                        claim=f"belief {index}",
                        confidence=1.0 - (index * 0.1),
                        source=BeliefSource.SYSTEM,
                    )
                    for index in range(5)
                ]
            ),
            config=ContextAssemblyConfig(
                max_goals=2,
                max_memories=2,
                max_beliefs=2,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )

        window = assembler.assemble(event=event, state=NexusState())

        goal_count = len(
            [item for item in window.items if item.item_type == ContextItemType.GOAL]
        )
        belief_count = len(
            [item for item in window.items if item.item_type == ContextItemType.BELIEF]
        )
        memory_ids = [
            item.id for item in window.items if item.item_type == ContextItemType.MEMORY
        ]
        self.assertEqual(goal_count, 2)
        self.assertEqual(belief_count, 2)
        self.assertEqual(memory_ids, ["memory-1", "memory-3"])

    def test_deduplicates_near_duplicate_active_goals(self) -> None:
        event = self.make_event("What should I do next about Fullerene?")
        best_goal = Goal(
            id="goal-best",
            description="finish Fullerene",
            priority=0.9,
            status=GoalStatus.ACTIVE,
            source=GoalSource.USER,
        )
        exact_duplicate = Goal(
            id="goal-exact",
            description="remember to finish Fullerene",
            priority=0.8,
            status=GoalStatus.ACTIVE,
            source=GoalSource.USER,
        )
        near_duplicate = Goal(
            id="goal-near",
            description="finishing Fullerene",
            priority=0.7,
            status=GoalStatus.ACTIVE,
            source=GoalSource.USER,
        )
        distinct_goal = Goal(
            id="goal-distinct",
            description="ship the release",
            priority=0.6,
            status=GoalStatus.ACTIVE,
            source=GoalSource.USER,
        )
        best_goal.updated_at = utcnow()
        exact_duplicate.updated_at = utcnow() - timedelta(minutes=5)
        near_duplicate.updated_at = utcnow() - timedelta(minutes=10)
        distinct_goal.updated_at = utcnow() - timedelta(minutes=15)

        assembler = DynamicContextAssembler(
            goal_store=TrackingGoalStore(
                [exact_duplicate, near_duplicate, best_goal, distinct_goal]
            ),
            config=ContextAssemblyConfig(
                max_goals=3,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )

        window = assembler.assemble(event=event, state=NexusState())

        goal_items = [
            item for item in window.items if item.item_type == ContextItemType.GOAL
        ]
        self.assertEqual([item.id for item in goal_items], ["goal-best", "goal-distinct"])
        self.assertEqual(window.metadata["deduped_goal_count"], 2)
        self.assertEqual(
            set(window.metadata["deduped_goal_ids"]),
            {"goal-exact", "goal-near"},
        )
        self.assertEqual(
            window.metadata["normalized_goal_keys"],
            ["finish fullerene", "ship the release"],
        )

    def test_handles_missing_stores_gracefully(self) -> None:
        event = self.make_event()
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(
                include_policy_summary=False,
                include_signal_summaries=False,
            )
        )

        window = assembler.assemble(event=event, state=NexusState())

        self.assertEqual(window.strategy, DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(len(window.items), 1)
        self.assertEqual(window.items[0].item_type, ContextItemType.EVENT)

    def test_includes_recent_working_memory_for_matching_session_only(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_working_turn(
            content="Do you know your name?",
            session_id="session-a",
            turn_index=1,
            dialogue_role="user",
        )
        memory_store.add_working_turn(
            content="I don't have a name.",
            session_id="session-a",
            turn_index=2,
            dialogue_role="assistant",
        )
        memory_store.add_working_turn(
            content="Other session turn",
            session_id="session-b",
            turn_index=1,
            dialogue_role="user",
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                max_working_turns=8,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )
        window = assembler.assemble(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="Would you like one?",
                metadata={"session_id": "session-a"},
            ),
            state=NexusState(),
        )
        working_items = [
            item for item in window.items if item.item_type == ContextItemType.WORKING_MEMORY
        ]
        self.assertEqual(len(working_items), 2)
        self.assertEqual(
            [item.content for item in working_items],
            ["Do you know your name?", "I don't have a name."],
        )
        self.assertEqual(window.metadata["working_memory_session_id"], "session-a")

    def test_pressure_relevance_v2_always_includes_current_event(self) -> None:
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                max_items_total=2,
                include_policy_summary=False,
                include_signal_summaries=False,
            )
        )
        window = assembler.assemble(
            event=Event(event_type=EventType.USER_MESSAGE, content="event text"),
            state=NexusState(),
        )
        self.assertEqual(window.strategy, PRESSURE_RELEVANCE_V2)
        self.assertGreaterEqual(len(window.items), 1)
        self.assertEqual(window.items[0].item_type, ContextItemType.EVENT)
        self.assertEqual(window.metadata["context_strategy"], PRESSURE_RELEVANCE_V2)

    def test_pressure_relevance_v2_reports_budget_evictions(self) -> None:
        event = self.make_event("What should I do next about Fullerene?")
        goal_store = TrackingGoalStore(
            [
                Goal(
                    id=f"goal-{index}",
                    description=f"goal {index}",
                    priority=0.9 - (index * 0.1),
                    status=GoalStatus.ACTIVE,
                    source=GoalSource.USER,
                )
                for index in range(6)
            ]
        )
        assembler = DynamicContextAssembler(
            goal_store=goal_store,
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                max_items_total=2,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )
        window = assembler.assemble(event=event, state=NexusState())
        self.assertEqual(window.metadata["context_budget"], 2)
        self.assertEqual(window.metadata["budget_used"], 2)
        excluded = window.metadata["excluded_context_items"]
        self.assertTrue(any(item.get("reason") == "budget_evicted" for item in excluded))

    def test_pressure_relevance_v2_includes_high_intensity_lpb_entry(self) -> None:
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        state = NexusState(
            facet_state={
                "signals": {
                    "latent_pressure": {
                        "last_result": {
                            "top_entries": [
                                {
                                    "id": "lpb-1",
                                    "entry_type": "verifier_failure",
                                    "description": "Verification failed repeatedly",
                                    "intensity": 0.91,
                                    "retrigger_count": 2,
                                    "status": "active",
                                    "source": "verifier",
                                }
                            ]
                        }
                    }
                }
            }
        )
        window = assembler.assemble(event=self.make_event(), state=state)
        self.assertIn("lpb:lpb-1", window.metadata["included_lpb_entry_ids"])

    def test_pressure_relevance_v2_excludes_low_pressure_lpb_entry(self) -> None:
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        state = NexusState(
            facet_state={
                "signals": {
                    "latent_pressure": {
                        "last_result": {
                            "top_entries": [
                                {
                                    "id": "lpb-low",
                                    "entry_type": "unresolved_query",
                                    "description": "Minor unresolved query",
                                    "intensity": 0.05,
                                    "retrigger_count": 0,
                                    "status": "active",
                                    "source": "nexus",
                                }
                            ]
                        }
                    }
                }
            }
        )
        window = assembler.assemble(event=self.make_event(), state=state)
        self.assertNotIn("lpb:lpb-low", window.metadata["included_lpb_entry_ids"])
        self.assertTrue(
            any(
                row.get("id") == "lpb:lpb-low" and row.get("reason") == "low_pressure"
                for row in window.metadata["excluded_context_items"]
            )
        )

    def test_pressure_relevance_v2_lpb_pressure_bypasses_relevance_cutoff(self) -> None:
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                min_relevance_score=0.95,
                min_pressure_score=0.20,
                include_policy_summary=False,
            ),
        )
        state = NexusState(
            facet_state={
                "signals": {
                    "latent_pressure": {
                        "last_result": {
                            "top_entries": [
                                {
                                    "id": "lpb-bypass",
                                    "entry_type": "policy_block",
                                    "description": "Action blocked by policy",
                                    "intensity": 0.85,
                                    "retrigger_count": 1,
                                    "status": "active",
                                    "source": "policy",
                                }
                            ]
                        }
                    }
                }
            }
        )
        window = assembler.assemble(event=self.make_event(), state=state)
        self.assertIn("lpb:lpb-bypass", window.metadata["included_lpb_entry_ids"])

    def test_pressure_relevance_v2_includes_attention_broadcast_without_duplication(self) -> None:
        event = Event(event_type=EventType.USER_MESSAGE, content="What should I do next?")
        broadcast = AttentionBroadcast(
            id="attention-broadcast:goal-focus",
            created_at=event.timestamp,
            item_id="goal:goal-1",
            source=AttentionSource.GOAL,
            source_id="goal-1",
            content="finish Fullerene",
            score=0.6,
            mode=AttentionMode.TOP_DOWN,
            components={"goal_priority": 0.4},
            recipients=["context"],
            repeated_count=2,
            pressure_contribution=0.2,
        )
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False)
        )
        window = assembler.assemble(
            event=event,
            state=NexusState(
                facet_state={"attention": {"last_attention_broadcast": broadcast.to_dict()}}
            ),
        )
        attention_ids = window.metadata["included_attention_ids"]
        self.assertEqual(len(attention_ids), 1)
        self.assertTrue(attention_ids[0].startswith("attention:attention-broadcast:goal-focus"))

    def test_pressure_relevance_v2_memory_retrieval_excludes_working_layer_records(self) -> None:
        working_record = MemoryRecord(
            id="memory-working",
            memory_type=MemoryType.WORKING,
            content="working-layer turn should not appear as long-term memory",
        )
        episodic_record = MemoryRecord(
            id="memory-episodic",
            memory_type=MemoryType.EPISODIC,
            content="episodic long-term candidate",
            salience=0.8,
        )
        memory_store = HybridTrackingMemoryStore(
            records=[episodic_record, working_record],
            hybrid_pairs=[
                (working_record, {"total": 0.99}),
                (episodic_record, {"total": 0.75}),
            ],
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        window = assembler.assemble(event=self.make_event(), state=NexusState())
        self.assertIn("memory-episodic", window.metadata["included_memory_ids"])
        self.assertNotIn("memory-working", window.metadata["included_memory_ids"])

    def test_pressure_relevance_v2_includes_high_score_memory(self) -> None:
        episodic_record = MemoryRecord(
            id="memory-high",
            memory_type=MemoryType.EPISODIC,
            content="high-score relevant episodic memory",
            salience=0.9,
            confidence=0.9,
        )
        memory_store = HybridTrackingMemoryStore(
            records=[episodic_record],
            hybrid_pairs=[(episodic_record, {"total": 0.95})],
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                min_relevance_score=0.20,
                include_policy_summary=False,
            ),
        )
        window = assembler.assemble(event=self.make_event(), state=NexusState())
        self.assertIn("memory-high", window.metadata["included_memory_ids"])

    def test_pressure_relevance_v2_excludes_low_score_memory_under_cutoff(self) -> None:
        low_record = MemoryRecord(
            id="memory-low",
            memory_type=MemoryType.EPISODIC,
            content="low-score memory",
            salience=0.0,
            confidence=0.0,
        )
        memory_store = HybridTrackingMemoryStore(
            records=[low_record],
            hybrid_pairs=[(low_record, {"total": 0.01})],
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                min_relevance_score=0.40,
                include_policy_summary=False,
            ),
        )
        window = assembler.assemble(event=self.make_event(), state=NexusState())
        self.assertNotIn("memory-low", window.metadata["included_memory_ids"])
        self.assertTrue(
            any(
                row.get("id") == "memory-low"
                and row.get("reason") == "below_relevance_cutoff"
                for row in window.metadata["excluded_context_items"]
            )
        )

    def test_pressure_relevance_v2_includes_active_high_priority_goal(self) -> None:
        goal_store = TrackingGoalStore(
            [
                Goal(
                    id="goal-high",
                    description="Ship Context v2",
                    priority=0.95,
                    status=GoalStatus.ACTIVE,
                    source=GoalSource.USER,
                )
            ]
        )
        assembler = DynamicContextAssembler(
            goal_store=goal_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        window = assembler.assemble(event=self.make_event(), state=NexusState())
        self.assertIn("goal-high", window.metadata["included_goal_ids"])

    def test_pressure_relevance_v2_includes_relevant_contradicted_belief(self) -> None:
        world_store = TrackingWorldModelStore(
            [
                Belief(
                    id="belief-risk",
                    claim="Do not execute without approval",
                    confidence=0.2,
                    status=BeliefStatus.CONTRADICTED,
                    tags=["approval", "execute"],
                    source=BeliefSource.SYSTEM,
                )
            ]
        )
        assembler = DynamicContextAssembler(
            world_model_store=world_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        window = assembler.assemble(
            event=self.make_event("Should we execute this action now?"),
            state=NexusState(),
        )
        self.assertIn("belief-risk", window.metadata["included_belief_ids"])

    def test_context_belief_metadata_includes_contradiction_counts(self) -> None:
        world_store = TrackingWorldModelStore(
            [
                Belief(
                    id="belief-ctr",
                    claim="The endpoint is stable",
                    confidence=0.25,
                    status=BeliefStatus.CONTRADICTED,
                    source=BeliefSource.SYSTEM,
                    metadata={"note": "test"},
                )
            ]
        )
        # mimic v1 counters present on model instance
        world_store.beliefs[0].contradiction_count = 3  # type: ignore[attr-defined]
        world_store.beliefs[0].support_count = 2  # type: ignore[attr-defined]
        assembler = DynamicContextAssembler(
            world_model_store=world_store,
            config=ContextAssemblyConfig(include_policy_summary=False, include_signal_summaries=False),
        )
        window = assembler.assemble(event=self.make_event("Is endpoint stable?"), state=NexusState())
        belief_items = [item for item in window.items if item.item_type == ContextItemType.BELIEF]
        self.assertEqual(len(belief_items), 1)
        self.assertEqual(belief_items[0].metadata.get("contradiction_count"), 3)

    def test_pressure_relevance_v2_includes_policy_summary_for_approval_required(self) -> None:
        policy_store = TrackingPolicyStore(
            [
                PolicyRule(
                    id="policy-approval",
                    name="Approval required for shell",
                    rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                    target_type=PolicyTargetType.SHELL,
                    target="*",
                    source=PolicySource.SYSTEM,
                )
            ]
        )
        assembler = DynamicContextAssembler(
            policy_store=policy_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2),
        )
        state = NexusState(facet_state={"nexus": {"policy_requires_approval": True}})
        window = assembler.assemble(event=self.make_event("Run command"), state=state)
        self.assertIn("policy-summary", window.metadata["included_policy_ids"])

    def test_derive_reference_anchors_empty_working_memory(self) -> None:
        continuity = derive_reference_anchors("what about that?", [])
        self.assertEqual(continuity.reference_anchors, [])
        self.assertIn("that", continuity.unresolved_references)

    def test_derive_reference_anchors_referential_message_creates_anchor(self) -> None:
        turns = [
            {
                "id": "u1",
                "content": "Can you suggest a name for this helper?",
                "metadata": {"dialogue_role": "user"},
            },
            {
                "id": "a1",
                "content": "A good option is \"context anchor\".",
                "metadata": {"dialogue_role": "assistant"},
            },
        ]
        continuity = derive_reference_anchors("Let's use that one.", turns)
        self.assertGreaterEqual(len(continuity.reference_anchors), 1)
        self.assertEqual(continuity.reference_anchors[0].surface_form, "that")

    def test_derive_reference_anchors_recent_candidates_score_higher(self) -> None:
        turns = [
            {"id": "a-old", "content": "Try the task board.", "metadata": {"dialogue_role": "assistant"}},
            {"id": "a-new", "content": "Try the release checklist.", "metadata": {"dialogue_role": "assistant"}},
        ]
        continuity = derive_reference_anchors("Let's do that.", turns)
        anchors = continuity.reference_anchors
        self.assertGreaterEqual(len(anchors), 1)
        self.assertEqual(anchors[0].referent_source_turn_id, "a-new")

    def test_derive_reference_anchors_quoted_phrase_scores_high(self) -> None:
        turns = [
            {"id": "a1", "content": 'Set the label to "final answer".', "metadata": {"dialogue_role": "assistant"}}
        ]
        continuity = derive_reference_anchors("Use that.", turns)
        self.assertGreaterEqual(len(continuity.reference_anchors), 1)
        self.assertGreaterEqual(continuity.reference_anchors[0].confidence, 0.8)

    def test_derive_reference_anchors_bounded_and_sorted(self) -> None:
        turns = [
            {"id": f"t{i}", "content": f'Try "{i} item".', "metadata": {"dialogue_role": "assistant"}}
            for i in range(10)
        ]
        continuity = derive_reference_anchors("Use that one there.", turns, max_anchors=3)
        self.assertLessEqual(len(continuity.reference_anchors), 3)
        confidences = [anchor.confidence for anchor in continuity.reference_anchors]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_derive_reference_anchors_topic_terms_and_hint(self) -> None:
        turns = [
            {"id": "u1", "content": "Can we rename the helper label?", "metadata": {"dialogue_role": "user"}},
            {"id": "a1", "content": "Yes, the helper label can be concise.", "metadata": {"dialogue_role": "assistant"}},
        ]
        continuity = derive_reference_anchors("that", turns)
        self.assertTrue(continuity.topic_terms)
        self.assertIsNotNone(continuity.current_topic_hint)

    def test_pressure_relevance_v2_metadata_includes_reference_anchor_fields(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_working_turn(
            content='Use the label "release note".',
            session_id="session-x",
            turn_index=1,
            dialogue_role="assistant",
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        window = assembler.assemble(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="Use that one.",
                metadata={"session_id": "session-x"},
            ),
            state=NexusState(),
        )
        self.assertIn("reference_anchors", window.metadata)
        self.assertIn("continuity_confidence", window.metadata)
        self.assertIn("reference_anchor_count", window.metadata)

    def test_self_editing_v3_emits_predictive_consolidation_and_pressure_metadata(self) -> None:
        event = self.make_event("plan next steps")
        state = NexusState(
            facet_state={
                "memory": {
                    "last_context_memory_communities": [
                        {"community_id": "c1", "activation_score": 0.9, "pressure_score": 0.6},
                        {"community_id": "c2", "activation_score": 0.8, "pressure_score": 0.5},
                    ]
                },
                "planner": {"last_relevant_goal_ids": ["goal-1"]},
            }
        )
        assembler = DynamicContextAssembler(
            config=ContextAssemblyConfig(
                strategy=SELF_EDITING_V3,
                max_items_total=12,
                max_working_memory_turns=0,
                include_policy_summary=False,
                include_signal_summaries=False,
            )
        )
        window = assembler.assemble(event=event, state=state)
        self.assertEqual(window.strategy, SELF_EDITING_V3)
        self.assertIn("context_pressure", window.metadata)
        self.assertIn("predictive_context_items", window.metadata)
        self.assertIn("context_item_lifecycle", window.metadata)

    def test_context_facet_state_updates_include_last_reference_anchors(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_working_turn(
            content="Set a new branch name.",
            session_id="session-y",
            turn_index=1,
            dialogue_role="assistant",
        )
        facet = ContextFacet(
            memory_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Do that.",
                metadata={"session_id": "session-y"},
            ),
            NexusState(),
        )
        self.assertIn("last_reference_anchors", result.state_updates)
        self.assertIn("last_reference_anchor_count", result.state_updates)

    def test_conversation_continuity_item_protected_for_referential_message(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_working_turn(
            content="I can call it a working anchor.",
            session_id="session-z",
            turn_index=1,
            dialogue_role="assistant",
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(
                strategy=PRESSURE_RELEVANCE_V2,
                max_items_total=3,
                include_policy_summary=False,
            ),
        )
        window = assembler.assemble(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="Let's use that.",
                metadata={"session_id": "session-z"},
            ),
            state=NexusState(),
        )
        continuity_items = [item for item in window.items if item.item_type == ContextItemType.CONVERSATION_CONTINUITY]
        self.assertTrue(continuity_items)

    def test_reference_anchor_derivation_ignores_other_session_working_memory(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        memory_store.add_working_turn(
            content="Use a session one label.",
            session_id="session-one",
            turn_index=1,
            dialogue_role="assistant",
        )
        memory_store.add_working_turn(
            content="Use another session label.",
            session_id="session-two",
            turn_index=1,
            dialogue_role="assistant",
        )
        assembler = DynamicContextAssembler(
            memory_store=memory_store,
            config=ContextAssemblyConfig(strategy=PRESSURE_RELEVANCE_V2, include_policy_summary=False),
        )
        window = assembler.assemble(
            event=Event(
                event_type=EventType.USER_MESSAGE,
                content="Use that.",
                metadata={"session_id": "session-one"},
            ),
            state=NexusState(),
        )
        anchors = window.metadata.get("reference_anchors", [])
        joined = " ".join(str(a.get("referent_text", "")) for a in anchors if isinstance(a, dict)).lower()
        self.assertNotIn("another session label", joined)


class ContextFacetTests(unittest.TestCase):
    def test_returns_empty_context_without_memory_store(self) -> None:
        facet = ContextFacet(None, window_size=4, strategy="static")

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="show context"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision.value, "wait")
        self.assertEqual(result.metadata["item_count"], 0)
        self.assertEqual(result.metadata["strategy"], STATIC_RECENT_EPISODIC_V0)
        self.assertEqual(result.metadata["max_items"], 4)
        self.assertEqual(result.metadata["source_types"], [])
        self.assertEqual(result.metadata["context_window"]["items"], [])
        self.assertIn("empty static context window", result.summary.lower())

    def test_returns_recent_episodic_items_with_memory_store(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        store.add_memory(
            MemoryRecord(
                id="episodic-1",
                created_at=utcnow() - timedelta(hours=2),
                memory_type=MemoryType.EPISODIC,
                content="first episodic memory",
                source_event_id="event-1",
                tags=["memory"],
            )
        )
        store.add_memory(
            MemoryRecord(
                id="semantic-1",
                created_at=utcnow() - timedelta(hours=1),
                memory_type=MemoryType.SEMANTIC,
                content="semantic memory",
                source_event_id="event-2",
                tags=["fact"],
            )
        )
        store.add_memory(
            MemoryRecord(
                id="episodic-2",
                created_at=utcnow(),
                memory_type=MemoryType.EPISODIC,
                content="second episodic memory",
                source_event_id="event-3",
                tags=["memory"],
            )
        )
        facet = ContextFacet(store, window_size=5, strategy="static")

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="show context"),
            NexusState(),
        )

        items = result.metadata["context_window"]["items"]
        self.assertEqual([item["id"] for item in items], ["episodic-2", "episodic-1"])
        self.assertEqual(result.metadata["item_count"], 2)
        self.assertEqual(result.metadata["source_types"], ["memory"])
        self.assertEqual(result.metadata["strategy"], STATIC_RECENT_EPISODIC_V0)

    def test_dynamic_strategy_returns_event_and_available_state(self) -> None:
        facet = ContextFacet(
            None,
            goal_store=TrackingGoalStore(
                [
                    Goal(
                        id="goal-1",
                        description="finish Fullerene",
                        priority=0.8,
                        status=GoalStatus.ACTIVE,
                        source=GoalSource.USER,
                    )
                ]
            ),
            config=ContextAssemblyConfig(
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="What should I do next?"),
            NexusState(),
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(result.metadata["strategy"], DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(result.metadata["item_count"], 2)
        self.assertEqual(result.metadata["included_goal_ids"], ["goal-1"])
        self.assertEqual(result.metadata["deduped_goal_count"], 0)
        self.assertEqual(result.metadata["limits"]["max_goals"], 3)
        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)

    def test_dynamic_context_includes_attention_broadcast_item(self) -> None:
        facet = ContextFacet(
            None,
            config=ContextAssemblyConfig(
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )
        event = Event(event_type=EventType.USER_MESSAGE, content="What should I do next?")
        broadcast = AttentionBroadcast(
            id="attention-broadcast:event-prior",
            created_at=event.timestamp,
            item_id="goal:goal-1",
            source=AttentionSource.GOAL,
            source_id="goal-1",
            content="finish Fullerene",
            score=0.4,
            mode=AttentionMode.TOP_DOWN,
            components={"goal_priority": 0.25, "pressure": 0.15},
            metadata={"normalized_content": "finish fullerene"},
            recipients=["context"],
            repeated_count=1,
            pressure_contribution=0.05,
        )

        result = facet.process(
            event,
            NexusState(
                facet_state={
                    "attention": {
                        "last_attention_broadcast": broadcast.to_dict(),
                    }
                }
            ),
        )

        attention_items = [
            item
            for item in result.metadata["context_window"]["items"]
            if item["item_type"] == "attention"
        ]
        self.assertEqual(len(attention_items), 1)
        self.assertEqual(attention_items[0]["content"], "finish Fullerene")
        self.assertEqual(
            attention_items[0]["metadata"]["attention_mode"],
            AttentionMode.TOP_DOWN.value,
        )

    def test_dynamic_context_does_not_duplicate_current_event_broadcast(self) -> None:
        facet = ContextFacet(
            None,
            config=ContextAssemblyConfig(
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="What should I do next?",
            event_id="event-current",
        )
        broadcast = AttentionBroadcast(
            id="attention-broadcast:event-current",
            created_at=event.timestamp,
            item_id="event:event-current",
            source=AttentionSource.EVENT,
            source_id="event-current",
            content=event.content,
            score=0.2,
            mode=AttentionMode.BOTTOM_UP,
            components={"novelty": 0.15},
            metadata={"normalized_content": "what should i do next?"},
            recipients=["context"],
        )

        result = facet.process(
            event,
            NexusState(
                facet_state={
                    "attention": {
                        "last_attention_broadcast": broadcast.to_dict(),
                    }
                }
            ),
        )

        attention_items = [
            item
            for item in result.metadata["context_window"]["items"]
            if item["item_type"] == "attention"
        ]
        self.assertEqual(attention_items, [])


class ContextRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_context_and_echo_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(root / "world.sqlite3")
        policy_store = SQLitePolicyStore(root / "policy.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="prior-memory",
                created_at=utcnow() - timedelta(minutes=5),
                memory_type=MemoryType.EPISODIC,
                content="prior episodic memory",
                source_event_id="event-prior",
                tags=["memory"],
            )
        )
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="finish Fullerene",
                priority=0.8,
                status=GoalStatus.ACTIVE,
                source=GoalSource.USER,
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="SQLite is the canonical memory store.",
                confidence=0.9,
                source=BeliefSource.SYSTEM,
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    memory_store,
                    goal_store=goal_store,
                    world_model_store=world_store,
                    policy_store=policy_store,
                    config=ContextAssemblyConfig(max_memories=2),
                ),
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store),
                WorldModelFacet(world_store),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="current message")
        )

        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["context", "memory", "goals", "world_model", "echo"],
        )
        context_result = record.facet_results[0]
        self.assertEqual(context_result.metadata["strategy"], DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(
            record.metadata["phase_execution_order"][0]["facets"],
            ["context", "memory"],
        )
        context_items = context_result.metadata["context_window"]["items"]
        self.assertEqual(context_items[0]["item_type"], "event")
        self.assertIn("goal-1", context_result.metadata["included_goal_ids"])
        self.assertIn("belief-1", context_result.metadata["included_belief_ids"])
        self.assertIn("prior-memory", context_result.metadata["included_memory_ids"])
        self.assertTrue((root / "memory.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())

    def test_model_prompt_builder_includes_active_goals_from_context(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="finish Fullerene",
                priority=0.8,
                status=GoalStatus.ACTIVE,
                source=GoalSource.USER,
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    memory_store,
                    goal_store=goal_store,
                    config=ContextAssemblyConfig(),
                ),
                GoalsFacet(goal_store),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="What should I do next?")
        )

        prompt = _build_model_prompt(
            record,
            {"query_intent": "planning_request", "response_template": "next_steps_available"},
        )

        self.assertIn("Current working context:", prompt)
        self.assertIn("- current event: What should I do next?", prompt)
        self.assertIn("- active goals: finish Fullerene (priority 0.8)", prompt)

    def test_model_prompt_builder_does_not_repeat_duplicate_goals(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="finish Fullerene",
                priority=0.9,
                status=GoalStatus.ACTIVE,
                source=GoalSource.USER,
            )
        )
        goal_store.add_goal(
            Goal(
                id="goal-2",
                description="remember to finish Fullerene",
                priority=0.8,
                status=GoalStatus.ACTIVE,
                source=GoalSource.USER,
            )
        )
        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    memory_store,
                    goal_store=goal_store,
                    config=ContextAssemblyConfig(),
                ),
                GoalsFacet(goal_store),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(event_type=EventType.USER_MESSAGE, content="What should I do next?")
        )

        prompt = _build_model_prompt(
            record,
            {"query_intent": "planning_request", "response_template": "next_steps_available"},
        )

        self.assertEqual(prompt.count("finish Fullerene (priority 0.9)"), 1)
        self.assertNotIn("remember to finish Fullerene", prompt)


class CLIContextIntegrationTests(unittest.TestCase):
    def test_cli_with_context_runs_without_error(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--context",
                    "--content",
                    "show context",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        context_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "context"
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(context_result["metadata"]["strategy"], DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(context_result["metadata"]["item_count"], 1)
        self.assertEqual(
            context_result["metadata"]["context_window"]["items"][0]["item_type"],
            "event",
        )
        self.assertTrue((root / "memory.sqlite3").exists())

    def test_cli_memory_and_context_can_load_recent_episodic_records_on_later_run(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        for content in ("first context memory", "second context memory"):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--json",
                        "--memory",
                        "--context",
                        "--content",
                        content,
                        "--state-dir",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)

        final_stdout = io.StringIO()
        with redirect_stdout(final_stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--memory",
                    "--context",
                    "--context-window-size",
                    "2",
                    "--content",
                    "show recent context",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(final_stdout.getvalue())
        context_result = next(
            result for result in payload["facet_results"] if result["facet_name"] == "context"
        )
        context_items = [
            item
            for item in context_result["metadata"]["context_window"]["items"]
            if item["item_type"] == "memory"
        ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(context_result["metadata"]["strategy"], DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(
            [item["content"] for item in context_items],
            ["second context memory", "first context memory"],
        )
        self.assertEqual(context_result["metadata"]["limits"]["max_memories"], 2)
        self.assertIn("memory", context_result["metadata"]["source_types"])

    def test_cli_full_persisted_goal_appears_in_later_context_and_prompt(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        for content in (
            "I should remember to finish Fullerene",
            "remember to finish Fullerene",
        ):
            with redirect_stdout(io.StringIO()):
                exit_code = cli_main(
                    [
                        "--full",
                        "--json",
                        "--content",
                        content,
                        "--state-dir",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)

        debug_stdout = io.StringIO()
        with redirect_stdout(debug_stdout):
            exit_code = cli_main(
                [
                    "--full",
                    "--json",
                    "--content",
                    "What should I do next?",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(exit_code, 0)
        second_payload = json.loads(debug_stdout.getvalue())
        context_result = next(
            result for result in second_payload["facet_results"] if result["facet_name"] == "context"
        )
        goal_items = [
            item
            for item in context_result["metadata"]["context_window"]["items"]
            if item["item_type"] == "goal"
        ]
        self.assertEqual(len(goal_items), 1)
        self.assertEqual(
            normalize_goal_description(goal_items[0]["content"]),
            "finish fullerene",
        )

        captured_prompts: list[str] = []

        def capture_prompt(prompt_text: str) -> str:
            captured_prompts.append(prompt_text)
            return "Model phrasing only."

        with patch(
            "fullerene.models.ollama.OllamaAdapter.generate",
            side_effect=capture_prompt,
        ):
            model_stdout = io.StringIO()
            with redirect_stdout(model_stdout):
                exit_code = cli_main(
                    [
                        "--full",
                        "--model",
                        "ollama:gemma3:4b",
                        "--content",
                        "What should I do next?",
                        "--state-dir",
                        str(root),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("- active goals: finish Fullerene", captured_prompts[0])
        self.assertIn("- current event: What should I do next?", captured_prompts[0])
        active_goals_line = next(
            line
            for line in captured_prompts[0].splitlines()
            if line.startswith("- active goals:")
        )
        self.assertEqual(active_goals_line.count("finish Fullerene"), 1)


if __name__ == "__main__":
    unittest.main()
