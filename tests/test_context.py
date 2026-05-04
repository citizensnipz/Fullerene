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
    DYNAMIC_ACTIVE_FACETS_V1,
    STATIC_RECENT_EPISODIC_V0,
    ContextAssemblyConfig,
    ContextItem,
    ContextItemType,
    ContextWindow,
    DynamicContextAssembler,
    StaticContextAssembler,
)
from fullerene.facets import ContextFacet, EchoFacet, GoalsFacet, MemoryFacet, WorldModelFacet
from fullerene.goals import Goal, GoalSource, GoalStatus, SQLiteGoalStore
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
from fullerene.world_model import Belief, BeliefSource, SQLiteWorldModelStore
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


class ContextModelTests(unittest.TestCase):
    def test_context_assembly_config_defaults_are_correct(self) -> None:
        config = ContextAssemblyConfig()

        self.assertEqual(config.max_goals, 3)
        self.assertEqual(config.max_memories, 5)
        self.assertEqual(config.max_beliefs, 5)
        self.assertEqual(config.salience_threshold, 0.0)
        self.assertTrue(config.include_policy_summary)
        self.assertTrue(config.include_signal_summaries)
        self.assertEqual(config.strategy, DYNAMIC_ACTIVE_FACETS_V1)
        self.assertEqual(config.max_items, 20)

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
        self.assertEqual(goal_store.calls, [3])
        self.assertEqual(world_store.calls, [10])
        self.assertEqual(policy_store.list_policies_calls, [(20, True)])
        self.assertEqual(policy_store.count_calls, 1)

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
        self.assertEqual(result.metadata["limits"]["max_goals"], 3)
        self.assertNotEqual(result.proposed_decision, DecisionAction.ACT)


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

        first_stdout = io.StringIO()
        with redirect_stdout(first_stdout):
            exit_code = cli_main(
                [
                    "--full",
                    "--json",
                    "--content",
                    "I should remember to finish Fullerene",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(exit_code, 0)

        second_stdout = io.StringIO()
        with redirect_stdout(second_stdout):
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
        second_payload = json.loads(second_stdout.getvalue())
        context_result = next(
            result for result in second_payload["facet_results"] if result["facet_name"] == "context"
        )
        goal_items = [
            item
            for item in context_result["metadata"]["context_window"]["items"]
            if item["item_type"] == "goal"
        ]
        self.assertTrue(any("finish Fullerene" in item["content"] for item in goal_items))

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


if __name__ == "__main__":
    unittest.main()
