from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from fullerene.attention import (
    AttentionBroadcast,
    AttentionConflict,
    AttentionHistoryEntry,
    AttentionMode,
    AttentionSource,
)
from fullerene.facets import AttentionFacet
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.memory import MemoryRecord, MemoryType, SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusState
from fullerene.workspace_state import workspace_state_root
from fullerene.world_model import Belief, SQLiteWorldModelStore


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-attention-v1-{uuid4().hex}"


def carry_attention_state(state_updates: dict[str, object]) -> NexusState:
    return NexusState(facet_state={"attention": dict(state_updates)})


class AttentionV1ModelTests(unittest.TestCase):
    def test_broadcast_conflict_and_history_round_trip(self) -> None:
        event = Event(event_type=EventType.USER_MESSAGE, content="focus here")
        broadcast = AttentionBroadcast(
            id="attention-broadcast:event-1",
            created_at=event.timestamp,
            item_id="goal:goal-1",
            source_id="goal-1",
            source=AttentionSource.GOAL,
            content="Finish the task",
            score=0.4,
            mode=AttentionMode.TOP_DOWN,
            components={"goal_priority": 0.25, "pressure": 0.15},
            metadata={"normalized_content": "finish the task"},
            recipients=["context", "behavior"],
            conflict_ids=["attention-conflict:event-1"],
            repeated_count=1,
            pressure_contribution=0.05,
        )
        conflict = AttentionConflict(
            id="attention-conflict:event-1",
            item_ids=["goal:goal-1", "event:event-1"],
            score_delta=0.01,
            reason="close_score_competition",
        )
        history = AttentionHistoryEntry(
            broadcast_id=broadcast.id,
            item_id=broadcast.item_id,
            source=broadcast.source,
            source_id=broadcast.source_id,
            score=broadcast.score,
            created_at=broadcast.created_at,
            metadata={"normalized_content": "finish the task"},
        )

        self.assertEqual(AttentionBroadcast.from_dict(broadcast.to_dict()), broadcast)
        self.assertEqual(AttentionConflict.from_dict(conflict.to_dict()), conflict)
        self.assertEqual(AttentionHistoryEntry.from_dict(history.to_dict()), history)


class AttentionV1FacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_goal_heavy_candidate_is_classified_top_down(self) -> None:
        facet = AttentionFacet(top_n=1)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {"id": "goal-1", "description": "Finish task", "priority": 0.8}
                    ]
                }
            }
        )

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="what matters here?"),
            state,
        )

        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(result.metadata["broadcast_mode"], AttentionMode.TOP_DOWN.value)

    def test_novelty_heavy_candidate_is_classified_bottom_up(self) -> None:
        facet = AttentionFacet(top_n=1)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Something new happened",
                metadata={"novelty": 0.9},
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["broadcast_mode"], AttentionMode.BOTTOM_UP.value)

    def test_top_down_wins_close_score_conflict(self) -> None:
        facet = AttentionFacet(top_n=2)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {"id": "goal-1", "description": "Finish task", "priority": 0.16}
                    ]
                }
            }
        )

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="Something new happened",
                metadata={"novelty": 0.333},
            ),
            state,
        )

        self.assertTrue(result.metadata["attention_conflict"])
        self.assertEqual(result.metadata["broadcast_item_id"], "goal:goal-1")
        self.assertEqual(result.metadata["broadcast_mode"], AttentionMode.TOP_DOWN.value)
        self.assertEqual(result.metadata["focus_items"][0]["id"], "goal:goal-1")
        self.assertIn("event:", result.metadata["conflict_items"][0])
        self.assertIn("goal:goal-1", result.metadata["conflict_items"])
        self.assertLessEqual(result.metadata["score_delta"], 0.05)

    def test_history_is_bounded_and_repeated_attention_adds_pressure(self) -> None:
        facet = AttentionFacet(top_n=1, history_size=2)

        first = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="repeat this",
                metadata={"novelty": 0.9},
            ),
            NexusState(),
        )
        second = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="repeat this",
                metadata={"novelty": 0.9},
            ),
            carry_attention_state(first.state_updates),
        )
        third = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="something else",
                metadata={"novelty": 0.8},
            ),
            carry_attention_state(second.state_updates),
        )

        self.assertEqual(first.metadata["broadcast"]["repeated_count"], 0)
        self.assertEqual(second.metadata["broadcast"]["repeated_count"], 1)
        self.assertEqual(second.metadata["pressure_contribution"], 0.05)
        self.assertEqual(third.metadata["attention_history_count"], 2)
        self.assertEqual(len(third.metadata["attention_history"]), 2)
        self.assertEqual(
            [entry["broadcast_id"] for entry in third.metadata["attention_history"]],
            [second.metadata["broadcast"]["id"], third.metadata["broadcast"]["id"]],
        )

    def test_attention_broadcast_does_not_mutate_other_stores(self) -> None:
        memory_store = SQLiteMemoryStore(self.root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(self.root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(self.root / "world.sqlite3")
        memory_store.add_memory(
            MemoryRecord(
                id="mem-1",
                memory_type=MemoryType.EPISODIC,
                content="Keep memory stable",
                salience=0.5,
                confidence=1.0,
            )
        )
        goal_store.add_goal(Goal(id="goal-1", description="Keep goal stable", priority=0.6))
        world_store.add_belief(
            Belief(id="belief-1", claim="Keep belief stable", confidence=0.7)
        )
        facet = AttentionFacet(memory_store=memory_store)
        state = NexusState(
            facet_state={
                "goals": {
                    "last_relevant_goals": [
                        {"id": "goal-1", "description": "Keep goal stable", "priority": 0.6}
                    ]
                },
                "world_model": {
                    "last_relevant_beliefs": [
                        {"id": "belief-1", "claim": "Keep belief stable", "confidence": 0.7}
                    ]
                },
            }
        )

        result = facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="keep stable"),
            state,
        )

        self.assertEqual(memory_store.get_memory("mem-1").salience, 0.5)
        self.assertEqual(goal_store.get_goal("goal-1").priority, 0.6)
        self.assertEqual(world_store.get_belief("belief-1").confidence, 0.7)
        self.assertIsNotNone(result.metadata["broadcast"])


if __name__ == "__main__":
    unittest.main()
