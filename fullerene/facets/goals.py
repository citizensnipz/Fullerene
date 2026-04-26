"""Deterministic goals facet for Fullerene Goals v0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fullerene.goals import Goal, GoalStore, SQLiteGoalStore
from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, FacetResult, NexusState


@dataclass(slots=True)
class _GoalMatch:
    goal: Goal
    score: float
    tag_overlap: float
    keyword_overlap: float
    shared_tags: list[str]
    shared_keywords: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.goal.id,
            "description": self.goal.description,
            "priority": self.goal.priority,
            "score": round(self.score, 3),
            "tag_overlap": round(self.tag_overlap, 3),
            "keyword_overlap": round(self.keyword_overlap, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "source": self.goal.source.value,
        }


class GoalsFacet:
    """Expose deterministic goal relevance signals without executing actions."""

    name = "goals"

    def __init__(
        self,
        store: GoalStore,
        *,
        active_limit: int = 10,
        relevant_limit: int = 3,
    ) -> None:
        self.store = store
        self.active_limit = max(int(active_limit), 1)
        self.relevant_limit = max(int(relevant_limit), 1)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        active_limit: int = 10,
        relevant_limit: int = 3,
    ) -> "GoalsFacet":
        return cls(
            SQLiteGoalStore(path),
            active_limit=active_limit,
            relevant_limit=relevant_limit,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state

        active_goals = self.store.list_active_goals(limit=self.active_limit)
        if not active_goals:
            return FacetResult(
                facet_name=self.name,
                summary="Goals facet found no active goals.",
                state_updates={
                    "last_active_goal_ids": [],
                    "last_relevant_goals": [],
                    "last_relevance_score": 0.0,
                },
                metadata={
                    "active_goal_count": 0,
                    "relevant_goals": [],
                    "relevance_score": 0.0,
                    "score_formula": "tag_overlap + keyword_overlap + priority",
                },
            )

        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        relevant_matches = [
            match
            for goal in active_goals
            if (match := self._score_goal(goal, event_tags, event_keywords)) is not None
        ]
        relevant_matches.sort(
            key=lambda match: (
                match.score,
                match.goal.priority,
                match.goal.updated_at.timestamp(),
                match.goal.id,
            ),
            reverse=True,
        )
        relevant_matches = relevant_matches[: self.relevant_limit]
        relevance_score = (
            round(relevant_matches[0].score, 3) if relevant_matches else 0.0
        )

        if relevant_matches:
            summary = (
                f"Goals facet matched {len(relevant_matches)} active goals; "
                f"top relevance score {relevance_score:.3f}."
            )
        else:
            summary = (
                f"Goals facet checked {len(active_goals)} active goals and "
                "found no relevant matches."
            )

        relevant_goal_payload = [match.to_dict() for match in relevant_matches]
        return FacetResult(
            facet_name=self.name,
            summary=summary,
            state_updates={
                "last_active_goal_ids": [goal.id for goal in active_goals],
                "last_relevant_goals": relevant_goal_payload,
                "last_relevance_score": relevance_score,
            },
            metadata={
                "active_goal_count": len(active_goals),
                "relevant_goals": relevant_goal_payload,
                "relevance_score": relevance_score,
                "event_tags": sorted(event_tags),
                "event_keywords": sorted(event_keywords),
                "score_formula": "tag_overlap + keyword_overlap + priority",
            },
        )

    @staticmethod
    def _score_goal(
        goal: Goal,
        event_tags: set[str],
        event_keywords: set[str],
    ) -> _GoalMatch | None:
        goal_tags = set(goal.tags)
        goal_keywords = tokenize(goal.description)
        shared_tags = sorted(event_tags & goal_tags)
        shared_keywords = sorted(event_keywords & goal_keywords)

        if not shared_tags and not shared_keywords:
            return None

        tag_overlap = len(shared_tags) / len(goal_tags) if goal_tags else 0.0
        keyword_overlap = (
            len(shared_keywords) / len(goal_keywords) if goal_keywords else 0.0
        )
        score = tag_overlap + keyword_overlap + goal.priority

        return _GoalMatch(
            goal=goal,
            score=score,
            tag_overlap=tag_overlap,
            keyword_overlap=keyword_overlap,
            shared_tags=shared_tags,
            shared_keywords=shared_keywords,
        )
