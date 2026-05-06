"""Deterministic goals facet for Fullerene Goals v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fullerene.goals import Goal, GoalStatus, GoalStore, SQLiteGoalStore
from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, FacetResult, NexusState


def _goal_payload(goal: Goal) -> dict[str, object]:
    return {
        "id": goal.id,
        "description": goal.description,
        "priority": goal.priority,
        "status": goal.status.value,
        "tags": list(goal.tags),
        "source": goal.source.value,
        "reinforcement_score": goal.reinforcement_score,
        "activation_count": goal.activation_count,
        "last_activated_at": goal.last_activated_at.isoformat()
        if goal.last_activated_at
        else None,
        "completion_score": goal.completion_score,
        "blocked_reason": goal.blocked_reason,
        "stale_score": goal.stale_score,
    }


@dataclass(slots=True)
class _GoalMatch:
    goal: Goal
    score: float
    tag_overlap: float
    keyword_overlap: float
    shared_tags: list[str]
    shared_keywords: list[str]
    relevance_score: float
    priority_component: float
    reinforcement_component: float
    recency_component: float

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
            "status": self.goal.status.value,
            "relevance_score": round(self.relevance_score, 3),
            "priority_component": round(self.priority_component, 3),
            "reinforcement_component": round(self.reinforcement_component, 3),
            "recency_component": round(self.recency_component, 3),
            "final_score": round(self.score, 3),
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

        include_inactive = bool(
            (event.metadata or {}).get("goals_include_inactive", False)
        )
        active_goals = self.store.list_active_goals(
            limit=self.active_limit,
            include_inactive=include_inactive,
        )
        reinforcement_updates = self._reinforce_active_goals(event=event, goals=active_goals)
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
                    "goal_reinforcement_updates": reinforcement_updates,
                    "score_formula": (
                        "priority*0.45 + relevance*0.30 + "
                        "reinforcement*0.15 + recency*0.10"
                    ),
                },
            )

        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        active_goal_payload = [_goal_payload(goal) for goal in active_goals]
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
                "last_active_goals": active_goal_payload[: self.relevant_limit],
                "last_relevant_goals": relevant_goal_payload,
                "last_relevance_score": relevance_score,
            },
            metadata={
                "active_goal_count": len(active_goals),
                "active_goals": active_goal_payload[: self.relevant_limit],
                "relevant_goals": relevant_goal_payload,
                "relevance_score": relevance_score,
                "event_tags": sorted(event_tags),
                "event_keywords": sorted(event_keywords),
                "score_formula": (
                    "priority*0.45 + relevance*0.30 + "
                    "reinforcement*0.15 + recency*0.10"
                ),
                "goal_reinforcement_updates": reinforcement_updates,
                **self._goal_pressure_metadata(active_goals),
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
        relevance_score = _clamp01((tag_overlap * 0.55) + (keyword_overlap * 0.45))
        priority_component = goal.priority * 0.45
        reinforcement_component = goal.reinforcement_score * 0.15
        recency_component = _recency_component(goal) * 0.10
        score = _clamp01(
            priority_component
            + (relevance_score * 0.30)
            + reinforcement_component
            + recency_component
        )

        return _GoalMatch(
            goal=goal,
            score=score,
            tag_overlap=tag_overlap,
            keyword_overlap=keyword_overlap,
            shared_tags=shared_tags,
            shared_keywords=shared_keywords,
            relevance_score=relevance_score,
            priority_component=priority_component,
            reinforcement_component=reinforcement_component,
            recency_component=recency_component,
        )

    def _reinforce_active_goals(self, event: Event, goals: list[Goal]) -> list[dict[str, object]]:
        salience = _coerce_float((event.metadata or {}).get("salience"), 0.0)
        if salience < 0.65:
            return []
        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        updates: list[dict[str, object]] = []
        for goal in goals:
            if goal.status != GoalStatus.ACTIVE:
                continue
            relatedness = self._goal_relatedness(goal, event_tags, event_keywords, event)
            if relatedness < 0.25:
                continue
            reinforcement_weight = _clamp01(salience * relatedness * 0.25)
            goal.activation_count += 1
            goal.last_activated_at = event.timestamp
            goal.last_activated_event_id = event.event_id
            goal.last_reinforced_at = event.timestamp
            goal.reinforcement_score = _clamp01(
                goal.reinforcement_score
                + ((1.0 - goal.reinforcement_score) * reinforcement_weight)
            )
            goal.stale_score = 0.0
            self.store.update_goal(goal)
            updates.append(
                {
                    "goal_id": goal.id,
                    "event_id": event.event_id,
                    "relatedness_score": round(relatedness, 3),
                    "reinforcement_weight": round(reinforcement_weight, 3),
                    "reinforcement_score": round(goal.reinforcement_score, 3),
                    "activation_count": goal.activation_count,
                }
            )
        return updates

    @staticmethod
    def _goal_relatedness(
        goal: Goal, event_tags: set[str], event_keywords: set[str], event: Event
    ) -> float:
        goal_tags = set(goal.tags)
        goal_keywords = tokenize(goal.description)
        tag_ratio = _overlap_ratio(goal_tags, event_tags)
        keyword_ratio = _overlap_ratio(goal_keywords, event_keywords)
        goal_reference = 0.25 if (event.metadata or {}).get("goal_id") == goal.id else 0.0
        return _clamp01((tag_ratio * 0.45) + (keyword_ratio * 0.55) + goal_reference)

    @staticmethod
    def _goal_pressure_metadata(goals: list[Goal]) -> dict[str, object]:
        blocked_goal_ids: list[str] = []
        pressure_goal_ids: list[str] = []
        pressure_total = 0.0
        unresolved_goal_count = 0
        for goal in goals:
            if goal.status != GoalStatus.ACTIVE:
                continue
            unresolved_goal_count += 1
            unresolved_factor = 1.0 - goal.completion_score
            blocked_factor = 1.0 if goal.blocked_reason else 0.0
            contribution = _clamp01(
                (goal.priority * unresolved_factor * 0.5)
                + (goal.reinforcement_score * 0.25)
                + (blocked_factor * 0.25)
            )
            if contribution > 0:
                pressure_goal_ids.append(goal.id)
                pressure_total += contribution
            if goal.blocked_reason:
                blocked_goal_ids.append(goal.id)
        return {
            "goal_pressure_contribution": _clamp01(pressure_total),
            "pressure_goal_ids": pressure_goal_ids,
            "blocked_goal_ids": blocked_goal_ids,
            "unresolved_goal_count": unresolved_goal_count,
        }


def _recency_component(goal: Goal) -> float:
    if goal.last_activated_at is None:
        return 0.0
    elapsed_seconds = (goal.updated_at - goal.last_activated_at).total_seconds()
    if elapsed_seconds <= 0:
        return 1.0
    if elapsed_seconds >= 3600:
        return 0.0
    return _clamp01(1.0 - (elapsed_seconds / 3600.0))


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    common = len(left.intersection(right))
    total = len(left.union(right))
    if total <= 0:
        return 0.0
    return common / total


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
