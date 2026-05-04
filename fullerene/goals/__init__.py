"""Public goals package exports."""

from fullerene.goals.models import Goal, GoalSource, GoalStatus
from fullerene.goals.normalization import (
    GOAL_NEAR_DUPLICATE_OVERLAP_THRESHOLD,
    GoalDeduplicationResult,
    dedupe_active_goals,
    find_matching_active_goal,
    goal_keyword_overlap,
    goal_keyword_tokens,
    normalize_goal_description,
)
from fullerene.goals.store import GoalStore, SQLiteGoalStore

__all__ = [
    "Goal",
    "GoalSource",
    "GoalStatus",
    "GOAL_NEAR_DUPLICATE_OVERLAP_THRESHOLD",
    "GoalDeduplicationResult",
    "GoalStore",
    "SQLiteGoalStore",
    "dedupe_active_goals",
    "find_matching_active_goal",
    "goal_keyword_overlap",
    "goal_keyword_tokens",
    "normalize_goal_description",
]
