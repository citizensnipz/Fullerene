"""Deterministic goal normalization and deduplication helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from fullerene.goals.models import Goal, GoalStatus

GOAL_NEAR_DUPLICATE_OVERLAP_THRESHOLD = 0.85
INTENT_PREFIXES = (
    "i should remember to",
    "dont forget to",
    "don't forget to",
    "remember to",
    "remember that",
    "make sure i",
    "make sure we",
    "make sure",
    "i should",
    "we should",
    "i need to",
    "we need to",
)
GOAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "i",
        "is",
        "my",
        "of",
        "our",
        "that",
        "the",
        "to",
        "we",
    }
)


@dataclass(slots=True)
class GoalDeduplicationResult:
    goals: list[Goal]
    deduped_goal_ids: list[str]
    normalized_goal_keys: list[str]

    @property
    def deduped_goal_count(self) -> int:
        return len(self.deduped_goal_ids)


def normalize_goal_description(description: str) -> str:
    """Return the deterministic normalized key for a goal description."""
    cleaned = _normalize_goal_text(description)
    for prefix in INTENT_PREFIXES:
        if cleaned == prefix:
            return ""
        prefix_with_space = f"{prefix} "
        if cleaned.startswith(prefix_with_space):
            cleaned = cleaned[len(prefix_with_space) :].strip()
            break
    return cleaned


def goal_keyword_tokens(description: str) -> set[str]:
    normalized = normalize_goal_description(description)
    raw_tokens = [token for token in normalized.split() if token]
    filtered_tokens = [
        _canonical_goal_token(token)
        for token in raw_tokens
        if token not in GOAL_STOPWORDS
    ]
    filtered_tokens = [token for token in filtered_tokens if token]
    if filtered_tokens:
        return set(filtered_tokens)
    return {
        _canonical_goal_token(token)
        for token in raw_tokens
        if _canonical_goal_token(token)
    }


def goal_keyword_overlap(left: str, right: str) -> float:
    left_tokens = goal_keyword_tokens(left)
    right_tokens = goal_keyword_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared_tokens = left_tokens & right_tokens
    return len(shared_tokens) / max(len(left_tokens), len(right_tokens))


def find_matching_active_goal(
    goals: Sequence[Goal],
    description: str,
    *,
    overlap_threshold: float = GOAL_NEAR_DUPLICATE_OVERLAP_THRESHOLD,
) -> Goal | None:
    normalized_description = normalize_goal_description(description)
    best_match: Goal | None = None
    best_overlap = 0.0

    for goal in _sort_goals_for_preference(goals):
        if goal.status != GoalStatus.ACTIVE:
            continue
        if normalize_goal_description(goal.description) == normalized_description:
            return goal
        overlap = goal_keyword_overlap(description, goal.description)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = goal

    if best_overlap >= overlap_threshold:
        return best_match
    return None


def dedupe_active_goals(
    goals: Sequence[Goal],
    *,
    limit: int | None = None,
    overlap_threshold: float = GOAL_NEAR_DUPLICATE_OVERLAP_THRESHOLD,
) -> GoalDeduplicationResult:
    selected: list[Goal] = []
    selected_keys: list[str] = []
    deduped_goal_ids: list[str] = []

    for goal in _sort_goals_for_preference(goals):
        if goal.status != GoalStatus.ACTIVE:
            continue
        goal_key = normalize_goal_description(goal.description)
        is_duplicate = False
        for selected_goal, selected_key in zip(selected, selected_keys):
            if goal_key == selected_key:
                is_duplicate = True
                break
            if goal_keyword_overlap(goal.description, selected_goal.description) >= overlap_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            deduped_goal_ids.append(goal.id)
            continue
        selected.append(goal)
        selected_keys.append(goal_key)

    if limit is not None:
        selected = selected[: max(int(limit), 0)]
        selected_keys = selected_keys[: max(int(limit), 0)]

    return GoalDeduplicationResult(
        goals=selected,
        deduped_goal_ids=deduped_goal_ids,
        normalized_goal_keys=selected_keys,
    )


def _sort_goals_for_preference(goals: Sequence[Goal]) -> list[Goal]:
    return sorted(
        goals,
        key=lambda goal: (
            goal.priority,
            goal.updated_at,
            goal.created_at,
            goal.id,
        ),
        reverse=True,
    )


def _normalize_goal_text(description: str) -> str:
    cleaned = str(description or "").strip().lower()
    cleaned = cleaned.replace("’", "").replace("'", "")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _canonical_goal_token(token: str) -> str:
    cleaned = token.strip().lower()
    if len(cleaned) <= 3:
        return cleaned
    if cleaned.endswith("ing") and len(cleaned) > 6:
        return cleaned[:-3]
    if cleaned.endswith("s") and len(cleaned) > 4 and not cleaned.endswith("ss"):
        return cleaned[:-1]
    return cleaned
