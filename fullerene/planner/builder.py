"""Deterministic plan construction for Fullerene Planner v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
import re
from typing import Any

from fullerene.context import ContextItem, ContextItemType, ContextWindow
from fullerene.goals import Goal, GoalStore
from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, NexusState
from fullerene.planner.models import Plan, PlanStatus, PlanStep, PlanStepStatus, RiskLevel
from fullerene.policy import (
    PolicyStatus,
    PolicyStore,
    PolicyTargetType,
    coerce_policy_target_type,
)
from fullerene.world_model import Belief, WorldModelStore

EXPLICIT_REQUEST_PHRASES = (
    "make a plan",
    "plan this",
    "break this down",
    "what are the steps",
    "what are the next steps",
    "how should we do this",
)
PLANNING_QUERY_PHRASES = (
    "what should i do next",
    "what should i focus on",
    "what next",
    "next step",
    "next steps",
    "what are the next steps",
    "how should i",
    "plan",
)
RECOMMENDATION_QUERY_PHRASES = (
    "what should i do this weekend",
    "what should i",
    "what should we",
    "recommend",
    "suggest",
    "this weekend",
)
STATUS_QUERY_PHRASES = (
    "what are you doing",
    "what do you know",
)
PLANNING_METADATA_FLAGS = ("request_plan", "allow_planning", "planning_allowed")
HIGH_PRIORITY_GOAL_THRESHOLD = 0.75
HIGH_PRESSURE_THRESHOLD = 0.7
POSITIVE_PREFERENCE_PATTERNS = (
    re.compile(r"\bi\s+(?:really\s+)?(?:enjoy|like|love|prefer)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am)\s+into\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bmy\s+favorite(?:\s+things?)?\s+(?:are|is)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
)
NEGATIVE_PREFERENCE_PATTERNS = (
    re.compile(r"\bi\s+(?:really\s+)?(?:hate|dislike|avoid)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bi\s+do(?:\s+not|n't)\s+like\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bi\s+prefer\s+not\s+to\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
)
SPLIT_PHRASE_PATTERN = re.compile(r"\s*(?:,| and | or |/)\s*", re.IGNORECASE)
CONFLICT_GROUPS = {
    "outside": {"outside", "outdoors", "outdoor"},
    "inside": {"inside", "indoors", "indoor"},
    "home": {"home", "house"},
    "travel": {"travel", "trip", "vacation"},
    "stay": {"stay", "remain"},
}
CONFLICT_PAIRS = {
    ("outside", "home"),
    ("outside", "inside"),
    ("travel", "stay"),
}
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "for",
        "go",
        "i",
        "into",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "we",
        "with",
    }
)


@dataclass(slots=True)
class _GoalCandidate:
    id: str
    description: str
    priority: float
    updated_at: datetime
    tags: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _MemoryCandidate:
    id: str
    content: str
    created_at: datetime
    salience: float
    confidence: float
    tags: list[str]
    context_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _BeliefCandidate:
    id: str
    claim: str
    confidence: float
    updated_at: datetime
    tags: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PlanningContext:
    context_window: ContextWindow | None
    current_event: str
    goals: list[_GoalCandidate]
    memories: list[_MemoryCandidate]
    beliefs: list[_BeliefCandidate]
    policy_summary: str | None
    signal_summaries: list[str]
    context_item_ids: list[str]


@dataclass(slots=True)
class _GoalMatch:
    goal: _GoalCandidate
    score: float
    shared_tags: list[str]
    shared_keywords: list[str]
    reasons: list[str]

    def to_dict(self, *, selected: bool, rank: int) -> dict[str, Any]:
        return {
            "id": self.goal.id,
            "item_type": "goal",
            "description": self.goal.description,
            "priority": round(self.goal.priority, 3),
            "score": round(self.score, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "reasons": list(self.reasons),
            "selected": selected,
            "rank": rank,
        }


@dataclass(slots=True)
class _MemoryMatch:
    memory: _MemoryCandidate
    score: float
    shared_tags: list[str]
    shared_keywords: list[str]
    preference_polarity: str | None
    preference_terms: list[str]
    reasons: list[str]

    def to_dict(self, *, selected: bool, rank: int) -> dict[str, Any]:
        return {
            "id": self.memory.id,
            "item_type": "memory",
            "content_preview": self.memory.content[:160],
            "score": round(self.score, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "preference_polarity": self.preference_polarity,
            "preference_terms": list(self.preference_terms),
            "reasons": list(self.reasons),
            "selected": selected,
            "rank": rank,
            "context_source": self.memory.context_source,
        }


@dataclass(slots=True)
class _BeliefMatch:
    belief: _BeliefCandidate
    score: float
    shared_tags: list[str]
    shared_keywords: list[str]
    reasons: list[str]

    def to_dict(self, *, selected: bool, rank: int) -> dict[str, Any]:
        return {
            "id": self.belief.id,
            "item_type": "belief",
            "claim": self.belief.claim,
            "confidence": round(self.belief.confidence, 3),
            "score": round(self.score, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "reasons": list(self.reasons),
            "selected": selected,
            "rank": rank,
        }


class DeterministicPlanBuilder:
    """Build inspectable plans without model calls or execution."""

    def __init__(
        self,
        *,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        policy_store: PolicyStore | None = None,
        state_dir: Path | str | None = None,
        active_goal_limit: int = 10,
        active_belief_limit: int = 20,
        relevant_limit: int = 3,
    ) -> None:
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.policy_store = policy_store
        self.state_dir = (
            Path(state_dir).expanduser().resolve()
            if state_dir is not None
            else Path(".").resolve()
        )
        self.active_goal_limit = max(int(active_goal_limit), 1)
        self.active_belief_limit = max(int(active_belief_limit), 1)
        self.relevant_limit = max(int(relevant_limit), 1)

    def build(self, event: Event, state: NexusState) -> Plan | None:
        explicit_request = self._is_explicit_request(event)
        query_intent = self._detect_query_intent(event.content)
        pressure, pressure_source = self._resolve_pressure(event, state)
        planning_context = self._planning_context(event, state)
        ranked_goals = self._select_goals(
            event,
            planning_context,
            query_intent=query_intent,
            explicit_request=explicit_request,
        )
        matched_goal = ranked_goals[0] if ranked_goals else None
        relevant_memories = self._select_memories(
            event,
            planning_context,
            query_intent=query_intent,
        )
        relevant_beliefs = self._select_beliefs(
            event,
            planning_context,
            matched_goal=matched_goal,
            relevant_memories=relevant_memories,
        )
        conflict_report = self._conflict_report(ranked_goals)
        trigger_reason = self._trigger_reason(
            explicit_request=explicit_request,
            query_intent=query_intent,
            matched_goal=matched_goal,
            ranked_goals=ranked_goals,
        )
        if trigger_reason is None:
            return None

        grounding_score, grounding_status, grounding_breakdown = self._grounding(
            query_intent=query_intent,
            ranked_goals=ranked_goals,
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
        )
        steps = self._build_steps(
            event,
            query_intent=query_intent,
            grounding_status=grounding_status,
            matched_goal=matched_goal,
            secondary_goals=ranked_goals[1:self.relevant_limit],
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
            conflict_report=conflict_report,
            pressure=pressure,
        )
        policy_satisfied = self._apply_policy_filters(event, state, steps)
        confidence, confidence_breakdown = self._confidence(
            explicit_request=explicit_request,
            query_intent=query_intent,
            matched_goal=matched_goal,
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
            policy_satisfied=policy_satisfied,
            grounding_score=grounding_score,
        )

        generation_mode = (
            "high_pressure_direct"
            if pressure >= HIGH_PRESSURE_THRESHOLD
            else "low_pressure_exploratory"
        )
        plan_id = self._stable_id(
            "plan",
            event.event_id,
            trigger_reason,
            query_intent or "none",
            matched_goal.goal.id if matched_goal is not None else "generic",
            grounding_status,
            f"{pressure:.3f}",
            generation_mode,
            event.content.casefold(),
        )
        steps = [
            PlanStep(
                id=self._stable_id("plan-step", plan_id, step.order, step.description),
                description=step.description,
                order=step.order,
                target_type=step.target_type,
                risk_level=step.risk_level,
                requires_approval=step.requires_approval,
                status=step.status,
                policy_status=step.policy_status,
                metadata=step.metadata,
            )
            for step in steps
        ]
        relevant_goal_ids = [
            match.goal.id for match in ranked_goals[: self.relevant_limit]
        ]
        relevant_memory_ids = [
            match.memory.id for match in relevant_memories[: self.relevant_limit]
        ]
        relevant_belief_ids = [match.belief.id for match in relevant_beliefs]
        blocked_steps = [step.id for step in steps if step.status == PlanStepStatus.BLOCKED]
        approval_required_steps = [
            step.id for step in steps if step.requires_approval
        ]
        plan_template_key = self._plan_template_key(
            query_intent=query_intent,
            grounding_status=grounding_status,
            matched_goal=matched_goal,
            relevant_memories=relevant_memories,
        )
        plan_memory_eligible = grounding_status in {
            "grounded",
            "partially_grounded",
        }

        return Plan(
            id=plan_id,
            created_at=event.timestamp,
            source_event_id=event.event_id,
            goal_id=matched_goal.goal.id if matched_goal is not None else None,
            title=self._title_for_plan(
                query_intent=query_intent,
                grounding_status=grounding_status,
                matched_goal=matched_goal,
                relevant_memories=relevant_memories,
            ),
            steps=steps,
            confidence=confidence,
            pressure=pressure,
            status=PlanStatus.PROPOSED,
            reasons=self._plan_reasons(
                explicit_request=explicit_request,
                query_intent=query_intent,
                matched_goal=matched_goal,
                relevant_memories=relevant_memories,
                relevant_beliefs=relevant_beliefs,
                policy_satisfied=policy_satisfied,
                pressure=pressure,
                grounding_status=grounding_status,
            ),
            metadata={
                "trigger_reason": trigger_reason,
                "query_intent": query_intent,
                "pressure_source": pressure_source,
                "generation_mode": generation_mode,
                "confidence_breakdown": confidence_breakdown,
                "grounding_status": grounding_status,
                "grounding_score": grounding_score,
                "grounding_breakdown": grounding_breakdown,
                "plan_memory_eligible": plan_memory_eligible,
                "plan_template_key": plan_template_key,
                "context_item_ids": list(planning_context.context_item_ids),
                "relevant_goal_ids": relevant_goal_ids,
                "secondary_goal_ids": [
                    match.goal.id for match in ranked_goals[1:self.relevant_limit]
                ],
                "relevant_memory_ids": relevant_memory_ids,
                "relevant_belief_ids": relevant_belief_ids,
                "blocked_steps": blocked_steps,
                "approval_required_steps": approval_required_steps,
                "policy_satisfied": policy_satisfied,
                "goal_priority": (
                    round(matched_goal.goal.priority, 3)
                    if matched_goal is not None
                    else None
                ),
                "belief_confidence_average": self._average_belief_confidence(
                    relevant_beliefs
                ),
                "policy_summary": planning_context.policy_summary,
                "signal_summaries": list(planning_context.signal_summaries),
                "conflict_report": conflict_report,
                "goal_ranking": [
                    match.to_dict(selected=index < self.relevant_limit, rank=index + 1)
                    for index, match in enumerate(ranked_goals)
                ],
                "memory_ranking": [
                    match.to_dict(selected=index < self.relevant_limit, rank=index + 1)
                    for index, match in enumerate(relevant_memories)
                ],
                "belief_ranking": [
                    match.to_dict(selected=index < self.relevant_limit, rank=index + 1)
                    for index, match in enumerate(relevant_beliefs)
                ],
                "selected_context_reasons": self._selected_context_reasons(
                    ranked_goals=ranked_goals,
                    relevant_memories=relevant_memories,
                    relevant_beliefs=relevant_beliefs,
                ),
            },
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: object) -> str:
        digest = sha1(
            "||".join(str(part) for part in parts).encode("utf-8")
        ).hexdigest()[:16]
        return f"{prefix}-{digest}"

    @staticmethod
    def _metadata_flag(metadata: dict[str, Any], key: str) -> bool:
        raw_value = metadata.get(key)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return bool(raw_value)
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _is_explicit_request(self, event: Event) -> bool:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if any(self._metadata_flag(metadata, key) for key in PLANNING_METADATA_FLAGS):
            return True
        normalized_content = event.content.casefold()
        return any(phrase in normalized_content for phrase in EXPLICIT_REQUEST_PHRASES)

    def _detect_query_intent(self, content: str) -> str | None:
        normalized_content = content.casefold()
        if any(phrase in normalized_content for phrase in STATUS_QUERY_PHRASES):
            return "status"
        if any(phrase in normalized_content for phrase in PLANNING_QUERY_PHRASES):
            return "planning"
        if any(phrase in normalized_content for phrase in RECOMMENDATION_QUERY_PHRASES):
            return "recommendation_advice"
        return None

    def _goal_trigger_allowed(self, event: Event) -> bool:
        normalized_content = event.content.casefold()
        return any(phrase in normalized_content for phrase in PLANNING_QUERY_PHRASES)

    def _trigger_reason(
        self,
        *,
        explicit_request: bool,
        query_intent: str | None,
        matched_goal: _GoalMatch | None,
        ranked_goals: list[_GoalMatch],
    ) -> str | None:
        if explicit_request:
            return "explicit_request"
        if matched_goal is not None and matched_goal.goal.priority >= HIGH_PRIORITY_GOAL_THRESHOLD:
            return "high_priority_goal"
        if any(goal.goal.priority >= HIGH_PRIORITY_GOAL_THRESHOLD for goal in ranked_goals):
            return "high_priority_goal"
        if query_intent == "recommendation_advice":
            return "recommendation_request"
        if query_intent == "planning":
            return "planning_request"
        return None

    def _resolve_pressure(
        self,
        event: Event,
        state: NexusState,
    ) -> tuple[float, str]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        explicit_pressure = self._numeric_unit_value(metadata.get("pressure"))
        if explicit_pressure is not None:
            return explicit_pressure, "event_metadata.pressure"

        salience = self._numeric_unit_value(metadata.get("salience"))
        if salience is not None:
            return salience, "event_metadata.salience"

        behavior_state = state.facet_state.get("behavior")
        if isinstance(behavior_state, dict):
            if behavior_state.get("last_event_id") == event.event_id:
                behavior_confidence = self._numeric_unit_value(
                    behavior_state.get("last_confidence")
                )
                if behavior_confidence is not None:
                    return behavior_confidence, "behavior.last_confidence"

        return 0.0, "default"

    @staticmethod
    def _numeric_unit_value(raw_value: Any) -> float | None:
        if not isinstance(raw_value, (int, float)):
            return None
        return round(max(0.0, min(float(raw_value), 1.0)), 3)

    def _load_active_goals(self) -> list[Goal]:
        if self.goal_store is None:
            return []
        return self.goal_store.list_active_goals(limit=self.active_goal_limit)
    def _load_active_beliefs(self) -> list[Belief]:
        if self.world_model_store is None:
            return []
        return self.world_model_store.list_active_beliefs(limit=self.active_belief_limit)

    def _planning_context(self, event: Event, state: NexusState) -> _PlanningContext:
        context_state = state.facet_state.get("context")
        if isinstance(context_state, dict):
            payload = context_state.get("last_context_window")
            if isinstance(payload, dict):
                try:
                    window = ContextWindow.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    window = None
                if window is not None:
                    return self._context_from_window(event, window)
        return self._fallback_context(event)

    def _context_from_window(
        self,
        event: Event,
        window: ContextWindow,
    ) -> _PlanningContext:
        goals: list[_GoalCandidate] = []
        memories: list[_MemoryCandidate] = []
        beliefs: list[_BeliefCandidate] = []
        policy_summary: str | None = None
        signal_summaries: list[str] = []
        for item in window.items:
            if item.item_type == ContextItemType.GOAL:
                goals.append(self._goal_candidate_from_item(item, fallback_time=event.timestamp))
            elif item.item_type == ContextItemType.MEMORY:
                memories.append(self._memory_candidate_from_item(item, fallback_time=event.timestamp))
            elif item.item_type == ContextItemType.BELIEF:
                beliefs.append(self._belief_candidate_from_item(item, fallback_time=event.timestamp))
            elif item.item_type == ContextItemType.POLICY and item.content.strip():
                policy_summary = item.content.strip()
            elif item.item_type == ContextItemType.SIGNAL and item.content.strip():
                signal_summaries.append(item.content.strip())
        return _PlanningContext(
            context_window=window,
            current_event=event.content,
            goals=goals,
            memories=memories,
            beliefs=beliefs,
            policy_summary=policy_summary,
            signal_summaries=signal_summaries,
            context_item_ids=[item.id for item in window.items],
        )

    def _fallback_context(self, event: Event) -> _PlanningContext:
        return _PlanningContext(
            context_window=None,
            current_event=event.content,
            goals=[self._goal_candidate_from_goal(goal) for goal in self._load_active_goals()],
            memories=[],
            beliefs=[self._belief_candidate_from_belief(belief) for belief in self._load_active_beliefs()],
            policy_summary=None,
            signal_summaries=[],
            context_item_ids=[],
        )

    def _goal_candidate_from_goal(self, goal: Goal) -> _GoalCandidate:
        return _GoalCandidate(
            id=goal.id,
            description=goal.description,
            priority=goal.priority,
            updated_at=goal.updated_at,
            tags=list(goal.tags),
            metadata=dict(goal.metadata),
        )

    def _goal_candidate_from_item(
        self,
        item: ContextItem,
        *,
        fallback_time: datetime,
    ) -> _GoalCandidate:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        priority = metadata.get("priority", 0.5)
        return _GoalCandidate(
            id=item.id,
            description=item.content,
            priority=float(priority) if isinstance(priority, (int, float)) else 0.5,
            updated_at=item.created_at or fallback_time,
            tags=self._metadata_tags(metadata),
            metadata=dict(metadata),
        )

    def _memory_candidate_from_item(
        self,
        item: ContextItem,
        *,
        fallback_time: datetime,
    ) -> _MemoryCandidate:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        return _MemoryCandidate(
            id=item.id,
            content=item.content,
            created_at=item.created_at or fallback_time,
            salience=self._numeric_unit_value(metadata.get("salience")) or 0.0,
            confidence=self._numeric_unit_value(metadata.get("confidence")) or 0.0,
            tags=self._metadata_tags(metadata),
            context_source=str(metadata.get("context_source") or "") or None,
            metadata=dict(metadata),
        )

    def _belief_candidate_from_belief(self, belief: Belief) -> _BeliefCandidate:
        return _BeliefCandidate(
            id=belief.id,
            claim=belief.claim,
            confidence=belief.confidence,
            updated_at=belief.updated_at,
            tags=list(belief.tags),
            metadata=dict(belief.metadata),
        )

    def _belief_candidate_from_item(
        self,
        item: ContextItem,
        *,
        fallback_time: datetime,
    ) -> _BeliefCandidate:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        return _BeliefCandidate(
            id=item.id,
            claim=item.content,
            confidence=self._numeric_unit_value(metadata.get("confidence")) or 0.0,
            updated_at=item.created_at or fallback_time,
            tags=self._metadata_tags(metadata),
            metadata=dict(metadata),
        )

    @staticmethod
    def _metadata_tags(metadata: dict[str, Any]) -> list[str]:
        raw_tags = metadata.get("tags", [])
        if not isinstance(raw_tags, list):
            return []
        return [str(tag) for tag in raw_tags if str(tag).strip()]

    def _select_goals(
        self,
        event: Event,
        planning_context: _PlanningContext,
        *,
        query_intent: str | None,
        explicit_request: bool,
    ) -> list[_GoalMatch]:
        if not planning_context.goals:
            return []
        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        matches: list[_GoalMatch] = []
        for goal in planning_context.goals:
            goal_tags = set(goal.tags)
            goal_keywords = tokenize(goal.description)
            shared_tags = sorted(event_tags & goal_tags)
            shared_keywords = sorted(event_keywords & goal_keywords)
            tag_overlap = len(shared_tags) / len(goal_tags) if goal_tags else 0.0
            keyword_overlap = (
                len(shared_keywords) / len(goal_keywords) if goal_keywords else 0.0
            )
            score = tag_overlap + keyword_overlap + goal.priority
            reasons: list[str] = []
            if shared_tags:
                reasons.append("shared_tags:" + ",".join(shared_tags))
            if shared_keywords:
                reasons.append("shared_keywords:" + ",".join(shared_keywords))
            if query_intent == "planning":
                score += 0.3
                reasons.append("planning_active_goal_context")
            elif explicit_request and shared_tags:
                score += 0.1
            if shared_tags or shared_keywords or query_intent == "planning":
                matches.append(
                    _GoalMatch(
                        goal=goal,
                        score=score,
                        shared_tags=shared_tags,
                        shared_keywords=shared_keywords,
                        reasons=reasons or ["active_goal_context"],
                    )
                )
        matches.sort(
            key=lambda match: (
                match.score,
                match.goal.priority,
                match.goal.updated_at.timestamp(),
                match.goal.id,
            ),
            reverse=True,
        )
        return matches

    def _select_memories(
        self,
        event: Event,
        planning_context: _PlanningContext,
        *,
        query_intent: str | None,
    ) -> list[_MemoryMatch]:
        if not planning_context.memories:
            return []
        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        matches: list[_MemoryMatch] = []
        for memory in planning_context.memories:
            memory_tags = set(memory.tags)
            memory_keywords = tokenize(memory.content)
            shared_tags = sorted(event_tags & memory_tags)
            shared_keywords = sorted(event_keywords & memory_keywords)
            tag_overlap = len(shared_tags) / len(memory_tags) if memory_tags else 0.0
            keyword_overlap = (
                len(shared_keywords) / len(memory_keywords) if memory_keywords else 0.0
            )
            score = tag_overlap + keyword_overlap + (memory.salience * 0.2)
            reasons: list[str] = []
            if memory.context_source == "relevant":
                score += 0.05
                reasons.append("context_marked_relevant")
            preference_polarity, preference_terms = self._preference_signature(memory.content)
            if query_intent == "recommendation_advice" and preference_terms:
                score += 0.85 if preference_polarity == "positive" else 0.7
                reasons.append(f"{preference_polarity}_preference_memory")
            elif preference_terms:
                score += 0.1
                reasons.append("stored_preference_memory")
            if shared_tags:
                reasons.append("shared_tags:" + ",".join(shared_tags))
            if shared_keywords:
                reasons.append("shared_keywords:" + ",".join(shared_keywords))
            if score > 0.0:
                matches.append(
                    _MemoryMatch(
                        memory=memory,
                        score=score,
                        shared_tags=shared_tags,
                        shared_keywords=shared_keywords,
                        preference_polarity=preference_polarity,
                        preference_terms=preference_terms,
                        reasons=reasons or ["memory_available"],
                    )
                )
        matches.sort(
            key=lambda match: (
                match.score,
                match.memory.salience,
                match.memory.created_at.timestamp(),
                match.memory.id,
            ),
            reverse=True,
        )
        return matches

    def _select_beliefs(
        self,
        event: Event,
        planning_context: _PlanningContext,
        *,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
    ) -> list[_BeliefMatch]:
        if not planning_context.beliefs:
            return []
        reference_tags = extract_event_tags(event)
        reference_keywords = tokenize(event.content)
        if matched_goal is not None:
            reference_tags |= set(matched_goal.goal.tags)
            reference_keywords |= tokenize(matched_goal.goal.description)
        for memory in relevant_memories[: self.relevant_limit]:
            reference_tags |= set(memory.memory.tags)
            reference_keywords |= tokenize(memory.memory.content)
        matches: list[_BeliefMatch] = []
        for belief in planning_context.beliefs:
            belief_tags = set(belief.tags)
            belief_keywords = tokenize(belief.claim)
            shared_tags = sorted(reference_tags & belief_tags)
            shared_keywords = sorted(reference_keywords & belief_keywords)
            if not shared_tags and not shared_keywords:
                continue
            tag_overlap = len(shared_tags) / len(belief_tags) if belief_tags else 0.0
            keyword_overlap = (
                len(shared_keywords) / len(belief_keywords) if belief_keywords else 0.0
            )
            reasons: list[str] = []
            if shared_tags:
                reasons.append("shared_tags:" + ",".join(shared_tags))
            if shared_keywords:
                reasons.append("shared_keywords:" + ",".join(shared_keywords))
            matches.append(
                _BeliefMatch(
                    belief=belief,
                    score=tag_overlap + keyword_overlap + belief.confidence,
                    shared_tags=shared_tags,
                    shared_keywords=shared_keywords,
                    reasons=reasons or ["belief_overlap"],
                )
            )
        matches.sort(
            key=lambda match: (
                match.score,
                match.belief.confidence,
                match.belief.updated_at.timestamp(),
                match.belief.id,
            ),
            reverse=True,
        )
        return matches[: self.relevant_limit]

    def _preference_signature(self, content: str) -> tuple[str | None, list[str]]:
        positive = self._extract_preference_terms(content, POSITIVE_PREFERENCE_PATTERNS)
        if positive:
            return "positive", positive
        negative = self._extract_preference_terms(content, NEGATIVE_PREFERENCE_PATTERNS)
        if negative:
            return "negative", negative
        return None, []

    def _extract_preference_terms(
        self,
        content: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> list[str]:
        for pattern in patterns:
            match = pattern.search(content)
            if match is None:
                continue
            fragment = match.group(1).strip()
            pieces = [
                self._clean_phrase(piece)
                for piece in SPLIT_PHRASE_PATTERN.split(fragment)
            ]
            terms = [piece for piece in pieces if piece]
            if terms:
                return list(dict.fromkeys(terms))
        return []

    @staticmethod
    def _clean_phrase(value: str) -> str:
        cleaned = " ".join(value.strip().strip("\"'`").split())
        return cleaned.strip(" .!?")

    def _grounding(
        self,
        *,
        query_intent: str | None,
        ranked_goals: list[_GoalMatch],
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
    ) -> tuple[float, str, dict[str, float]]:
        relevant_goal_count = len(ranked_goals[: self.relevant_limit])
        relevant_memory_count = len(relevant_memories[: self.relevant_limit])
        relevant_belief_count = len(relevant_beliefs[: self.relevant_limit])
        preference_memory_count = len(
            [match for match in relevant_memories[: self.relevant_limit] if match.preference_terms]
        )
        overlap_signal = (
            sum(len(match.shared_keywords) + len(match.shared_tags) for match in ranked_goals[: self.relevant_limit])
            + sum(len(match.shared_keywords) + len(match.shared_tags) for match in relevant_memories[: self.relevant_limit])
            + sum(len(match.shared_keywords) + len(match.shared_tags) for match in relevant_beliefs[: self.relevant_limit])
        )
        breakdown = {
            "relevant_goals": min(0.45, relevant_goal_count * 0.3),
            "relevant_memories": min(0.3, relevant_memory_count * 0.18),
            "relevant_beliefs": min(0.15, relevant_belief_count * 0.08),
            "keyword_or_tag_overlap": min(0.15, overlap_signal * 0.03),
            "preference_matches": min(0.25, preference_memory_count * 0.2),
            "intent_bonus": 0.0,
        }
        if query_intent == "planning" and relevant_goal_count > 0:
            breakdown["intent_bonus"] = 0.2
        elif query_intent == "recommendation_advice" and preference_memory_count > 0:
            breakdown["intent_bonus"] = 0.15
        score = round(max(0.0, min(sum(breakdown.values()), 1.0)), 3)
        if score >= 0.5:
            status = "grounded"
        elif score >= 0.25:
            status = "partially_grounded"
        else:
            status = "insufficient_context"
        breakdown["total"] = score
        return score, status, breakdown

    def _build_steps(
        self,
        event: Event,
        *,
        query_intent: str | None,
        grounding_status: str,
        matched_goal: _GoalMatch | None,
        secondary_goals: list[_GoalMatch],
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
        conflict_report: dict[str, Any],
        pressure: float,
    ) -> list[PlanStep]:
        if query_intent == "recommendation_advice":
            return self._build_recommendation_steps(
                event,
                grounding_status=grounding_status,
                matched_goal=matched_goal,
                secondary_goals=secondary_goals,
                relevant_memories=relevant_memories,
                relevant_beliefs=relevant_beliefs,
                conflict_report=conflict_report,
                pressure=pressure,
            )
        return self._build_goal_or_generic_steps(
            event,
            query_intent=query_intent,
            grounding_status=grounding_status,
            matched_goal=matched_goal,
            secondary_goals=secondary_goals,
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
            conflict_report=conflict_report,
            pressure=pressure,
        )

    def _build_recommendation_steps(
        self,
        event: Event,
        *,
        grounding_status: str,
        matched_goal: _GoalMatch | None,
        secondary_goals: list[_GoalMatch],
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
        conflict_report: dict[str, Any],
        pressure: float,
    ) -> list[PlanStep]:
        high_pressure = pressure >= HIGH_PRESSURE_THRESHOLD
        action_target_type = self._action_target_type(event)
        action_risk = self._risk_level_for_target_type(action_target_type)
        shared_metadata = self._action_step_metadata(
            event,
            matched_goal=matched_goal,
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
        )
        positive_summary = self._join_terms(
            self._flatten_preference_terms(relevant_memories, polarity="positive")
        ) or "remembered preferences"
        negative_summary = self._join_terms(
            self._flatten_preference_terms(relevant_memories, polarity="negative")
        )
        if grounding_status == "insufficient_context":
            steps = [
                PlanStep(
                    description="Ask for weekend preferences, constraints, and desired energy level.",
                    order=1,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "clarify_recommendation_context", "action_type": "noop"},
                ),
                PlanStep(
                    description="Clarify whether the recommendation should optimize for rest, adventure, or social time.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "clarify_recommendation_goal", "action_type": "noop"},
                ),
                PlanStep(
                    description="Only then propose a grounded weekend option.",
                    order=3,
                    target_type=action_target_type,
                    risk_level=action_risk,
                    metadata={"step_kind": "propose_grounded_weekend_option", "action_type": "noop", **shared_metadata},
                ),
            ]
            return [steps[0], steps[-1]] if high_pressure else steps
        steps: list[PlanStep] = [
            PlanStep(
                description=f"Start from remembered preferences such as {positive_summary}.",
                order=1,
                target_type="general",
                risk_level=RiskLevel.LOW,
                metadata={
                    "step_kind": "review_preference_grounding",
                    "action_type": "noop",
                    "memory_ids": [match.memory.id for match in relevant_memories[: self.relevant_limit]],
                },
            )
        ]
        if conflict_report.get("has_conflicts"):
            steps.append(
                PlanStep(
                    description=f"Resolve goal conflicts before committing: {conflict_report['summary']}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "resolve_goal_conflict", "action_type": "noop"},
                )
            )
        elif negative_summary is not None:
            steps.append(
                PlanStep(
                    description=f"Filter out options that conflict with your preferences, such as {negative_summary}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "avoid_negative_preferences", "action_type": "noop"},
                )
            )
        elif secondary_goals:
            secondary_summary = "; ".join(goal.goal.description for goal in secondary_goals[:2])
            steps.append(
                PlanStep(
                    description=f"Balance the recommendation with secondary goals: {secondary_summary}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "balance_secondary_goals", "action_type": "noop"},
                )
            )
        else:
            steps.append(
                PlanStep(
                    description="Check practical constraints and immediate fit for this weekend.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "check_weekend_constraints", "action_type": "noop"},
                )
            )
        steps.append(
            PlanStep(
                description=f"Shortlist a weekend option that stays close to {positive_summary}.",
                order=3,
                target_type=action_target_type,
                risk_level=action_risk,
                metadata={"step_kind": "shortlist_grounded_weekend_option", "action_type": "noop", **shared_metadata},
            )
        )
        return [steps[0], steps[-1]] if high_pressure else steps

    def _build_goal_or_generic_steps(
        self,
        event: Event,
        *,
        query_intent: str | None,
        grounding_status: str,
        matched_goal: _GoalMatch | None,
        secondary_goals: list[_GoalMatch],
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
        conflict_report: dict[str, Any],
        pressure: float,
    ) -> list[PlanStep]:
        high_pressure = pressure >= HIGH_PRESSURE_THRESHOLD
        action_target_type = self._action_target_type(event)
        action_risk = self._risk_level_for_target_type(action_target_type)
        shared_metadata = self._action_step_metadata(
            event,
            matched_goal=matched_goal,
            relevant_memories=relevant_memories,
            relevant_beliefs=relevant_beliefs,
        )
        if matched_goal is None:
            if (
                grounding_status == "insufficient_context"
                and query_intent == "planning"
                and not self._is_explicit_request(event)
            ):
                steps = [
                    PlanStep(
                        description="Clarify which goal or outcome should drive the next step.",
                        order=1,
                        target_type="general",
                        risk_level=RiskLevel.LOW,
                        metadata={"step_kind": "clarify_goal_driver", "action_type": "noop"},
                    ),
                    PlanStep(
                        description="Identify the main constraint before proposing a next step.",
                        order=2,
                        target_type="general",
                        risk_level=RiskLevel.LOW,
                        metadata={"step_kind": "identify_primary_constraint", "action_type": "noop"},
                    ),
                    PlanStep(
                        description="Propose the next safe action once the goal is explicit.",
                        order=3,
                        target_type=action_target_type,
                        risk_level=action_risk,
                        metadata={"step_kind": "propose_next_safe_action", "action_type": "noop", **shared_metadata},
                    ),
                ]
                return [steps[0], steps[-1]] if high_pressure else steps
            if high_pressure:
                return [
                    PlanStep(
                        description="Clarify the objective.",
                        order=1,
                        target_type="general",
                        risk_level=RiskLevel.LOW,
                        metadata={"step_kind": "clarify_objective", "action_type": "noop"},
                    ),
                    PlanStep(
                        description="Propose the next safe action.",
                        order=2,
                        target_type=action_target_type,
                        risk_level=action_risk,
                        metadata={"step_kind": "propose_next_safe_action", "action_type": "noop", **shared_metadata},
                    ),
                ]
            return [
                PlanStep(
                    description="Clarify the objective.",
                    order=1,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "clarify_objective", "action_type": "noop"},
                ),
                PlanStep(
                    description="Identify constraints.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "identify_constraints", "action_type": "noop"},
                ),
                PlanStep(
                    description="Propose the next safe action.",
                    order=3,
                    target_type=action_target_type,
                    risk_level=action_risk,
                    metadata={"step_kind": "propose_next_safe_action", "action_type": "noop", **shared_metadata},
                ),
            ]
        goal_description = matched_goal.goal.description
        steps = [
            PlanStep(
                description=f"Review progress and open constraints for {goal_description}.",
                order=1,
                target_type="general",
                risk_level=RiskLevel.LOW,
                metadata={"step_kind": "review_goal_context", "action_type": "noop", "goal_id": matched_goal.goal.id},
            )
        ]
        if conflict_report.get("has_conflicts"):
            steps.append(
                PlanStep(
                    description=f"Resolve the active-goal conflict: {conflict_report['summary']}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "resolve_goal_conflict", "action_type": "noop", "goal_id": matched_goal.goal.id},
                )
            )
        elif secondary_goals:
            secondary_summary = "; ".join(goal.goal.description for goal in secondary_goals[:2])
            steps.append(
                PlanStep(
                    description=f"Sequence the primary goal against secondary goals: {secondary_summary}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "sequence_secondary_goals", "action_type": "noop", "goal_id": matched_goal.goal.id},
                )
            )
        elif relevant_beliefs:
            steps.append(
                PlanStep(
                    description="Check relevant beliefs and constraints before choosing the next action.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "check_beliefs_and_constraints", "action_type": "noop", "goal_id": matched_goal.goal.id, "belief_ids": [match.belief.id for match in relevant_beliefs]},
                )
            )
        else:
            steps.append(
                PlanStep(
                    description=f"Confirm what completion looks like for {goal_description}.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "confirm_goal_success_criteria", "action_type": "noop", "goal_id": matched_goal.goal.id},
                )
            )
        steps.append(
            PlanStep(
                description=f"Propose the next safe action toward {goal_description}.",
                order=3,
                target_type=action_target_type,
                risk_level=action_risk,
                metadata={"step_kind": "propose_next_safe_action_toward_goal", "action_type": "noop", "goal_id": matched_goal.goal.id, **shared_metadata},
            )
        )
        return [steps[0], steps[-1]] if high_pressure else steps

    def _action_step_metadata(
        self,
        event: Event,
        *,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
    ) -> dict[str, Any]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        payload: dict[str, Any] = {
            "goal_id": matched_goal.goal.id if matched_goal is not None else None,
            "memory_ids": [match.memory.id for match in relevant_memories[: self.relevant_limit]],
            "belief_ids": [match.belief.id for match in relevant_beliefs],
        }
        for key in ("target", "path", "operation"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()
        return payload

    def _action_target_type(self, event: Event) -> str:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        raw_target_type = coerce_policy_target_type(metadata.get("target_type"))
        if raw_target_type is None:
            return "general"
        return raw_target_type.value

    @staticmethod
    def _risk_level_for_target_type(target_type: str) -> RiskLevel:
        if target_type in {
            PolicyTargetType.SHELL.value,
            PolicyTargetType.NETWORK.value,
            PolicyTargetType.MESSAGE.value,
            PolicyTargetType.GIT.value,
            PolicyTargetType.TOOL.value,
            PolicyTargetType.FILE_DELETE.value,
        }:
            return RiskLevel.HIGH
        if target_type == PolicyTargetType.FILE_WRITE.value:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _apply_policy_filters(
        self,
        event: Event,
        state: NexusState,
        steps: list[PlanStep],
    ) -> bool:
        from fullerene.facets.policy import PolicyFacet

        policy_evaluator = (
            PolicyFacet(self.policy_store, state_dir=self.state_dir)
            if self.policy_store is not None
            else None
        )
        all_constraints_satisfied = policy_evaluator is not None

        for step in steps:
            if policy_evaluator is not None:
                policy_event = Event(
                    event_type=event.event_type,
                    content=step.description,
                    metadata=self._policy_metadata_for_step(event, step),
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                )
                result = policy_evaluator.process(policy_event, state)
                policy_status = result.metadata.get("policy_status")
                if isinstance(policy_status, str):
                    step.policy_status = policy_status
                if step.policy_status == PolicyStatus.DENIED.value:
                    step.status = PlanStepStatus.BLOCKED
                    all_constraints_satisfied = False
                elif step.policy_status == PolicyStatus.APPROVAL_REQUIRED.value:
                    step.status = PlanStepStatus.REQUIRES_APPROVAL
                    step.requires_approval = True
                    all_constraints_satisfied = False

            if step.risk_level == RiskLevel.HIGH:
                step.requires_approval = True
                if step.status != PlanStepStatus.BLOCKED:
                    step.status = PlanStepStatus.REQUIRES_APPROVAL
                all_constraints_satisfied = False

        return all_constraints_satisfied

    @staticmethod
    def _policy_metadata_for_step(event: Event, step: PlanStep) -> dict[str, Any]:
        event_metadata = event.metadata if isinstance(event.metadata, dict) else {}
        metadata = {
            "explicit_action": True,
            "target_type": step.target_type,
        }
        for key in ("target", "path", "operation", "tags"):
            if key in step.metadata:
                metadata[key] = step.metadata[key]
                continue
            if key in event_metadata:
                metadata[key] = event_metadata[key]
        return metadata

    def _confidence(
        self,
        *,
        explicit_request: bool,
        query_intent: str | None,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
        policy_satisfied: bool,
        grounding_score: float,
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {
            "base": 0.25,
            "grounding": round(grounding_score * 0.4, 3),
        }
        if explicit_request:
            breakdown["explicit_request"] = 0.15
        if matched_goal is not None:
            breakdown["goal_priority"] = round(matched_goal.goal.priority * 0.15, 3)
        if query_intent == "recommendation_advice":
            preference_count = len(
                [
                    match
                    for match in relevant_memories[: self.relevant_limit]
                    if match.preference_terms
                ]
            )
            if preference_count > 0:
                breakdown["preference_context"] = min(0.15, preference_count * 0.1)
        average_belief_confidence = self._average_belief_confidence(relevant_beliefs)
        if average_belief_confidence > 0.0:
            breakdown["belief_confidence"] = round(average_belief_confidence * 0.1, 3)
        if policy_satisfied:
            breakdown["policy_satisfied"] = 0.1
        breakdown["total"] = round(
            max(0.0, min(sum(breakdown.values()), 1.0)),
            3,
        )
        return breakdown["total"], breakdown

    @staticmethod
    def _average_belief_confidence(relevant_beliefs: list[_BeliefMatch]) -> float:
        if not relevant_beliefs:
            return 0.0
        total = sum(match.belief.confidence for match in relevant_beliefs)
        return round(total / len(relevant_beliefs), 3)

    def _title_for_plan(
        self,
        *,
        query_intent: str | None,
        grounding_status: str,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
    ) -> str:
        if query_intent == "recommendation_advice":
            preference_terms = self._flatten_preference_terms(
                relevant_memories,
                polarity="positive",
            )
            if grounding_status == "insufficient_context":
                return "Clarify recommendation context before suggesting options"
            if preference_terms:
                return f"Grounded recommendation plan around {self._join_terms(preference_terms)}"
            return "Grounded recommendation plan"
        if matched_goal is None:
            return "Proposed safe next-step plan"
        return f"Proposed plan for goal: {matched_goal.goal.description}"

    def _plan_template_key(
        self,
        *,
        query_intent: str | None,
        grounding_status: str,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
    ) -> str:
        if query_intent == "recommendation_advice":
            preference_terms = self._flatten_preference_terms(
                relevant_memories,
                polarity="positive",
            )
            if grounding_status == "insufficient_context":
                return "recommendation.clarify"
            if preference_terms:
                return "recommendation.preference_grounded"
            return "recommendation.generic_grounded"
        if matched_goal is not None:
            return "planning.goal_grounded"
        if grounding_status == "insufficient_context":
            return "planning.clarify"
        return "planning.generic"

    def _plan_reasons(
        self,
        *,
        explicit_request: bool,
        query_intent: str | None,
        matched_goal: _GoalMatch | None,
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
        policy_satisfied: bool,
        pressure: float,
        grounding_status: str,
    ) -> list[str]:
        reasons: list[str] = [f"grounding_status:{grounding_status}"]
        if explicit_request:
            reasons.append("explicit_plan_request")
        if query_intent is not None:
            reasons.append(f"query_intent:{query_intent}")
        if matched_goal is not None:
            reasons.append(f"goal:{matched_goal.goal.id}")
        for memory in relevant_memories[: self.relevant_limit]:
            if memory.preference_terms:
                reasons.append(
                    f"memory:{memory.memory.id}:{memory.preference_polarity}:{self._join_terms(memory.preference_terms)}"
                )
            else:
                reasons.append(f"memory:{memory.memory.id}")
        if relevant_beliefs:
            reasons.append(
                "beliefs:" + ",".join(match.belief.id for match in relevant_beliefs)
            )
        if policy_satisfied:
            reasons.append("policy_constraints_satisfied")
        if pressure >= HIGH_PRESSURE_THRESHOLD:
            reasons.append("high_pressure_direct_path")
        else:
            reasons.append("low_pressure_exploratory_path")
        return reasons

    def _selected_context_reasons(
        self,
        *,
        ranked_goals: list[_GoalMatch],
        relevant_memories: list[_MemoryMatch],
        relevant_beliefs: list[_BeliefMatch],
    ) -> list[str]:
        reasons: list[str] = []
        for match in ranked_goals[: self.relevant_limit]:
            reasons.append(f"goal:{match.goal.id}:" + "|".join(match.reasons))
        for match in relevant_memories[: self.relevant_limit]:
            reasons.append(f"memory:{match.memory.id}:" + "|".join(match.reasons))
        for match in relevant_beliefs[: self.relevant_limit]:
            reasons.append(f"belief:{match.belief.id}:" + "|".join(match.reasons))
        return reasons

    def _flatten_preference_terms(
        self,
        relevant_memories: list[_MemoryMatch],
        *,
        polarity: str,
    ) -> list[str]:
        terms: list[str] = []
        for match in relevant_memories[: self.relevant_limit]:
            if match.preference_polarity != polarity:
                continue
            terms.extend(match.preference_terms)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(term)
        return deduped[:3]

    @staticmethod
    def _join_terms(terms: list[str]) -> str | None:
        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        if len(terms) == 2:
            return f"{terms[0]} and {terms[1]}"
        return f"{', '.join(terms[:-1])}, and {terms[-1]}"

    def _conflict_report(self, ranked_goals: list[_GoalMatch]) -> dict[str, Any]:
        selected_goals = ranked_goals[: self.relevant_limit]
        conflicts: list[dict[str, Any]] = []
        for left_index, left in enumerate(selected_goals):
            for right in selected_goals[left_index + 1:]:
                signal = self._goal_conflict_signal(
                    left.goal.description,
                    right.goal.description,
                )
                if signal is None:
                    continue
                conflicts.append(
                    {
                        "goal_ids": [left.goal.id, right.goal.id],
                        "descriptions": [left.goal.description, right.goal.description],
                        "signal": signal,
                    }
                )
        if not conflicts:
            return {"has_conflicts": False, "conflicts": [], "summary": None}
        summary = "; ".join(
            f"{conflict['descriptions'][0]} <-> {conflict['descriptions'][1]}"
            for conflict in conflicts
        )
        return {"has_conflicts": True, "conflicts": conflicts, "summary": summary}

    def _goal_conflict_signal(self, left: str, right: str) -> str | None:
        left_tokens = tokenize(left)
        right_tokens = tokenize(right)
        shared_context = [
            token
            for token in sorted(left_tokens & right_tokens)
            if token not in STOPWORDS
        ]
        if not shared_context:
            return None
        left_groups = self._conflict_groups(left_tokens)
        right_groups = self._conflict_groups(right_tokens)
        for first, second in CONFLICT_PAIRS:
            if (first in left_groups and second in right_groups) or (
                second in left_groups and first in right_groups
            ):
                return f"{first}_vs_{second}"
        if "stay" in left_groups and "travel" in right_groups:
            return "stay_vs_travel"
        if "stay" in right_groups and "travel" in left_groups:
            return "stay_vs_travel"
        return None

    def _conflict_groups(self, tokens: set[str]) -> set[str]:
        groups: set[str] = set()
        for group_name, members in CONFLICT_GROUPS.items():
            if tokens & members:
                groups.add(group_name)
        return groups
