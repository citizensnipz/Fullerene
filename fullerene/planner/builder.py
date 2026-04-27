"""Deterministic plan construction for Fullerene Planner v0."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

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
NEXT_STEP_PHRASES = (
    "what next",
    "next step",
    "next steps",
    "what are the next steps",
)
PLANNING_METADATA_FLAGS = ("request_plan", "allow_planning", "planning_allowed")
HIGH_PRIORITY_GOAL_THRESHOLD = 0.75
HIGH_PRESSURE_THRESHOLD = 0.7


@dataclass(slots=True)
class _GoalMatch:
    goal: Goal
    score: float
    shared_tags: list[str]
    shared_keywords: list[str]


@dataclass(slots=True)
class _BeliefMatch:
    belief: Belief
    score: float
    shared_tags: list[str]
    shared_keywords: list[str]


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
        pressure, pressure_source = self._resolve_pressure(event, state)

        active_goals = self._load_active_goals()
        matched_goal = self._select_goal(event, active_goals, explicit_request)
        high_priority_goals = [
            goal for goal in active_goals if goal.priority >= HIGH_PRIORITY_GOAL_THRESHOLD
        ]
        goal_trigger_allowed = self._goal_trigger_allowed(event)
        trigger_reason = self._trigger_reason(
            explicit_request=explicit_request,
            matched_goal=matched_goal,
            high_priority_goals=high_priority_goals,
            goal_trigger_allowed=goal_trigger_allowed,
        )
        if trigger_reason is None:
            return None

        relevant_beliefs = self._select_beliefs(event, matched_goal)
        steps = self._build_steps(event, matched_goal, relevant_beliefs, pressure)
        policy_satisfied = self._apply_policy_filters(event, state, steps)
        confidence, confidence_breakdown = self._confidence(
            explicit_request=explicit_request,
            matched_goal=matched_goal,
            relevant_beliefs=relevant_beliefs,
            policy_satisfied=policy_satisfied,
        )
        if confidence < self._proposal_threshold(pressure):
            return None

        generation_mode = (
            "high_pressure_direct"
            if pressure >= HIGH_PRESSURE_THRESHOLD
            else "low_pressure_exploratory"
        )
        plan_id = self._stable_id(
            "plan",
            event.event_id,
            trigger_reason,
            matched_goal.goal.id if matched_goal is not None else "generic",
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
        relevant_goal_ids = [matched_goal.goal.id] if matched_goal is not None else []
        relevant_belief_ids = [match.belief.id for match in relevant_beliefs]
        blocked_steps = [step.id for step in steps if step.status == PlanStepStatus.BLOCKED]
        approval_required_steps = [
            step.id for step in steps if step.requires_approval
        ]

        return Plan(
            id=plan_id,
            created_at=event.timestamp,
            source_event_id=event.event_id,
            goal_id=matched_goal.goal.id if matched_goal is not None else None,
            title=self._title_for_plan(matched_goal),
            steps=steps,
            confidence=confidence,
            pressure=pressure,
            status=PlanStatus.PROPOSED,
            reasons=self._plan_reasons(
                explicit_request=explicit_request,
                matched_goal=matched_goal,
                relevant_beliefs=relevant_beliefs,
                policy_satisfied=policy_satisfied,
                pressure=pressure,
            ),
            metadata={
                "trigger_reason": trigger_reason,
                "pressure_source": pressure_source,
                "generation_mode": generation_mode,
                "confidence_breakdown": confidence_breakdown,
                "relevant_goal_ids": relevant_goal_ids,
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

    def _goal_trigger_allowed(self, event: Event) -> bool:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if any(self._metadata_flag(metadata, key) for key in PLANNING_METADATA_FLAGS):
            return True
        normalized_content = event.content.casefold()
        return any(phrase in normalized_content for phrase in NEXT_STEP_PHRASES)

    def _trigger_reason(
        self,
        *,
        explicit_request: bool,
        matched_goal: _GoalMatch | None,
        high_priority_goals: list[Goal],
        goal_trigger_allowed: bool,
    ) -> str | None:
        if explicit_request:
            return "explicit_request"
        if (
            matched_goal is not None
            and matched_goal.goal.priority >= HIGH_PRIORITY_GOAL_THRESHOLD
            and goal_trigger_allowed
        ):
            return "high_priority_goal"
        if high_priority_goals and goal_trigger_allowed:
            return "high_priority_goal"
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

    def _select_goal(
        self,
        event: Event,
        active_goals: list[Goal],
        explicit_request: bool,
    ) -> _GoalMatch | None:
        if not active_goals:
            return None

        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        matches: list[_GoalMatch] = []
        for goal in active_goals:
            goal_tags = set(goal.tags)
            goal_keywords = tokenize(goal.description)
            shared_tags = sorted(event_tags & goal_tags)
            shared_keywords = sorted(event_keywords & goal_keywords)
            if not shared_tags and not shared_keywords:
                continue
            tag_overlap = len(shared_tags) / len(goal_tags) if goal_tags else 0.0
            keyword_overlap = (
                len(shared_keywords) / len(goal_keywords) if goal_keywords else 0.0
            )
            matches.append(
                _GoalMatch(
                    goal=goal,
                    score=tag_overlap + keyword_overlap + goal.priority,
                    shared_tags=shared_tags,
                    shared_keywords=shared_keywords,
                )
            )

        if matches:
            matches.sort(
                key=lambda match: (
                    match.score,
                    match.goal.priority,
                    match.goal.updated_at.timestamp(),
                    match.goal.id,
                ),
                reverse=True,
            )
            return matches[0]

        if explicit_request:
            return None

        high_priority_goals = [
            goal for goal in active_goals if goal.priority >= HIGH_PRIORITY_GOAL_THRESHOLD
        ]
        if not high_priority_goals:
            return None
        high_priority_goals.sort(
            key=lambda goal: (goal.priority, goal.updated_at.timestamp(), goal.id),
            reverse=True,
        )
        top_goal = high_priority_goals[0]
        return _GoalMatch(
            goal=top_goal,
            score=top_goal.priority,
            shared_tags=[],
            shared_keywords=[],
        )

    def _select_beliefs(
        self,
        event: Event,
        matched_goal: _GoalMatch | None,
    ) -> list[_BeliefMatch]:
        if self.world_model_store is None:
            return []

        active_beliefs = self.world_model_store.list_active_beliefs(
            limit=self.active_belief_limit
        )
        if not active_beliefs:
            return []

        reference_tags = extract_event_tags(event)
        reference_keywords = tokenize(event.content)
        if matched_goal is not None:
            reference_tags |= set(matched_goal.goal.tags)
            reference_keywords |= tokenize(matched_goal.goal.description)

        matches: list[_BeliefMatch] = []
        for belief in active_beliefs:
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
            matches.append(
                _BeliefMatch(
                    belief=belief,
                    score=tag_overlap + keyword_overlap + belief.confidence,
                    shared_tags=shared_tags,
                    shared_keywords=shared_keywords,
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

    def _build_steps(
        self,
        event: Event,
        matched_goal: _GoalMatch | None,
        relevant_beliefs: list[_BeliefMatch],
        pressure: float,
    ) -> list[PlanStep]:
        high_pressure = pressure >= HIGH_PRESSURE_THRESHOLD
        action_target_type = self._action_target_type(event)
        action_risk = self._risk_level_for_target_type(action_target_type)
        shared_metadata = self._action_step_metadata(
            event,
            matched_goal=matched_goal,
            relevant_beliefs=relevant_beliefs,
        )

        if matched_goal is None:
            if high_pressure:
                return [
                    PlanStep(
                        description="Clarify the objective.",
                        order=1,
                        target_type="general",
                        risk_level=RiskLevel.LOW,
                        metadata={"step_kind": "clarify_objective"},
                    ),
                    PlanStep(
                        description="Propose the next safe action.",
                        order=2,
                        target_type=action_target_type,
                        risk_level=action_risk,
                        metadata={
                            "step_kind": "propose_next_safe_action",
                            **shared_metadata,
                        },
                    ),
                ]
            return [
                PlanStep(
                    description="Clarify the objective.",
                    order=1,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "clarify_objective"},
                ),
                PlanStep(
                    description="Identify constraints.",
                    order=2,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={"step_kind": "identify_constraints"},
                ),
                PlanStep(
                    description="Propose the next safe action.",
                    order=3,
                    target_type=action_target_type,
                    risk_level=action_risk,
                    metadata={
                        "step_kind": "propose_next_safe_action",
                        **shared_metadata,
                    },
                ),
            ]

        if high_pressure:
            return [
                PlanStep(
                    description="Review goal context and immediate constraints.",
                    order=1,
                    target_type="general",
                    risk_level=RiskLevel.LOW,
                    metadata={
                        "step_kind": "review_goal_context",
                        "goal_id": matched_goal.goal.id,
                    },
                ),
                PlanStep(
                    description="Propose the next safe action toward the goal.",
                    order=2,
                    target_type=action_target_type,
                    risk_level=action_risk,
                    metadata={
                        "step_kind": "propose_next_safe_action_toward_goal",
                        "goal_id": matched_goal.goal.id,
                        **shared_metadata,
                    },
                ),
            ]
        return [
            PlanStep(
                description="Review goal context.",
                order=1,
                target_type="general",
                risk_level=RiskLevel.LOW,
                metadata={
                    "step_kind": "review_goal_context",
                    "goal_id": matched_goal.goal.id,
                },
            ),
            PlanStep(
                description="Check relevant beliefs and constraints.",
                order=2,
                target_type="general",
                risk_level=RiskLevel.LOW,
                metadata={
                    "step_kind": "check_beliefs_and_constraints",
                    "goal_id": matched_goal.goal.id,
                    "belief_ids": [match.belief.id for match in relevant_beliefs],
                },
            ),
            PlanStep(
                description="Propose the next safe action toward the goal.",
                order=3,
                target_type=action_target_type,
                risk_level=action_risk,
                metadata={
                    "step_kind": "propose_next_safe_action_toward_goal",
                    "goal_id": matched_goal.goal.id,
                    **shared_metadata,
                },
            ),
        ]

    def _action_step_metadata(
        self,
        event: Event,
        *,
        matched_goal: _GoalMatch | None,
        relevant_beliefs: list[_BeliefMatch],
    ) -> dict[str, Any]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        payload: dict[str, Any] = {
            "goal_id": matched_goal.goal.id if matched_goal is not None else None,
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
        matched_goal: _GoalMatch | None,
        relevant_beliefs: list[_BeliefMatch],
        policy_satisfied: bool,
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {"base": 0.4}
        if explicit_request:
            breakdown["explicit_request"] = 0.2
        if matched_goal is not None:
            breakdown["goal_priority"] = round(matched_goal.goal.priority * 0.3, 3)
        average_belief_confidence = self._average_belief_confidence(relevant_beliefs)
        if average_belief_confidence > 0.0:
            breakdown["belief_confidence"] = round(average_belief_confidence * 0.2, 3)
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

    @staticmethod
    def _proposal_threshold(pressure: float) -> float:
        if pressure >= HIGH_PRESSURE_THRESHOLD:
            return 0.35
        return 0.45

    @staticmethod
    def _title_for_plan(matched_goal: _GoalMatch | None) -> str:
        if matched_goal is None:
            return "Proposed safe next-step plan"
        return f"Proposed plan for goal: {matched_goal.goal.description}"

    @staticmethod
    def _plan_reasons(
        *,
        explicit_request: bool,
        matched_goal: _GoalMatch | None,
        relevant_beliefs: list[_BeliefMatch],
        policy_satisfied: bool,
        pressure: float,
    ) -> list[str]:
        reasons: list[str] = []
        if explicit_request:
            reasons.append("explicit_plan_request")
        if matched_goal is not None:
            reasons.append(f"goal:{matched_goal.goal.id}")
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
