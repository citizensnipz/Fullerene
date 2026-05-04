"""Deterministic planner facet for Fullerene Planner v0."""

from __future__ import annotations

from pathlib import Path

from fullerene.goals import GoalStore
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.planner import DeterministicPlanBuilder
from fullerene.policy import PolicyStore
from fullerene.world_model import WorldModelStore


class PlannerFacet:
    """Propose deterministic plans without execution or tool use."""

    name = "planner"

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
        self.builder = DeterministicPlanBuilder(
            goal_store=goal_store,
            world_model_store=world_model_store,
            policy_store=policy_store,
            state_dir=state_dir,
            active_goal_limit=active_goal_limit,
            active_belief_limit=active_belief_limit,
            relevant_limit=relevant_limit,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        plan = self.builder.build(event, state)
        if plan is None:
            return FacetResult(
                facet_name=self.name,
                summary="Planner facet did not propose a plan for this event.",
                proposed_decision=DecisionAction.WAIT,
                state_updates={
                    "last_plan": None,
                    "last_plan_id": None,
                    "last_trigger_reason": None,
                    "last_plan_confidence": 0.0,
                    "last_plan_pressure": 0.0,
                },
                metadata={
                    "plan": None,
                    "trigger_reason": None,
                    "query_intent": None,
                    "confidence": 0.0,
                    "pressure": 0.0,
                    "grounding_status": "insufficient_context",
                    "grounding_score": 0.0,
                    "plan_memory_eligible": False,
                    "plan_template_key": None,
                    "context_item_ids": [],
                    "relevant_memory_ids": [],
                    "relevant_goal_ids": [],
                    "relevant_belief_ids": [],
                    "goal_ranking": [],
                    "memory_ranking": [],
                    "belief_ranking": [],
                    "selected_context_reasons": [],
                    "conflict_report": {
                        "has_conflicts": False,
                        "conflicts": [],
                        "summary": None,
                    },
                    "blocked_steps": [],
                    "approval_required_steps": [],
                    "reasons": ["planner_not_triggered"],
                },
            )

        blocked_steps = list(plan.metadata.get("blocked_steps", []))
        approval_required_steps = list(plan.metadata.get("approval_required_steps", []))
        relevant_memory_ids = list(plan.metadata.get("relevant_memory_ids", []))
        relevant_goal_ids = list(plan.metadata.get("relevant_goal_ids", []))
        relevant_belief_ids = list(plan.metadata.get("relevant_belief_ids", []))
        trigger_reason = plan.metadata.get("trigger_reason")

        return FacetResult(
            facet_name=self.name,
            summary=(
                f"Planner facet proposed {len(plan.steps)} steps at confidence "
                f"{plan.confidence:.3f}."
            ),
            proposed_decision=DecisionAction.RECORD,
            state_updates={
                "last_plan": plan.to_dict(),
                "last_plan_id": plan.id,
                "last_trigger_reason": trigger_reason,
                "last_plan_confidence": plan.confidence,
                "last_plan_pressure": plan.pressure,
                "last_relevant_memory_ids": relevant_memory_ids,
                "last_relevant_goal_ids": relevant_goal_ids,
                "last_relevant_belief_ids": relevant_belief_ids,
            },
            metadata={
                "plan": plan.to_dict(),
                "trigger_reason": trigger_reason,
                "query_intent": plan.metadata.get("query_intent"),
                "confidence": plan.confidence,
                "pressure": plan.pressure,
                "grounding_status": plan.metadata.get("grounding_status"),
                "grounding_score": plan.metadata.get("grounding_score"),
                "plan_memory_eligible": plan.metadata.get("plan_memory_eligible"),
                "plan_template_key": plan.metadata.get("plan_template_key"),
                "context_item_ids": list(plan.metadata.get("context_item_ids", [])),
                "relevant_memory_ids": relevant_memory_ids,
                "relevant_goal_ids": relevant_goal_ids,
                "relevant_belief_ids": relevant_belief_ids,
                "goal_ranking": list(plan.metadata.get("goal_ranking", [])),
                "memory_ranking": list(plan.metadata.get("memory_ranking", [])),
                "belief_ranking": list(plan.metadata.get("belief_ranking", [])),
                "selected_context_reasons": list(
                    plan.metadata.get("selected_context_reasons", [])
                ),
                "conflict_report": plan.metadata.get(
                    "conflict_report",
                    {"has_conflicts": False, "conflicts": [], "summary": None},
                ),
                "blocked_steps": blocked_steps,
                "approval_required_steps": approval_required_steps,
                "reasons": list(plan.reasons),
            },
        )
