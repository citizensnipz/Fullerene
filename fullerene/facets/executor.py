"""Deterministic execution facet for Fullerene Executor v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fullerene.executor import ExecutionMode, InternalActionExecutor
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.planner import Plan


class ExecutorFacet:
    """Execute approved internal plans only when explicitly requested."""

    name = "executor"

    def __init__(
        self,
        *,
        goal_store=None,
        world_model_store=None,
        memory_store=None,
        state_dir: Path | str | None = None,
    ) -> None:
        self.executor = InternalActionExecutor(
            goal_store=goal_store,
            world_model_store=world_model_store,
            memory_store=memory_store,
            state_dir=state_dir,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        if not self._metadata_flag(event.metadata, "execute_plan"):
            return FacetResult(
                facet_name=self.name,
                summary="Executor facet did not run because execute_plan was not requested.",
                proposed_decision=DecisionAction.WAIT,
                state_updates={
                    "last_execution_requested": False,
                    "last_execution_result": None,
                },
                metadata={
                    "execution_requested": False,
                    "execution_result": None,
                    "dry_run": True,
                    "reasons": ["execution_not_requested"],
                },
            )

        plan = self._load_plan(state)
        if plan is None:
            return FacetResult(
                facet_name=self.name,
                summary="Executor facet found no planner output to execute.",
                proposed_decision=DecisionAction.WAIT,
                state_updates={
                    "last_execution_requested": True,
                    "last_execution_result": None,
                },
                metadata={
                    "execution_requested": True,
                    "execution_result": None,
                    "dry_run": self._is_dry_run(event.metadata),
                    "reasons": ["no_plan_found"],
                },
            )

        mode = (
            ExecutionMode.DRY_RUN
            if self._is_dry_run(event.metadata)
            else ExecutionMode.LIVE
        )
        execution_result = self.executor.execute(plan, mode=mode)
        execution_payload = execution_result.to_dict()
        return FacetResult(
            facet_name=self.name,
            summary=(
                f"Executor facet {execution_result.overall_status.value} "
                f"{len(execution_result.records)} execution step(s)."
            ),
            proposed_decision=DecisionAction.RECORD,
            state_updates={
                "last_execution_requested": True,
                "last_execution_plan_id": execution_result.plan_id,
                "last_execution_result": execution_payload,
                "last_execution_status": execution_result.overall_status.value,
                "last_execution_halted": execution_result.halted,
                "last_execution_dry_run": execution_result.dry_run,
            },
            metadata={
                "execution_requested": True,
                "execution_result": execution_payload,
                "records": execution_payload["records"],
                "halted": execution_result.halted,
                "dry_run": execution_result.dry_run,
                "reasons": list(execution_result.reasons),
            },
        )

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

    @staticmethod
    def _is_dry_run(metadata: dict[str, Any]) -> bool:
        raw_value = metadata.get("dry_run")
        if raw_value is False:
            return False
        if isinstance(raw_value, str) and raw_value.strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        return True

    @staticmethod
    def _load_plan(state: NexusState) -> Plan | None:
        planner_state = state.facet_state.get("planner")
        if not isinstance(planner_state, dict):
            return None
        raw_plan = planner_state.get("last_plan")
        if not isinstance(raw_plan, dict):
            return None
        return Plan.from_dict(raw_plan)
