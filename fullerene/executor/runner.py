"""Internal-only plan execution for Fullerene Executor v0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fullerene.executor.models import (
    ActionType,
    ExecutionMode,
    ExecutionRecord,
    ExecutionResult,
    ExecutionStatus,
    coerce_action_type,
)
from fullerene.goals import Goal, GoalStatus, GoalStore
from fullerene.memory import MemoryStore
from fullerene.planner import Plan, PlanStep, PlanStepStatus, RiskLevel
from fullerene.policy import PolicyStatus
from fullerene.world_model import Belief, BeliefStatus, WorldModelStore

EXTERNAL_TARGET_TYPES = frozenset(
    {"file_write", "file_delete", "shell", "network", "message", "git", "tool"}
)
ACTION_TARGETS: dict[ActionType, frozenset[str]] = {
    ActionType.UPDATE_MEMORY: frozenset({"memory", "internal_state"}),
    ActionType.UPDATE_GOAL: frozenset({"goal", "internal_state"}),
    ActionType.UPDATE_BELIEF: frozenset({"belief", "internal_state"}),
    ActionType.EMIT_EVENT: frozenset({"event", "internal_state"}),
    ActionType.NOOP: frozenset({"noop", "general", "internal_state"}),
}


@dataclass(slots=True)
class _PreparedAction:
    step: PlanStep
    action_type: ActionType
    target_type: str
    payload: dict[str, Any]
    goal: Goal | None = None
    belief: Belief | None = None


class InternalActionExecutor:
    """Execute approved internal actions and halt before partial mutation."""

    def __init__(
        self,
        *,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        memory_store: MemoryStore | None = None,
        state_dir: Path | str | None = None,
    ) -> None:
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.memory_store = memory_store
        self.state_dir = (
            Path(state_dir).expanduser().resolve() if state_dir is not None else None
        )

    def execute(
        self,
        plan_or_steps: Plan | Sequence[PlanStep],
        *,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
    ) -> ExecutionResult:
        dry_run = mode != ExecutionMode.LIVE
        plan_id, steps = self._normalize_input(plan_or_steps)
        if not steps:
            return ExecutionResult(
                plan_id=plan_id,
                overall_status=ExecutionStatus.SKIPPED,
                halted=False,
                dry_run=dry_run,
                reasons=["no_steps_to_execute"],
                metadata={"mode": mode.value},
            )

        prepared: list[_PreparedAction] = []
        for step in steps:
            action, record = self._preflight(step, plan_id=plan_id, dry_run=dry_run)
            if record is not None:
                return ExecutionResult(
                    plan_id=plan_id,
                    records=[record],
                    overall_status=record.status,
                    halted=True,
                    dry_run=dry_run,
                    reasons=[str(record.metadata.get("reason", ""))],
                    metadata={"mode": mode.value, "preflight_failed": True},
                )
            assert action is not None
            prepared.append(action)

        records = [
            self._execute_action(action, plan_id=plan_id, dry_run=dry_run)
            for action in prepared
        ]
        emitted_events = [
            record.metadata["emitted_event"]
            for record in records
            if isinstance(record.metadata.get("emitted_event"), dict)
        ]
        return ExecutionResult(
            plan_id=plan_id,
            records=records,
            overall_status=ExecutionStatus.SUCCESS,
            halted=False,
            dry_run=dry_run,
            reasons=["execution_completed"],
            metadata={
                "mode": mode.value,
                "preflight_failed": False,
                "emitted_events": emitted_events,
            },
        )

    @staticmethod
    def _normalize_input(
        plan_or_steps: Plan | Sequence[PlanStep],
    ) -> tuple[str | None, list[PlanStep]]:
        if isinstance(plan_or_steps, Plan):
            return plan_or_steps.id, list(plan_or_steps.steps)
        steps = list(plan_or_steps)
        return None, sorted(steps, key=lambda step: (step.order, step.id))

    def _preflight(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
    ) -> tuple[_PreparedAction | None, ExecutionRecord | None]:
        step_status = self._coerce_status(step.status)
        risk = self._coerce_risk(step.risk_level)
        target_type = self._coerce_target(step.target_type)
        policy_status = self._coerce_policy(step.policy_status)

        if bool(step.requires_approval) or step_status == PlanStepStatus.REQUIRES_APPROVAL:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="requires_approval",
                message="Step requires approval and Executor v0 cannot run it.",
                target_type=target_type,
                policy_status=policy_status,
                risk_level=risk,
            )
        if step_status == PlanStepStatus.BLOCKED or policy_status == PolicyStatus.DENIED.value:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="blocked_by_policy",
                message="Step is blocked by policy and cannot execute.",
                target_type=target_type,
                policy_status=policy_status,
                risk_level=risk,
            )
        if policy_status == PolicyStatus.APPROVAL_REQUIRED.value:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="requires_approval",
                message="Policy requires approval for this step.",
                target_type=target_type,
                policy_status=policy_status,
                risk_level=risk,
            )
        if risk == RiskLevel.HIGH.value:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="high_risk_not_allowed_v0",
                message="High-risk steps are not allowed in Executor v0.",
                target_type=target_type,
                policy_status=policy_status,
                risk_level=risk,
            )
        if target_type in EXTERNAL_TARGET_TYPES:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="unsupported_external_action_v0",
                message="Executor v0 refuses external side-effect target types.",
                target_type=target_type,
                policy_status=policy_status,
                risk_level=risk,
            )
        if target_type == "internal_state":
            outside_record = self._check_internal_state_path(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                policy_status=policy_status,
                risk_level=risk,
            )
            if outside_record is not None:
                return None, outside_record

        action_type = self._resolve_action_type(step)
        if action_type is None or target_type not in ACTION_TARGETS[action_type]:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="unsupported_action_type",
                message="Step does not declare a supported Executor v0 action.",
                target_type=target_type,
                action_type=action_type,
                policy_status=policy_status,
                risk_level=risk,
            )

        if action_type == ActionType.NOOP:
            return _PreparedAction(step, action_type, target_type, {}), None
        if action_type == ActionType.EMIT_EVENT:
            return self._prepare_emit_event(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                target_type=target_type,
            )
        if action_type == ActionType.UPDATE_GOAL:
            return self._prepare_goal_update(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                target_type=target_type,
            )
        if action_type == ActionType.UPDATE_BELIEF:
            return self._prepare_belief_update(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                target_type=target_type,
            )
        return self._prepare_memory_update(
            step,
            plan_id=plan_id,
            dry_run=dry_run,
            target_type=target_type,
        )

    def _prepare_emit_event(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        target_type: str,
    ) -> tuple[_PreparedAction | None, ExecutionRecord | None]:
        raw_event = step.metadata.get("event")
        if raw_event is not None and not isinstance(raw_event, dict):
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="emit_event requires dict-like metadata['event'] when present.",
                target_type=target_type,
                action_type=ActionType.EMIT_EVENT,
            )
        event_payload = dict(raw_event or {})
        if "content" not in event_payload and step.description:
            event_payload["content"] = step.description
        event_payload.setdefault("event_type", "system_note")
        return (
            _PreparedAction(
                step=step,
                action_type=ActionType.EMIT_EVENT,
                target_type=target_type,
                payload={"event": event_payload},
            ),
            None,
        )

    def _prepare_goal_update(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        target_type: str,
    ) -> tuple[_PreparedAction | None, ExecutionRecord | None]:
        goal_id = self._coerce_string(step.metadata.get("goal_id"))
        raw_status = step.metadata.get("status", step.metadata.get("goal_status"))
        if goal_id is None or raw_status is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="update_goal requires goal_id and status.",
                target_type=target_type,
                action_type=ActionType.UPDATE_GOAL,
            )
        try:
            status = GoalStatus(str(raw_status).strip().lower())
        except ValueError:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="update_goal status is invalid.",
                target_type=target_type,
                action_type=ActionType.UPDATE_GOAL,
            )
        goal = self.goal_store.get_goal(goal_id) if self.goal_store else None
        if self.goal_store is not None and goal is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="unknown_goal",
                message=f"Goal {goal_id!r} does not exist.",
                target_type=target_type,
                action_type=ActionType.UPDATE_GOAL,
            )
        if not dry_run and self.goal_store is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="unsupported_live_action",
                message="Live update_goal is unsupported without a goal store.",
                target_type=target_type,
                action_type=ActionType.UPDATE_GOAL,
            )
        return (
            _PreparedAction(
                step=step,
                action_type=ActionType.UPDATE_GOAL,
                target_type=target_type,
                payload={"goal_id": goal_id, "status": status},
                goal=goal,
            ),
            None,
        )

    def _prepare_belief_update(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        target_type: str,
    ) -> tuple[_PreparedAction | None, ExecutionRecord | None]:
        belief_id = self._coerce_string(step.metadata.get("belief_id"))
        if belief_id is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="update_belief requires belief_id.",
                target_type=target_type,
                action_type=ActionType.UPDATE_BELIEF,
            )
        status = None
        if "status" in step.metadata or "belief_status" in step.metadata:
            raw_status = step.metadata.get("status", step.metadata.get("belief_status"))
            try:
                status = BeliefStatus(str(raw_status).strip().lower())
            except ValueError:
                return None, self._terminal_record(
                    step,
                    plan_id=plan_id,
                    dry_run=dry_run,
                    status=ExecutionStatus.FAILED,
                    reason="invalid_action_payload",
                    message="update_belief status is invalid.",
                    target_type=target_type,
                    action_type=ActionType.UPDATE_BELIEF,
                )
        confidence = step.metadata.get("confidence")
        if status is None and not isinstance(confidence, (int, float)):
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="update_belief requires status or numeric confidence.",
                target_type=target_type,
                action_type=ActionType.UPDATE_BELIEF,
            )
        belief = self.world_model_store.get_belief(belief_id) if self.world_model_store else None
        if self.world_model_store is not None and belief is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="unknown_belief",
                message=f"Belief {belief_id!r} does not exist.",
                target_type=target_type,
                action_type=ActionType.UPDATE_BELIEF,
            )
        if not dry_run and self.world_model_store is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="unsupported_live_action",
                message="Live update_belief is unsupported without a world-model store.",
                target_type=target_type,
                action_type=ActionType.UPDATE_BELIEF,
            )
        return (
            _PreparedAction(
                step=step,
                action_type=ActionType.UPDATE_BELIEF,
                target_type=target_type,
                payload={
                    "belief_id": belief_id,
                    "status": status,
                    "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                },
                belief=belief,
            ),
            None,
        )

    def _prepare_memory_update(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        target_type: str,
    ) -> tuple[_PreparedAction | None, ExecutionRecord | None]:
        memory_id = self._coerce_string(step.metadata.get("memory_id"))
        if memory_id is None:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="update_memory requires memory_id.",
                target_type=target_type,
                action_type=ActionType.UPDATE_MEMORY,
            )
        if not dry_run:
            return None, self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="unsupported_live_action",
                message="Live update_memory is unsupported in Executor v0.",
                target_type=target_type,
                action_type=ActionType.UPDATE_MEMORY,
            )
        return (
            _PreparedAction(
                step=step,
                action_type=ActionType.UPDATE_MEMORY,
                target_type=target_type,
                payload={"memory_id": memory_id},
            ),
            None,
        )

    def _execute_action(
        self,
        action: _PreparedAction,
        *,
        plan_id: str | None,
        dry_run: bool,
    ) -> ExecutionRecord:
        if action.action_type == ActionType.NOOP:
            return self._success_record(
                action.step,
                plan_id=plan_id,
                dry_run=dry_run,
                action_type=action.action_type,
                message="Dry-run validated noop action." if dry_run else "Executed noop action.",
                metadata={"target_type": action.target_type},
            )
        if action.action_type == ActionType.EMIT_EVENT:
            return self._success_record(
                action.step,
                plan_id=plan_id,
                dry_run=dry_run,
                action_type=action.action_type,
                message="Dry-run captured internal event emission." if dry_run else "Recorded internal event emission.",
                metadata={
                    "target_type": action.target_type,
                    "emitted_event": dict(action.payload["event"]),
                },
            )
        if action.action_type == ActionType.UPDATE_GOAL:
            goal_id = action.payload["goal_id"]
            status = action.payload["status"]
            if not dry_run:
                assert self.goal_store is not None
                assert action.goal is not None
                goal = Goal.from_dict(action.goal.to_dict())
                goal.status = status
                self.goal_store.update_goal(goal)
            return self._success_record(
                action.step,
                plan_id=plan_id,
                dry_run=dry_run,
                action_type=action.action_type,
                message=f"Dry-run would update goal {goal_id!r}." if dry_run else f"Updated goal {goal_id!r}.",
                metadata={
                    "target_type": action.target_type,
                    "goal_id": goal_id,
                    "status": status.value,
                },
            )
        if action.action_type == ActionType.UPDATE_BELIEF:
            belief_id = action.payload["belief_id"]
            if not dry_run:
                assert self.world_model_store is not None
                assert action.belief is not None
                belief = Belief.from_dict(action.belief.to_dict())
                if action.payload["status"] is not None:
                    belief.status = action.payload["status"]
                if action.payload["confidence"] is not None:
                    belief.confidence = Belief._validate_confidence(action.payload["confidence"])
                self.world_model_store.update_belief(belief)
            return self._success_record(
                action.step,
                plan_id=plan_id,
                dry_run=dry_run,
                action_type=action.action_type,
                message=f"Dry-run would update belief {belief_id!r}." if dry_run else f"Updated belief {belief_id!r}.",
                metadata={
                    "target_type": action.target_type,
                    "belief_id": belief_id,
                    "status": (
                        action.payload["status"].value
                        if action.payload["status"] is not None
                        else None
                    ),
                    "confidence": action.payload["confidence"],
                },
            )
        return self._success_record(
            action.step,
            plan_id=plan_id,
            dry_run=dry_run,
            action_type=action.action_type,
            message=f"Dry-run would update memory {action.payload['memory_id']!r}.",
            metadata={
                "target_type": action.target_type,
                "memory_id": action.payload["memory_id"],
            },
        )

    def _check_internal_state_path(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        policy_status: str | None,
        risk_level: str | None,
    ) -> ExecutionRecord | None:
        raw_path = step.metadata.get("path")
        if raw_path is None or self.state_dir is None:
            return None
        path = self._coerce_string(raw_path)
        if path is None:
            return self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.FAILED,
                reason="invalid_action_payload",
                message="internal_state path must be a non-empty string.",
                target_type="internal_state",
                policy_status=policy_status,
                risk_level=risk_level,
            )
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.state_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.state_dir)
        except ValueError:
            return self._terminal_record(
                step,
                plan_id=plan_id,
                dry_run=dry_run,
                status=ExecutionStatus.SKIPPED,
                reason="outside_state_dir",
                message="internal_state paths must stay inside the configured state-dir.",
                target_type="internal_state",
                policy_status=policy_status,
                risk_level=risk_level,
                metadata={"resolved_path": str(candidate)},
            )
        return None

    @staticmethod
    def _resolve_action_type(step: PlanStep) -> ActionType | None:
        action_type = coerce_action_type(step.metadata.get("action_type"))
        if action_type is not None:
            return action_type
        return {
            "memory": ActionType.UPDATE_MEMORY,
            "goal": ActionType.UPDATE_GOAL,
            "belief": ActionType.UPDATE_BELIEF,
            "event": ActionType.EMIT_EVENT,
            "noop": ActionType.NOOP,
        }.get(InternalActionExecutor._coerce_target(step.target_type))

    @staticmethod
    def _coerce_target(raw_value: Any) -> str:
        cleaned = str(raw_value or "").strip().lower()
        return cleaned or "unknown"

    @staticmethod
    def _coerce_status(raw_value: Any) -> PlanStepStatus | None:
        if isinstance(raw_value, PlanStepStatus):
            return raw_value
        try:
            return PlanStepStatus(str(raw_value).strip().lower())
        except ValueError:
            return None

    @staticmethod
    def _coerce_risk(raw_value: Any) -> str | None:
        if isinstance(raw_value, RiskLevel):
            return raw_value.value
        cleaned = str(raw_value or "").strip().lower()
        return cleaned or None

    @staticmethod
    def _coerce_policy(raw_value: Any) -> str | None:
        cleaned = str(raw_value or "").strip().lower()
        return cleaned or None

    @staticmethod
    def _coerce_string(raw_value: Any) -> str | None:
        if not isinstance(raw_value, str):
            return None
        cleaned = raw_value.strip()
        return cleaned or None

    @staticmethod
    def _record_reason(record: ExecutionRecord) -> str | None:
        reason = record.metadata.get("reason")
        return reason if isinstance(reason, str) and reason.strip() else None

    @staticmethod
    def _fallback_action_type(target_type: str) -> ActionType:
        return {
            "memory": ActionType.UPDATE_MEMORY,
            "goal": ActionType.UPDATE_GOAL,
            "belief": ActionType.UPDATE_BELIEF,
            "event": ActionType.EMIT_EVENT,
        }.get(target_type, ActionType.NOOP)

    def _success_record(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        action_type: ActionType,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            action_type=action_type,
            plan_id=plan_id,
            plan_step_id=step.id,
            status=ExecutionStatus.SUCCESS,
            dry_run=dry_run,
            message=message,
            metadata={
                "step_order": step.order,
                "step_description": step.description,
                **dict(metadata or {}),
            },
        )

    def _terminal_record(
        self,
        step: PlanStep,
        *,
        plan_id: str | None,
        dry_run: bool,
        status: ExecutionStatus,
        reason: str,
        message: str,
        target_type: str,
        action_type: ActionType | None = None,
        policy_status: str | None = None,
        risk_level: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            action_type=action_type or self._fallback_action_type(target_type),
            plan_id=plan_id,
            plan_step_id=step.id,
            status=status,
            dry_run=dry_run,
            message=message,
            metadata={
                "reason": reason,
                "step_order": step.order,
                "step_description": step.description,
                "target_type": target_type,
                "policy_status": policy_status,
                "risk_level": risk_level,
                **dict(metadata or {}),
            },
        )
