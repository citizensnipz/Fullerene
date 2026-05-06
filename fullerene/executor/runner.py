"""Manifest-backed execution engine for Fullerene Executor v1."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fullerene.executor.models import ActionType, ExecutionMode, ExecutionRecord, ExecutionResult, ExecutionStatus, SkillManifestEntry
from fullerene.executor.registry import SkillRegistry
from fullerene.executor.skills import register_builtin_skills
from fullerene.goals import Goal, GoalStatus, GoalStore
from fullerene.memory import MemoryStore
from fullerene.planner import Plan, PlanStep, PlanStepStatus, RiskLevel
from fullerene.policy import PolicyStatus
from fullerene.world_model import Belief, BeliefStatus, WorldModelStore


class InternalActionExecutor:
    """Execute deterministic manifest-registered skills only."""

    def __init__(
        self,
        *,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        memory_store: MemoryStore | None = None,
        state_dir: Path | str | None = None,
        sandbox_dir: Path | str | None = None,
    ) -> None:
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.memory_store = memory_store
        self.state_dir = (
            Path(state_dir).expanduser().resolve() if state_dir is not None else None
        )
        self.registry = SkillRegistry()
        register_builtin_skills(self.registry)
        self.pending_approvals: dict[str, dict[str, Any]] = {}
        self.approval_timeout_cycles = 3
        self.file_operation_log: list[dict[str, Any]] = []
        self.max_file_log_entries = 100
        if sandbox_dir is not None:
            self.sandbox_root = Path(sandbox_dir).expanduser().resolve()
        else:
            self.sandbox_root = (self.state_dir / "sandbox").resolve() if self.state_dir else None
        if self.sandbox_root is not None:
            self.sandbox_root.mkdir(parents=True, exist_ok=True)

    def register_skill(self, entry: SkillManifestEntry, handler) -> None:
        self.registry.register_skill(entry, handler)

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

        # Preflight all steps first: no partial execution.
        preflight_records: list[ExecutionRecord] = []
        for step in steps:
            preflight = self._preflight_step(step=step, plan_id=plan_id, dry_run=dry_run)
            if preflight is not None:
                preflight_records.append(preflight)
        if preflight_records:
            first = preflight_records[0]
            return ExecutionResult(
                plan_id=plan_id,
                records=preflight_records,
                overall_status=first.status,
                halted=True,
                dry_run=dry_run,
                reasons=[str(first.metadata.get("reason", "preflight_failed"))],
                metadata={"mode": mode.value, "preflight_failed": True, "file_operation_log": list(self.file_operation_log)},
            )

        records: list[ExecutionRecord] = []
        for index, step in enumerate(steps):
            record = self._execute_step(step=step, plan_id=plan_id, dry_run=dry_run)
            records.append(record)
            if record.status not in {ExecutionStatus.SUCCESS, ExecutionStatus.PENDING_APPROVAL}:
                for remaining in steps[index + 1 :]:
                    records.append(
                        self._make_record(
                            step=remaining,
                            plan_id=plan_id,
                            status=ExecutionStatus.SKIPPED,
                            dry_run=dry_run,
                            reason="skipped_due_to_prior_failure",
                            message="Skipped because a prior step failed.",
                        )
                    )
                return ExecutionResult(
                    plan_id=plan_id,
                    records=records,
                    overall_status=record.status,
                    halted=True,
                    dry_run=dry_run,
                    reasons=[self._record_reason(record) or "execution_failed"],
                    metadata={"mode": mode.value, "preflight_failed": False, "file_operation_log": list(self.file_operation_log)},
                )
            if record.status == ExecutionStatus.PENDING_APPROVAL:
                return ExecutionResult(
                    plan_id=plan_id,
                    records=records,
                    overall_status=ExecutionStatus.PENDING_APPROVAL,
                    halted=True,
                    dry_run=dry_run,
                    reasons=["pending_approval"],
                    metadata={"mode": mode.value, "preflight_failed": False, "file_operation_log": list(self.file_operation_log)},
                )

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
                "file_operation_log": list(self.file_operation_log),
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

    def _preflight_step(self, *, step: PlanStep, plan_id: str | None, dry_run: bool) -> ExecutionRecord | None:
        target_type = self._coerce_target(step.target_type)
        action_type = self._resolve_action_type(step, target_type=target_type)
        declared_action = str(step.metadata.get("action_type") or "").strip().lower()
        action_name = action_type.value if action_type else declared_action
        skill_name = str(step.metadata.get("skill_name") or self._default_skill_name(action_type)).strip().lower()
        policy_status = self._coerce_policy(step.policy_status)

        if not dry_run and policy_status != PolicyStatus.ALLOWED.value:
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.FAILED, dry_run=dry_run, reason="policy_not_allowed_for_live", message="Live execution requires policy allowed.", skill_name=skill_name, target_type=target_type, policy_status=policy_status, action_type_name=action_name)
        if step.status == PlanStepStatus.BLOCKED or policy_status == PolicyStatus.DENIED.value:
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.SKIPPED, dry_run=dry_run, reason="blocked_by_policy", message="Step is blocked by policy.", skill_name=skill_name, target_type=target_type, policy_status=policy_status, action_type_name=action_name)
        if not action_name:
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.FAILED, dry_run=dry_run, reason="unsupported_action_type", message="Unsupported action type.", skill_name=skill_name, target_type=target_type, policy_status=policy_status)

        ok, reason = self.registry.validate_skill_invocation(skill_name=skill_name, action_type=action_name, target_type=target_type)
        if not ok:
            status = ExecutionStatus.FAILED if reason == "skill_not_registered" else ExecutionStatus.SKIPPED
            return self._make_record(step=step, plan_id=plan_id, status=status, dry_run=dry_run, reason=reason, message="Skill invocation rejected by registry.", skill_name=skill_name, target_type=target_type, policy_status=policy_status, action_type_name=action_name)
        return None

    def _execute_step(self, *, step: PlanStep, plan_id: str | None, dry_run: bool) -> ExecutionRecord:
        target_type = self._coerce_target(step.target_type)
        action_type = self._resolve_action_type(step, target_type=target_type)
        action_name = action_type.value if action_type else str(step.metadata.get("action_type") or "").strip().lower()
        skill_name = str(step.metadata.get("skill_name") or self._default_skill_name(action_type)).strip().lower()
        registered = self.registry.get_skill(skill_name)
        policy_status = self._coerce_policy(step.policy_status)
        cycle = int(step.metadata.get("cycle_id", 0) or 0)

        if registered is None:
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.FAILED, dry_run=dry_run, reason="skill_not_registered", message="Skill is not registered.", skill_name=skill_name, target_type=target_type, policy_status=policy_status, action_type_name=action_name)
        entry = registered.entry

        requires_approval = bool(step.requires_approval or step.status == PlanStepStatus.REQUIRES_APPROVAL or policy_status == PolicyStatus.APPROVAL_REQUIRED.value or entry.requires_approval)
        approval = self._extract_approval(step.metadata.get("approval"))
        if requires_approval and approval is None:
            approval_id = f"approval-{step.id}"
            pending = self.pending_approvals.get(approval_id)
            if pending is None:
                pending = {"approval_id": approval_id, "requested_cycle": cycle, "expires_after_cycles": self.approval_timeout_cycles}
                self.pending_approvals[approval_id] = pending
            elapsed = max(0, cycle - int(pending.get("requested_cycle", cycle)))
            if elapsed >= int(pending.get("expires_after_cycles", self.approval_timeout_cycles)):
                return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.APPROVAL_TIMEOUT, dry_run=dry_run, reason="approval_timeout", message="Approval timed out.", skill_name=skill_name, skill_version=entry.version, target_type=target_type, policy_status=policy_status, action_type_name=action_name, approval_status="timed_out")
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.PENDING_APPROVAL, dry_run=dry_run, reason="pending_approval", message="Awaiting approval.", skill_name=skill_name, skill_version=entry.version, target_type=target_type, policy_status=policy_status, action_type_name=action_name, approval_status="pending")

        try:
            payload = dict(step.metadata)
            response = self._dispatch_skill(registered=registered, payload=payload, dry_run=dry_run, step=step)
        except Exception as exc:
            return self._make_record(step=step, plan_id=plan_id, status=ExecutionStatus.FAILED, dry_run=dry_run, reason="execution_failed", message=f"Execution failed: {exc}", skill_name=skill_name, skill_version=entry.version, target_type=target_type, policy_status=policy_status, action_type_name=action_name)

        success = bool(response.get("success", False))
        status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
        reason = str(response.get("reason", "ok" if success else "execution_failed"))
        sandbox_status = "ok" if entry.sandbox_required else "not_required"
        metadata = dict(response)
        if skill_name in {"file_read", "file_write", "file_list"}:
            self._append_file_op(skill_name=skill_name, step_id=step.id, payload=payload, dry_run=dry_run, response=response)
        return self._make_record(
            step=step,
            plan_id=plan_id,
            status=status,
            dry_run=dry_run,
            reason=reason,
            message="Skill executed." if success else "Skill execution failed.",
            skill_name=skill_name,
            skill_version=entry.version,
            target_type=target_type,
            policy_status=policy_status,
            action_type_name=action_name,
            sandbox_status=sandbox_status,
            metadata=metadata,
            retryable=not success,
            requires_replan=not success and reason not in {"approval_timeout", "blocked_by_policy"},
        )

    def _dispatch_skill(self, *, registered, payload: dict[str, Any], dry_run: bool, step: PlanStep) -> dict[str, Any]:
        handler = registered.handler
        skill_name = registered.entry.skill_name
        if skill_name in {"goal_update", "world_model_belief_update", "internal_event", "memory_write"}:
            return self._dispatch_legacy(skill_name=skill_name, payload=payload, dry_run=dry_run)
        return handler(payload=payload, dry_run=dry_run, sandbox_root=self.sandbox_root, plan_step=step)

    def _dispatch_legacy(self, *, skill_name: str, payload: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        if skill_name == "internal_event":
            event_payload = payload.get("event")
            if not isinstance(event_payload, dict):
                return {"success": False, "reason": "invalid_action_payload"}
            return {"success": True, "emitted_event": dict(event_payload)}
        if skill_name == "goal_update":
            goal_id = self._coerce_string(payload.get("goal_id"))
            raw_status = payload.get("status")
            if not goal_id or raw_status is None:
                return {"success": False, "reason": "invalid_action_payload"}
            try:
                status = GoalStatus(str(raw_status).strip().lower())
            except ValueError:
                return {"success": False, "reason": "invalid_action_payload"}
            goal = self.goal_store.get_goal(goal_id) if self.goal_store else None
            if self.goal_store and goal is None:
                return {"success": False, "reason": "unknown_goal"}
            if not dry_run and self.goal_store and goal:
                updated = Goal.from_dict(goal.to_dict())
                updated.status = status
                self.goal_store.update_goal(updated)
            return {"success": True, "goal_id": goal_id, "status": status.value}
        if skill_name == "world_model_belief_update":
            belief_id = self._coerce_string(payload.get("belief_id"))
            if not belief_id:
                return {"success": False, "reason": "invalid_action_payload"}
            belief = self.world_model_store.get_belief(belief_id) if self.world_model_store else None
            if self.world_model_store and belief is None:
                return {"success": False, "reason": "unknown_belief"}
            if not dry_run and self.world_model_store and belief:
                updated = Belief.from_dict(belief.to_dict())
                if isinstance(payload.get("confidence"), (int, float)):
                    updated.confidence = Belief._validate_confidence(payload["confidence"])
                self.world_model_store.update_belief(updated)
            return {"success": True, "belief_id": belief_id}
        if skill_name == "memory_write":
            return {"success": dry_run, "reason": "unsupported_live_action" if not dry_run else "ok"}
        return {"success": False, "reason": "skill_not_registered"}

    def _append_file_op(self, *, skill_name: str, step_id: str, payload: dict[str, Any], dry_run: bool, response: dict[str, Any]) -> None:
        op = "read" if skill_name == "file_read" else "write" if skill_name == "file_write" else "list"
        row = {
            "operation_id": f"file-op-{len(self.file_operation_log) + 1}",
            "skill_name": skill_name,
            "operation": op,
            "requested_path": payload.get("path", "."),
            "resolved_relative_path": response.get("resolved_relative_path"),
            "dry_run": dry_run,
            "success": bool(response.get("success", False)),
            "bytes_read": response.get("bytes_read"),
            "bytes_written": response.get("bytes_written"),
            "reason": response.get("reason"),
            "plan_step_id": step_id,
        }
        self.file_operation_log.append(row)
        self.file_operation_log = self.file_operation_log[-self.max_file_log_entries :]

    @staticmethod
    def _extract_approval(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        if raw.get("approved") is not True:
            return None
        return dict(raw)

    @staticmethod
    def _default_skill_name(action_type: ActionType | None) -> str:
        if action_type == ActionType.UPDATE_MEMORY:
            return "memory_write"
        if action_type == ActionType.UPDATE_GOAL:
            return "goal_update"
        if action_type == ActionType.UPDATE_BELIEF:
            return "world_model_belief_update"
        if action_type == ActionType.EMIT_EVENT:
            return "internal_event"
        if action_type == ActionType.NOOP:
            return "internal_event"
        return ""

    def _make_record(
        self,
        *,
        step: PlanStep,
        plan_id: str | None,
        status: ExecutionStatus,
        dry_run: bool,
        reason: str,
        message: str,
        skill_name: str = "",
        skill_version: str = "v1",
        action_type_name: str = "",
        target_type: str = "",
        policy_status: str | None = None,
        approval_status: str | None = None,
        sandbox_status: str | None = None,
        metadata: dict[str, Any] | None = None,
        retryable: bool = False,
        requires_replan: bool = False,
    ) -> ExecutionRecord:
        merged = {"reason": reason, "step_order": step.order, "target_type": target_type, **dict(metadata or {})}
        if action_type_name:
            merged.setdefault("action_type", action_type_name)
        if policy_status:
            merged.setdefault("policy_status", policy_status)
        return ExecutionRecord(
            action_type=self._resolve_action_type(step, target_type=target_type),
            plan_id=plan_id,
            plan_step_id=step.id,
            status=status,
            dry_run=dry_run,
            message=message,
            metadata=merged,
            skill_name=skill_name,
            skill_version=skill_version,
            action_type_name=action_type_name,
            target_type=target_type,
            policy_status=policy_status,
            approval_status=approval_status,
            sandbox_status=sandbox_status,
            retryable=retryable,
            requires_replan=requires_replan,
        )

    @staticmethod
    def _resolve_action_type(
        step: PlanStep,
        *,
        target_type: str,
    ) -> ActionType | None:
        raw = step.metadata.get("action_type")
        if isinstance(raw, ActionType):
            return raw
        if isinstance(raw, str):
            cleaned = raw.strip().lower()
            for action in ActionType:
                if action.value == cleaned:
                    return action
            return None
        return ActionType.NOOP if target_type == "noop" else None

    @staticmethod
    def _coerce_target(raw_value: Any) -> str:
        cleaned = str(raw_value or "").strip().lower()
        return cleaned or "unknown"

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
