"""Deterministic artifact schema validators for Verifier v1 (no LLM, no network)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from fullerene.executor.models import ExecutionStatus, coerce_action_type
from fullerene.learning.models import AdjustmentStatus, AdjustmentTarget
from fullerene.nexus.models import DecisionAction
from fullerene.planner.models import PlanStatus, PlanStepStatus, RiskLevel

DECISION_STRINGS = frozenset({"wait", "record", "ask", "act"})
KNOWN_POLICY_STATUS = frozenset(
    {
        "allowed",
        "allow",
        "denied",
        "approval_required",
        "no_match",
        "unknown",
        "preferred",
    }
)
CANONICAL_PRESSURE_KEYS = (
    "event_pressure",
    "attention_pressure",
    "latent_pressure",
    "contradiction_pressure",
    "context_overload_pressure",
    "interrupt_pressure",
)


def _clamp01(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def artifact_result(
    *,
    validator: str,
    artifact_kind: str,
    status: str,
    severity: str,
    code: str,
    path: str,
    message: str,
    retry_recommended: bool = False,
    escalation_recommended: bool = False,
) -> dict[str, Any]:
    return {
        "validator": validator,
        "artifact_kind": artifact_kind,
        "status": status,
        "severity": severity,
        "code": code,
        "path": path,
        "message": message,
        "retry_recommended": bool(retry_recommended),
        "escalation_recommended": bool(escalation_recommended),
    }


def is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_unit_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    v = float(value)
    return 0.0 <= v <= 1.0


def _coerce_decision_str(value: Any) -> str | None:
    if isinstance(value, DecisionAction):
        return value.value
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    return s if s in DECISION_STRINGS else None


def validate_behavior_decision_trace_v2(
    trace: Mapping[str, Any] | None,
    *,
    decision_is_act: bool,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    """Returns (check rows, recommended downgrade or None)."""
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if not trace:
        return out, reco

    artifact_kind = "behavior_decision_trace_v2"
    event_obj = trace.get("event")
    if not isinstance(event_obj, Mapping):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="error",
                code="missing_required_field",
                path="behavior.decision_trace.event",
                message="decision_trace.event must be dict-like when trace is present.",
                retry_recommended=decision_is_act,
                escalation_recommended=decision_is_act,
            )
        )
        if decision_is_act:
            reco = DecisionAction.ASK
    else:
        eid = event_obj.get("event_id") or event_obj.get("id")
        et = event_obj.get("event_type") or event_obj.get("type")
        summary = event_obj.get("content_summary") or event_obj.get("content")
        if not eid or not et or summary is None:
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="error",
                    code="missing_required_field",
                    path="behavior.decision_trace.event",
                    message="event must expose id/event_id, type/event_type, and content_summary (or content).",
                    retry_recommended=True,
                    escalation_recommended=decision_is_act,
                )
            )
            if decision_is_act:
                reco = DecisionAction.ASK

    pairs = [
        ("pressure_score", trace.get("pressure_score")),
        ("latent_pressure", trace.get("latent_pressure")),
        ("memory_relevance_score", trace.get("memory_relevance_score")),
        ("goal_relevance_score", trace.get("goal_relevance_score")),
        ("world_model_belief_confidence", trace.get("world_model_belief_confidence")),
        ("context_load_ratio", trace.get("context_load_ratio")),
        ("confidence", trace.get("confidence")),
    ]
    for path_key, raw in pairs:
        if raw is None:
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="warning",
                    code="missing_optional_numeric",
                    path=f"behavior.decision_trace.{path_key}",
                    message=f"Expected numeric [0,1] for {path_key} when trace is present.",
                    retry_recommended=True,
                )
            )
        elif not _is_unit_number(raw):
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="error",
                    code="numeric_out_of_range",
                    path=f"behavior.decision_trace.{path_key}",
                    message=f"{path_key} must be numeric in [0,1].",
                    retry_recommended=decision_is_act,
                    escalation_recommended=decision_is_act,
                )
            )
            if decision_is_act:
                reco = DecisionAction.ASK

    if "contradiction_flag" in trace and not isinstance(trace.get("contradiction_flag"), bool):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="warning",
                code="invalid_type",
                path="behavior.decision_trace.contradiction_flag",
                message="contradiction_flag must be boolean when present.",
            )
        )

    if "policy_result" in trace and not isinstance(trace.get("policy_result"), str):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="warning",
                code="invalid_type",
                path="behavior.decision_trace.policy_result",
                message="policy_result must be a string when present.",
            )
        )

    for key in ("raw_candidate_scores", "adjusted_candidate_scores"):
        blob = trace.get(key)
        if blob is None:
            continue
        if not isinstance(blob, Mapping):
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="warning",
                    code="invalid_type",
                    path=f"behavior.decision_trace.{key}",
                    message=f"{key} must be dict-like when present.",
                )
            )
            continue
        for dec in DECISION_STRINGS:
            if dec not in blob:
                out.append(
                    artifact_result(
                        validator="behavior_decision_trace_schema",
                        artifact_kind=artifact_kind,
                        status="failed",
                        severity="warning",
                        code="missing_score_key",
                        path=f"behavior.decision_trace.{key}.{dec}",
                        message=f"{key} missing {dec} score when present.",
                    )
                )
            elif not _is_unit_number(blob.get(dec)):
                out.append(
                    artifact_result(
                        validator="behavior_decision_trace_schema",
                        artifact_kind=artifact_kind,
                        status="failed",
                        severity="warning",
                        code="invalid_numeric",
                        path=f"behavior.decision_trace.{key}.{dec}",
                        message=f"{key}.{dec} must be numeric in [0,1].",
                    )
                )

    fd = _coerce_decision_str(trace.get("final_decision"))
    if fd is None and "final_decision" in trace:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="error",
                code="invalid_enum",
                path="behavior.decision_trace.final_decision",
                message="final_decision must be wait|record|ask|act.",
                retry_recommended=True,
                escalation_recommended=decision_is_act,
            )
        )
        if decision_is_act:
            reco = DecisionAction.ASK

    if "reasons" not in trace or not isinstance(trace.get("reasons"), list):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="warning",
                code="missing_reasons",
                path="behavior.decision_trace.reasons",
                message="reasons list is required when trace is present.",
                retry_recommended=True,
            )
        )

    if "interrupt_recommended" in trace and not isinstance(
        trace.get("interrupt_recommended"), bool
    ):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="warning",
                code="invalid_type",
                path="behavior.decision_trace.interrupt_recommended",
                message="interrupt_recommended must be boolean when present.",
            )
        )

    if not trace.get("timestamp"):
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="failed",
                severity="warning",
                code="missing_timestamp",
                path="behavior.decision_trace.timestamp",
                message="timestamp is required when trace is present.",
                retry_recommended=True,
            )
        )

    has_failed = any(r.get("status") == "failed" for r in out)
    if not has_failed:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_schema",
                artifact_kind=artifact_kind,
                status="passed",
                severity="info",
                code="ok",
                path="behavior.decision_trace",
                message="Behavior v2 decision trace schema satisfied.",
            )
        )
    elif decision_is_act and reco is None:
        reco = DecisionAction.ASK
    return out, reco


def validate_cycle_signal_map_pressure(
    signal_map: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not signal_map:
        return out
    kind = "nexus_cycle_signal_map"
    missing = [k for k in CANONICAL_PRESSURE_KEYS if k not in signal_map.get("pressure_components", {})]
    if missing:
        out.append(
            artifact_result(
                validator="cycle_signal_map_pressure",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="missing_pressure_component",
                path="nexus.signal_map.pressure_components",
                message=f"pressure_components missing canonical keys: {sorted(missing)}.",
            )
        )

    comp = signal_map.get("pressure_components")
    if isinstance(comp, Mapping):
        total = sum(float(v) for v in comp.values() if isinstance(v, (int, float)))
        total = _clamp01(total)
        sys_p_raw = signal_map.get("system_pressure", 0.0)
        try:
            sys_p = float(sys_p_raw)
        except (TypeError, ValueError):
            sys_p = None
        if sys_p is not None:
            diff = abs(round(total, 3) - round(_clamp01(sys_p), 3))
            if diff > 0.35:
                out.append(
                    artifact_result(
                        validator="cycle_signal_map_pressure",
                        artifact_kind=kind,
                        status="failed",
                        severity="error",
                        code="system_pressure_mismatch",
                        path="nexus.signal_map.system_pressure",
                        message=(
                            "system_pressure differs strongly from sum(pressure_components); "
                            f"expected≈{total:.3f} got {sys_p:.3f}."
                        ),
                    )
                )
            elif diff > 0.15:
                out.append(
                    artifact_result(
                        validator="cycle_signal_map_pressure",
                        artifact_kind=kind,
                        status="failed",
                        severity="warning",
                        code="system_pressure_mismatch",
                        path="nexus.signal_map.system_pressure",
                        message=(
                            "system_pressure differs from clamped sum(pressure_components): "
                            f"expected≈{total:.3f} got {sys_p:.3f}."
                        ),
                    )
                )
            elif diff > 0.02:
                out.append(
                    artifact_result(
                        validator="cycle_signal_map_pressure",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="system_pressure_mismatch",
                        path="nexus.signal_map.system_pressure",
                        message=(
                            "system_pressure loosely matches sum(pressure_components); verify aggregation."
                        ),
                    )
                )

    out.append(
        artifact_result(
            validator="cycle_signal_map_pressure",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="nexus.signal_map",
            message="CycleSignalMap pressure aggregation checked.",
        )
    )
    return out


def validate_cycle_trace_structure(
    trace: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    """Returns checks and optional downgrade (critical internal routing)."""
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if not trace:
        return out, reco
    kind = "nexus_cycle_trace"
    required = (
        "facet_order",
        "signal_map",
        "pressure_components",
        "final_decision",
        "learning_events",
        "internal_events_queued",
        "internal_events_processed",
    )
    missing = [k for k in required if k not in trace]
    if missing:
        out.append(
            artifact_result(
                validator="cycle_trace_schema",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="missing_cycle_trace_field",
                path="nexus.cycle_trace",
                message=f"cycle trace missing fields: {sorted(missing)}.",
            )
        )

    if trace.get("signal_map") in (None, {}) and "signal_map" in trace:
        out.append(
            artifact_result(
                validator="cycle_trace_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="missing_signal_map",
                path="nexus.cycle_trace.signal_map",
                message="signal_map must be present and non-empty in cycle trace.",
            )
        )

    if trace.get("final_decision") is None:
        out.append(
            artifact_result(
                validator="cycle_trace_schema",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="missing_final_decision",
                path="nexus.cycle_trace.final_decision",
                message="final_decision should be set on cycle trace.",
            )
        )

    processed = trace.get("internal_events_processed")
    if isinstance(processed, Sequence) and not isinstance(processed, (str, bytes)):
        if len(processed) > 1:
            out.append(
                artifact_result(
                    validator="cycle_trace_internal_events",
                    artifact_kind=kind,
                    status="failed",
                    severity="critical",
                    code="internal_events_over_limit",
                    path="nexus.cycle_trace.internal_events_processed",
                    message="At most one internal event may be processed per outer event.",
                    escalation_recommended=True,
                )
            )
            reco = DecisionAction.RECORD

    summ_status = "passed"
    summ_severity = "info"
    if any(r.get("severity") == "critical" for r in out):
        summ_status = "failed"
        summ_severity = "critical"
    elif any(r.get("status") == "failed" for r in out):
        summ_status = "failed"
        summ_severity = "error"

    out.append(
        artifact_result(
            validator="cycle_trace_schema",
            artifact_kind=kind,
            status=summ_status,
            severity=summ_severity,
            code="summary",
            path="nexus.cycle_trace",
            message="cycle trace structural validation complete.",
        )
    )
    return out, reco


def _step_action_type(step: Mapping[str, Any]) -> str | None:
    md = step.get("metadata")
    if isinstance(md, Mapping):
        raw = md.get("action_type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return None


def validate_planner_plan_schema(
    plan: Mapping[str, Any] | None,
    *,
    executable_intent: bool,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if plan is None:
        return out, reco
    kind = "planner_plan"

    for key in ("id", "status"):
        if key not in plan:
            out.append(
                artifact_result(
                    validator="planner_plan_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="missing_required_field",
                    path=f"planner.plan.{key}",
                    message=f"plan.{key} is required when plan is present.",
                )
            )

    if "confidence" in plan and not _is_unit_number(plan.get("confidence")):
        out.append(
            artifact_result(
                validator="planner_plan_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="plan_confidence_range",
                path="planner.plan.confidence",
                message="plan.confidence must be numeric in [0,1] when present.",
            )
        )

    pmd_early = plan.get("metadata") if isinstance(plan.get("metadata"), Mapping) else {}
    gs = plan.get("grounding_status") or pmd_early.get("grounding_status")
    if gs is not None and not isinstance(gs, str):
        out.append(
            artifact_result(
                validator="planner_plan_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="invalid_grounding_status",
                path="planner.plan.metadata.grounding_status",
                message="grounding_status must be string when present.",
            )
        )

    if "grounding_score" in plan.get("metadata", {}) or "grounding_score" in plan:
        gscore = plan.get("grounding_score")
        if gscore is None and isinstance(plan.get("metadata"), Mapping):
            gscore = plan["metadata"].get("grounding_score")
        if gscore is not None and not _is_unit_number(gscore):
            out.append(
                artifact_result(
                    validator="planner_plan_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="invalid_grounding_score",
                    path="planner.plan.grounding_score",
                    message="grounding_score must be numeric in [0,1] when present.",
                )
            )

    pmd = plan.get("metadata")
    if isinstance(pmd, Mapping):
        hooks = {
            k: pmd[k]
            for k in (
                "plan_memory_eligible",
                "plan_template_key",
                "context_item_ids",
                "relevant_memory_ids",
            )
            if k in pmd
        }
        for k, v in hooks.items():
            if not is_json_serializable(v):
                out.append(
                    artifact_result(
                        validator="planner_plan_schema",
                        artifact_kind=kind,
                        status="failed",
                        severity="warning",
                        code="non_serializable_plan_metadata",
                        path=f"planner.plan.metadata.{k}",
                        message="planner memory hooks must be JSON-serializable.",
                    )
                )

    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        out.append(
            artifact_result(
                validator="planner_plan_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_steps",
                path="planner.plan.steps",
                message="plan.steps must be a list when plan is present.",
            )
        )
        if executable_intent:
            reco = DecisionAction.ASK
        return out, reco

    plan_status = str(plan.get("status") or "").strip().lower()
    actionable = False
    for idx, step in enumerate(raw_steps):
        if not isinstance(step, Mapping):
            out.append(
                artifact_result(
                    validator="planner_plan_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="error",
                    code="malformed_step",
                    path=f"planner.plan.steps[{idx}]",
                    message="each step must be dict-like.",
                )
            )
            actionable = True
            continue
        sid = str(step.get("id") or f"step_{idx}")

        raw_action = _step_action_type(step)
        if raw_action and raw_action != "noop":
            actionable = True

        risk = str(step.get("risk_level") or "").strip().lower()
        requires = bool(step.get("requires_approval"))
        if risk == RiskLevel.HIGH.value and not requires:
            out.append(
                artifact_result(
                    validator="planner_plan_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="error",
                    code="high_risk_requires_approval",
                    path=f"planner.plan.steps.{sid}.requires_approval",
                    message="high-risk steps must set requires_approval=true.",
                    escalation_recommended=True,
                )
            )
            actionable = True

        st = str(step.get("status") or "").strip().lower()
        if st == PlanStepStatus.BLOCKED.value and plan_status == PlanStatus.APPROVED.value:
            out.append(
                artifact_result(
                    validator="planner_plan_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="critical",
                    code="blocked_step_in_approved_plan",
                    path=f"planner.plan.steps.{sid}.status",
                    message="blocked steps cannot appear in an approved plan.",
                    escalation_recommended=True,
                )
            )
            actionable = True

    if actionable and executable_intent and reco is None:
        reco = DecisionAction.ASK

    out.append(
        artifact_result(
            validator="planner_plan_schema",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="planner.plan",
            message="Planner plan schema checks complete.",
        )
    )
    return out, reco


_EXTERNAL_TARGETS = frozenset(
    {
        "shell",
        "network",
        "git",
        "tool",
        "message",
        "file_write",
        "file_delete",
    }
)


def validate_executor_result_schema(
    exec_meta: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if exec_meta is None:
        return out, reco
    if not exec_meta:
        return out, reco
    kind = "executor_result"

    overall = str(exec_meta.get("overall_status") or "").strip().lower()
    try:
        ExecutionStatus(overall)
    except ValueError:
        out.append(
            artifact_result(
                validator="executor_result_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_execution_status",
                path="executor.execution_result.overall_status",
                message=f"Unknown overall_status {overall!r}.",
            )
        )
        reco = DecisionAction.RECORD

    if "dry_run" not in exec_meta:
        out.append(
            artifact_result(
                validator="executor_result_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="dry_run_not_explicit",
                path="executor.execution_result.dry_run",
                message="dry_run must be explicit on execution results.",
            )
        )

    halted = bool(exec_meta.get("halted"))
    records = exec_meta.get("records", [])
    if not isinstance(records, list):
        out.append(
            artifact_result(
                validator="executor_result_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_records",
                path="executor.execution_result.records",
                message="records must be a list.",
            )
        )
        reco = reco or DecisionAction.RECORD
        return out, reco

    risky_external = False
    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            continue
        status = str(rec.get("status") or "").strip().lower()
        try:
            ExecutionStatus(status)
        except ValueError:
            out.append(
                artifact_result(
                    validator="executor_result_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="invalid_record_status",
                    path=f"executor.execution_result.records[{idx}].status",
                    message="ExecutionRecord status invalid.",
                )
            )
        md = rec.get("metadata")
        if isinstance(md, Mapping):
            rc = md.get("reason_code") or md.get("reason")
            tgt = str(md.get("target_type") or "").strip().lower()
            if tgt in _EXTERNAL_TARGETS:
                risky_external = True
            if (
                status == ExecutionStatus.FAILED.value
                and rc in (None, "")
                and coerce_action_type(md.get("requested_action_type") or md.get("action_type"))
                is not None
            ):
                out.append(
                    artifact_result(
                        validator="executor_failure_reason",
                        artifact_kind=kind,
                        status="failed",
                        severity="error",
                        code="missing_failure_reason_code",
                        path=f"executor.execution_result.records[{idx}].metadata",
                        message="Failed execution records should include explicit reason codes.",
                        escalation_recommended=risky_external,
                    )
                )
                reco = DecisionAction.ASK if risky_external else (reco or DecisionAction.RECORD)

        if "dry_run" not in rec:
            out.append(
                artifact_result(
                    validator="executor_result_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="record_dry_run_missing",
                    path=f"executor.execution_result.records[{idx}].dry_run",
                    message="Each ExecutionRecord should expose dry_run explicitly.",
                )
            )

    if halted and overall == ExecutionStatus.SUCCESS.value and any(
        str(r.get("status") or "").strip().lower() == ExecutionStatus.FAILED.value for r in records
    ):
        out.append(
            artifact_result(
                validator="executor_partial_execution",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="preflight_inconsistent",
                path="executor.execution_result",
                message="No partial execution when preflight failed (halted with failed records).",
            )
        )
        reco = reco or DecisionAction.RECORD

    if not exec_meta.get("dry_run", True) and not any(
        str(r.get("status") or "").strip().lower() == ExecutionStatus.SUCCESS.value
        for r in records
        if isinstance(r, Mapping)
    ):
        out.append(
            artifact_result(
                validator="executor_live_scope",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="live_execution_suspicious",
                path="executor.execution_result",
                message="Live execution should not broaden permissions; verify records reflect internal-only success paths.",
            )
        )

    unsupported_refusal_ok = False
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        md = rec.get("metadata")
        if not isinstance(md, Mapping):
            continue
        if md.get("reason_code") == "unsupported_target_type" or md.get("reason") == "unsupported_target_type":
            tgt = str(md.get("target_type") or "").strip().lower()
            if tgt in _EXTERNAL_TARGETS:
                unsupported_refusal_ok = True

    if unsupported_refusal_ok:
        out.append(
            artifact_result(
                validator="executor_external_refusal",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="unsupported_target_refused",
                path="executor.execution_result",
                message="External targets rejected with explicit reason code (Executor v0).",
            )
        )

    out.append(
        artifact_result(
            validator="executor_result_schema",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="executor.execution_result",
            message="Executor result schema checks complete.",
        )
    )
    return out, reco


def _adjustment_provenance_ok(rec: Mapping[str, Any]) -> bool:
    if rec.get("source_signal_id"):
        return True
    md = rec.get("metadata")
    if isinstance(md, Mapping) and (
        md.get("source_event_id") or md.get("source_signal_id")
    ):
        return True
    return False


def validate_learning_v1_metadata(
    lr_meta: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if not lr_meta:
        return out, reco
    kind = "learning_v1"
    if lr_meta.get("learning_version") not in (None, "", "v1"):
        out.append(
            artifact_result(
                validator="learning_version",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="learning_version_mismatch",
                path="learning.metadata.learning_version",
                message="learning_version should be v1 when v1 lists are present.",
            )
        )

    def _need_list(name: str) -> None:
        if name not in lr_meta:
            out.append(
                artifact_result(
                    validator="learning_lists",
                    artifact_kind=kind,
                    status="warning",
                    severity="warning",
                    code="missing_list",
                    path=f"learning.metadata.{name}",
                    message=f"expected {name} list for v1 payloads when learning ran.",
                )
            )
        elif not isinstance(lr_meta.get(name), list):
            out.append(
                artifact_result(
                    validator="learning_lists",
                    artifact_kind=kind,
                    status="failed",
                    severity="error",
                    code="invalid_list_type",
                    path=f"learning.metadata.{name}",
                    message=f"{name} must be a list when present.",
                )
            )

    v1_evidence = lr_meta.get("learning_version") == "v1" or "adjustment_records" in lr_meta
    if v1_evidence:
        for key in (
            "consumed_learning_events",
            "adjustment_records",
            "proposed_adjustments",
            "applied_adjustments",
            "skipped_adjustments",
            "cross_facet_routes",
        ):
            _need_list(key)

    applied = lr_meta.get("applied_adjustments") or []
    if isinstance(applied, list):
        for idx, rec in enumerate(applied):
            if not isinstance(rec, Mapping):
                continue
            st = str(rec.get("status") or "").strip().lower()
            if st != AdjustmentStatus.APPLIED.value:
                continue
            if not _adjustment_provenance_ok(rec):
                out.append(
                    artifact_result(
                        validator="learning_adjustment_provenance",
                        artifact_kind=kind,
                        status="failed",
                        severity="critical",
                        code="missing_provenance",
                        path=f"learning.applied_adjustments[{idx}]",
                        message="applied adjustments must carry source_signal_id or source_event_id in metadata.",
                        escalation_recommended=True,
                    )
                )
                reco = DecisionAction.RECORD

            target_facet = str(rec.get("target_facet") or "").strip().casefold()
            field = str(rec.get("field") or "").strip().casefold()
            tgt = str(rec.get("target") or "").strip().casefold()
            if target_facet in {"policy", "executor"} or "permission" in field or "permission" in tgt:
                out.append(
                    artifact_result(
                        validator="learning_permission_mutation",
                        artifact_kind=kind,
                        status="failed",
                        severity="critical",
                        code="forbidden_permission_learning_apply",
                        path=f"learning.applied_adjustments[{idx}].target_facet",
                        message="Learning must not apply policy or executor permission mutations.",
                        escalation_recommended=True,
                    )
                )
                reco = DecisionAction.RECORD

            raw_target = rec.get("target")
            if raw_target is not None:
                try:
                    AdjustmentTarget(str(raw_target))
                except ValueError:
                    out.append(
                        artifact_result(
                            validator="learning_adjustment_target",
                            artifact_kind=kind,
                            status="warning",
                            severity="warning",
                            code="unknown_adjustment_target_enum",
                            path=f"learning.applied_adjustments[{idx}].target",
                            message="adjustment target should be a known AdjustmentTarget value.",
                        )
                    )

    if not out:
        out.append(
            artifact_result(
                validator="learning_v1_schema",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="ok",
                path="learning.metadata",
                message="Learning v1 metadata checked.",
            )
        )
    else:
        out.append(
            artifact_result(
                validator="learning_v1_schema",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="ok",
                path="learning.metadata",
                message="Learning v1 metadata checked with findings.",
            )
        )
    return out, reco


def validate_policy_facet_metadata(policy_meta: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not policy_meta or "policy_status" not in policy_meta:
        return out
    kind = "policy_metadata"
    ps = str(policy_meta.get("policy_status") or "").strip().lower()
    if ps not in KNOWN_POLICY_STATUS:
        out.append(
            artifact_result(
                validator="policy_metadata",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="unknown_policy_status",
                path="policy.policy_status",
                message=f"policy_status {ps!r} is not in known verifier allowlist.",
            )
        )
    for key in ("policy_reason", "reasons"):
        if key in policy_meta and not is_json_serializable(policy_meta.get(key)):
            out.append(
                artifact_result(
                    validator="policy_reason_serializable",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="policy_reason_not_serializable",
                    path=f"policy.{key}",
                    message="policy reasons must be JSON-serializable.",
                )
            )
    out.append(
        artifact_result(
            validator="policy_metadata",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="policy.metadata",
            message="Policy metadata schema checked.",
        )
    )
    return out


def validate_context_load_metadata(
    blob: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Malformed ratios warn; never raises."""
    out: list[dict[str, Any]] = []
    if blob is None:
        return out
    kind = "context_load"
    try:
        ic = blob.get("item_count")
        mx = blob.get("max_items")
        ratio = blob.get("load_ratio")
        overloaded = blob.get("overloaded")

        if ic is not None:
            v = int(ic)
            if v < 0:
                out.append(
                    artifact_result(
                        validator="context_load_schema",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="negative_item_count",
                        path="context_load.item_count",
                        message="item_count should be >= 0.",
                    )
                )
        if mx is not None:
            v = int(mx)
            if v < 0:
                out.append(
                    artifact_result(
                        validator="context_load_schema",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="negative_max_items",
                        path="context_load.max_items",
                        message="max_items should be >= 0.",
                    )
                )
        if ratio is not None:
            if isinstance(ratio, bool) or not _is_unit_number(ratio):
                out.append(
                    artifact_result(
                        validator="context_load_schema",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="load_ratio_malformed",
                        path="context_load.load_ratio",
                        message="load_ratio must be numeric in [0,1] when present.",
                    )
                )
            elif mx is not None and ic is not None:
                try:
                    mx_i = int(mx)
                    ic_i = int(ic)
                    if mx_i > 0:
                        expected = _clamp01(ic_i / mx_i)
                        if abs(expected - float(ratio)) > 0.05:
                            out.append(
                                artifact_result(
                                    validator="context_load_schema",
                                    artifact_kind=kind,
                                    status="warning",
                                    severity="warning",
                                    code="load_ratio_inconsistent",
                                    path="context_load.load_ratio",
                                    message="load_ratio should approximate item_count/max_items when max_items>0.",
                                )
                            )
                except (TypeError, ValueError):
                    pass

        if overloaded is not None and not isinstance(overloaded, bool):
            out.append(
                artifact_result(
                    validator="context_load_schema",
                    artifact_kind=kind,
                    status="warning",
                    severity="warning",
                    code="overloaded_not_bool",
                    path="context_load.overloaded",
                    message="overloaded must be boolean when present.",
                )
            )
    except Exception as exc:  # noqa: BLE001 deliberate broad catch
        out.append(
            artifact_result(
                validator="context_load_schema",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="context_load_validation_error",
                path="context_load",
                message=f"Malformed context load metadata handled as warning: {exc}",
            )
        )

    out.append(
        artifact_result(
            validator="context_load_schema",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="context_load",
            message="Context load metadata checked.",
        )
    )
    return out


ExpressionGateForbiddenPayloadKeys = frozenset(
    {
        "text",
        "message",
        "utterance",
        "speech",
        "prose",
        "natural_language",
        "nlp",
        "response_text",
        "final_text",
    }
)
_EXPRESSION_GATE_VALID_MODES = frozenset(
    {
        "silent",
        "log_only",
        "status_only",
        "short_utterance",
        "ask_user",
    },
)


def validate_expression_gate_v0(
    recommendation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Structural checks for Nexus Expression Gate recommendations (support infrastructure)."""
    out: list[dict[str, Any]] = []
    kind = "expression_gate_v0"
    if recommendation is None or not isinstance(recommendation, Mapping):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="missing_recommendation",
                path="nexus.expression_gate.last_recommendation",
                message="Expression recommendation mapping is required when present.",
            )
        )
        return out

    mode = str(recommendation.get("mode") or "").strip().lower()
    if mode not in _EXPRESSION_GATE_VALID_MODES:
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_mode",
                path="expression.mode",
                message=f"Expression mode '{mode}' is not a valid ExpressionMode.",
            )
        )

    raw_score = recommendation.get("expression_score")
    if isinstance(raw_score, bool) or not _is_unit_number(raw_score):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="expression_score_oob",
                path="expression.expression_score",
                message="expression_score must be numeric in [0,1].",
            )
        )

    max_words_raw = recommendation.get("max_words", 0)
    try:
        max_words_val = int(max_words_raw)
    except (TypeError, ValueError):
        max_words_val = -1
    if max_words_val < 0 or max_words_val > 4096:
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="max_words_oob",
                path="expression.max_words",
                message="max_words must be bounded and non-negative.",
            )
        )

    suppressed = recommendation.get("suppressed") is True
    allowed_uf = recommendation.get("allowed_user_facing") is True
    if suppressed and allowed_uf:
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="critical",
                code="suppressed_user_facing_conflict",
                path="expression.allowed_user_facing",
                message="User-facing expression must not be allowed when suppressed is true.",
                escalation_recommended=True,
            )
        )

    payload = recommendation.get("payload")
    if not isinstance(payload, Mapping):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="payload_not_mapping",
                path="expression.payload",
                message="payload should be an object.",
            )
        )
    elif isinstance(payload, Mapping):
        pl_keys = {str(k).strip().lower() for k in payload.keys()}
        blocked = ExpressionGateForbiddenPayloadKeys & pl_keys
        if blocked:
            out.append(
                artifact_result(
                    validator="expression_gate_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="error",
                    code="candidate_prose_keys",
                    path="expression.payload",
                    message=f"Expression payload forbids prose-like keys: {sorted(blocked)}",
                )
            )

    sint = str(recommendation.get("suggested_intent") or "").strip().lower()
    userish_modes = {"short_utterance", "ask_user"}
    if (
        mode in userish_modes
        and sint in {"", "none"}
        and not suppressed
        and recommendation.get("max_words", 0) > 0
    ):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="userish_mode_requires_intent",
                path="expression.suggested_intent",
                message="short_utterance / ask_user should declare a suggested_intent.",
                retry_recommended=True,
            )
        )

    if not is_json_serializable(recommendation):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="not_json_serializable",
                path="expression",
                message="Expression recommendation must be JSON-serializable.",
            )
        )

    if not any(
        isinstance(it, Mapping) and str(it.get("status") or "").lower() == "failed"
        for it in out
    ):
        out.append(
            artifact_result(
                validator="expression_gate_schema",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="ok",
                path="expression_gate",
                message="Expression Gate v0 recommendation shape checked.",
            )
        )
    return out


_PRESENTATION_MODES = frozenset(
    {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "blocked",
        "overloaded",
        "verifying",
        "learning",
        "warning",
        "sleeping",
        "unknown",
    }
)
_PRESENTATION_MOTIONS = frozenset(
    {
        "still",
        "blink",
        "slow_blink",
        "pulse",
        "bounce",
        "jitter",
        "mouth_loop",
        "ellipsis",
        "none",
    }
)
_PRESENTATION_CHANNELS = frozenset(
    {
        "none",
        "internal",
        "status",
        "user_expression",
        "ask_user",
        "warning",
    }
)


def validate_presentation_vector_v0(
    presentation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Structural checks for optional Presentation Vector v0 dicts."""
    kind = "presentation_vector_v0"
    out: list[dict[str, Any]] = []
    if presentation is None or not isinstance(presentation, Mapping):
        return out

    mode = str(presentation.get("mode") or "").strip().lower()
    if mode not in _PRESENTATION_MODES:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_mode",
                path="presentation.mode",
                message=f"Presentation mode '{mode}' is not a valid PresentationMode.",
            )
        )

    motion = str(presentation.get("motion") or "").strip().lower()
    if motion not in _PRESENTATION_MOTIONS:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_motion",
                path="presentation.motion",
                message=f"Presentation motion '{motion}' is not valid.",
            )
        )

    channel = str(presentation.get("channel") or "").strip().lower()
    if channel not in _PRESENTATION_CHANNELS:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="invalid_channel",
                path="presentation.channel",
                message=f"Presentation channel '{channel}' is not valid.",
            )
        )

    for key in (
        "intensity",
        "pressure",
        "latent_pressure",
        "confidence",
        "novelty",
        "attention_motion",
    ):
        raw = presentation.get(key)
        if isinstance(raw, bool) or not _is_unit_number(raw):
            out.append(
                artifact_result(
                    validator="presentation_vector_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code=f"{key}_oob",
                    path=f"presentation.{key}",
                    message=f"{key} must be numeric in [0,1].",
                )
            )

    for bk in (
        "user_attention_needed",
        "expression_active",
        "blocked",
        "overloaded",
        "warning",
        "speaking",
        "thinking",
        "idle",
    ):
        if bk in presentation and not isinstance(presentation.get(bk), bool):
            out.append(
                artifact_result(
                    validator="presentation_vector_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code=f"{bk}_not_bool",
                    path=f"presentation.{bk}",
                    message=f"{bk} must be a boolean.",
                )
            )

    expr_active = presentation.get("expression_active") is True
    expr_mode = str(presentation.get("expression_mode") or "").strip().lower()
    if expr_active and expr_mode not in {"short_utterance", "ask_user"}:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="expression_active_mode_mismatch",
                path="presentation.expression_active",
                message="expression_active requires expression_mode short_utterance or ask_user.",
            )
        )

    meta = presentation.get("metadata")
    if isinstance(meta, Mapping) and meta.get("expression_suppressed") is True and expr_active:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="expression_active_while_suppressed",
                path="presentation.metadata.expression_suppressed",
                message="expression_active cannot be true when expression is suppressed.",
                escalation_recommended=True,
            )
        )

    try:
        dumped = json.dumps(presentation)
    except (TypeError, ValueError):
        dumped = ""
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="not_json_serializable",
                path="presentation",
                message="Presentation vector must be JSON-serializable.",
            )
        )
    if dumped and len(dumped) > 32000:
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="presentation_too_large",
                path="presentation",
                message="Presentation vector JSON exceeds compact size budget.",
            )
        )

    if not any(
        isinstance(it, Mapping) and str(it.get("status") or "").lower() == "failed"
        for it in out
    ):
        out.append(
            artifact_result(
                validator="presentation_vector_schema",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="ok",
                path="presentation_vector_v0",
                message="Presentation Vector v0 checked.",
            )
        )
    return out


def validate_nexus_interrupt_v2_audit(
    prior_cycle_trace: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Inspect prior persisted cycle_trace for Nexus v2 interrupt/suppression semantics."""
    out: list[dict[str, Any]] = []
    kind = "nexus_interrupt_v2"
    if not isinstance(prior_cycle_trace, Mapping):
        return out
    suppressions = prior_cycle_trace.get("suppression_decisions")
    if suppressions is None:
        return out

    if not isinstance(suppressions, Sequence) or isinstance(suppressions, (str, bytes)):
        out.append(
            artifact_result(
                validator="nexus_interrupt_v2",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="malformed_suppression_decisions",
                path="nexus.last_cycle_trace.suppression_decisions",
                message="suppression_decisions must be a list when present.",
            )
        )
        return out

    for idx, row in enumerate(suppressions):
        if not isinstance(row, Mapping):
            out.append(
                artifact_result(
                    validator="nexus_interrupt_v2",
                    artifact_kind=kind,
                    status="failed",
                    severity="error",
                    code="malformed_suppression_row",
                    path=f"nexus.last_cycle_trace.suppression_decisions[{idx}]",
                    message="Suppression decision row must be a mapping.",
                )
            )
            continue
        if row.get("allowed_user_expression") is True:
            out.append(
                artifact_result(
                    validator="nexus_interrupt_v2",
                    artifact_kind=kind,
                    status="failed",
                    severity="critical",
                    code="illegal_user_expression_flag",
                    path=f"nexus.last_cycle_trace.suppression_decisions[{idx}].allowed_user_expression",
                    message="allowed_user_expression must remain false until an Expression Gate exists.",
                    escalation_recommended=True,
                )
            )

    allowed_any_internal = any(
        isinstance(row, Mapping) and row.get("allowed_internal_event") is True
        for row in suppressions
    )

    suppressed_ids = {
        str(d.get("candidate_id"))
        for d in suppressions
        if isinstance(d, Mapping) and d.get("suppressed") is True and d.get("candidate_id")
    }

    queued_id_blob = prior_cycle_trace.get("interrupt_processed")
    iq = queued_id_blob.get("candidate_id") if isinstance(queued_id_blob, dict) else None
    cand_raw = prior_cycle_trace.get("allowed_interrupt_candidate")

    cq = cand_raw.get("id") if isinstance(cand_raw, dict) else None
    if iq and str(iq) in suppressed_ids:
        out.append(
            artifact_result(
                validator="nexus_interrupt_v2",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="processed_suppressed_interrupt",
                path="nexus.last_cycle_trace.interrupt_processed.candidate_id",
                message="Interrupt candidate marked processed but its row was suppressed.",
                retry_recommended=True,
            )
        )

    if allowed_any_internal and cq and str(cq) in suppressed_ids:
        out.append(
            artifact_result(
                validator="nexus_interrupt_v2",
                artifact_kind=kind,
                status="failed",
                severity="error",
                code="allowed_interrupt_was_suppressed",
                path="nexus.last_cycle_trace.allowed_interrupt_candidate.id",
                message="Allowed interrupt candidate id appears in suppressed outcomes.",
                retry_recommended=True,
            )
        )

    cand = cand_raw if isinstance(cand_raw, dict) else None
    proc = queued_id_blob if isinstance(queued_id_blob, dict) else {}
    if cand is not None and allowed_any_internal:
        meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
        crit = cand.get("interrupt_type") == "verifier_escalation" and (
            cand.get("suppressible") is False or meta.get("artifact_critical_or_error") is True
        )
        if not crit:
            pval = cand.get("priority")
            try:
                pfloat = float(pval)
            except (TypeError, ValueError):
                pfloat = 0.0
            if proc.get("queued") is True and pfloat + 1e-9 < 0.55:
                out.append(
                    artifact_result(
                        validator="nexus_interrupt_v2",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="allowed_interrupt_priority_below_threshold",
                        path="nexus.last_cycle_trace.allowed_interrupt_candidate.priority",
                        message="Queued internal interrupt priority below nominal safety threshold.",
                    )
                )

    if not any(
        isinstance(it, Mapping) and str(it.get("status") or "").lower() == "failed"
        for it in out
    ):
        out.append(
            artifact_result(
                validator="nexus_interrupt_v2",
                artifact_kind=kind,
                status="passed",
                severity="info",
                code="ok",
                path="nexus.last_cycle_trace",
                message="Nexus v2 interrupt audit checked prior cycle.",
            )
        )
    return out


def run_all_artifact_validators(
    *,
    facet_results: Sequence[Any],
    decision_action: DecisionAction | None,
    nexus_ctx: Mapping[str, Any] | None,
    prior_cycle_trace: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    """Collect structured artifact checks plus optional risk downgrade."""
    rows: list[dict[str, Any]] = []
    reco: DecisionAction | None = None

    def _merge_r(new: DecisionAction | None) -> None:
        nonlocal reco
        if new is None:
            return
        if reco is None:
            reco = new
            return
        if reco == DecisionAction.RECORD:
            return
        if new == DecisionAction.RECORD:
            reco = DecisionAction.RECORD
            return
        reco = new

    behavior_trace = None
    policy_meta = None
    context_load = None
    learning_meta = None
    plan = None
    executor_er = None

    for fr in facet_results:
        name = getattr(fr, "facet_name", "")
        md = getattr(fr, "metadata", None)
        if not isinstance(md, Mapping):
            continue
        if name == "behavior":
            behavior_trace = md.get("decision_trace")
            context_load = md.get("context_load") or context_load
        if name == "policy":
            policy_meta = md
        if name == "learning":
            lr = md.get("learning_result")
            if isinstance(lr, Mapping):
                learning_meta = lr.get("metadata")
        if name == "planner" and isinstance(md.get("plan"), Mapping):
            plan = md["plan"]
        if name == "executor" and isinstance(md.get("execution_result"), Mapping):
            executor_er = md["execution_result"]

    decision_is_act = decision_action == DecisionAction.ACT

    if isinstance(behavior_trace, Mapping):
        chunk, r = validate_behavior_decision_trace_v2(
            behavior_trace, decision_is_act=decision_is_act
        )
        rows.extend(chunk)
        _merge_r(r)

    if isinstance(policy_meta, Mapping):
        rows.extend(validate_policy_facet_metadata(policy_meta))

    if isinstance(context_load, Mapping):
        rows.extend(validate_context_load_metadata(context_load))

    if isinstance(learning_meta, Mapping):
        chunk, r = validate_learning_v1_metadata(learning_meta)
        rows.extend(chunk)
        _merge_r(r)

    exec_meta = None
    for fr in facet_results:
        if getattr(fr, "facet_name", "") == "executor":
            md = getattr(fr, "metadata", None)
            if isinstance(md, Mapping):
                exec_meta = md
                break
    execution_requested = bool(exec_meta and exec_meta.get("execution_requested"))
    exec_intent = bool(
        execution_requested
        and isinstance(executor_er, Mapping)
        and executor_er.get("records") is not None
    )
    if isinstance(plan, Mapping):
        chunk, r = validate_planner_plan_schema(
            plan,
            executable_intent=bool(exec_intent or execution_requested),
        )
        rows.extend(chunk)
        _merge_r(r)

    if isinstance(executor_er, Mapping) and executor_er != {}:
        chunk, r = validate_executor_result_schema(executor_er)
        rows.extend(chunk)
        _merge_r(r)

    sig = None
    trace = None
    if isinstance(nexus_ctx, Mapping):
        sig = nexus_ctx.get("signal_map")
        trace = {
            "facet_order": nexus_ctx.get("facet_order"),
            "signal_map": nexus_ctx.get("signal_map"),
            "pressure_components": nexus_ctx.get("pressure_components"),
            "final_decision": nexus_ctx.get("final_decision"),
            "learning_events": nexus_ctx.get("learning_events"),
            "internal_events_queued": nexus_ctx.get("internal_events_queued"),
            "internal_events_processed": nexus_ctx.get("internal_events_processed"),
        }

    if isinstance(sig, Mapping):
        rows.extend(validate_cycle_signal_map_pressure(sig))

    if isinstance(nexus_ctx, Mapping) and trace is not None:
        chunk, r = validate_cycle_trace_structure(trace)
        rows.extend(chunk)
        _merge_r(r)

    if isinstance(prior_cycle_trace, Mapping):
        rows.extend(validate_nexus_interrupt_v2_audit(prior_cycle_trace))

    return rows, reco
