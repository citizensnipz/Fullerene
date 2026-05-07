"""Deterministic artifact schema validators for Verifier v1 (no LLM, no network)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from fullerene.executor.models import ExecutionStatus, coerce_action_type
from fullerene.learning.models import AdjustmentStatus, AdjustmentTarget
from fullerene.nexus.models import DecisionAction
from fullerene.planner.models import PlanStatus, PlanStepStatus, RiskLevel
from fullerene.context.models import (
    DYNAMIC_ACTIVE_FACETS_V1,
    PRESSURE_RELEVANCE_V2,
    SELF_EDITING_V3,
    STATIC_RECENT_EPISODIC_V0,
    ContextItemType,
)
from fullerene.world_model.models import BeliefStatus

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
BEHAVIOR_DECISION_KEYS = ("wait", "record", "ask", "act")
ACT_OVERLOAD_THRESHOLD = 0.85
VALID_CONTEXT_STRATEGIES = frozenset(
    {
        STATIC_RECENT_EPISODIC_V0,
        DYNAMIC_ACTIVE_FACETS_V1,
        PRESSURE_RELEVANCE_V2,
        SELF_EDITING_V3,
    }
)
VALID_CONTEXT_TYPES = frozenset(item.value for item in ContextItemType)
VALID_BELIEF_STATUS = frozenset(
    {BeliefStatus.VALID.value, BeliefStatus.CONTRADICTED.value, BeliefStatus.REDUNDANT.value}
)
UNSUPPORTED_CAPABILITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("external_access_claim", ("searched", "looked up", "browsed", "accessed the web")),
    ("file_or_database_access_claim", ("read file", "checked database", "queried api")),
    ("tool_use_claim", ("called tool", "used clock", "accessed real-time data")),
    ("memory_source_claim", ("remembered", "retrieved from memory", "based on records")),
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


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)


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
        # Keep empty content_summary/content valid (SYSTEM_TICK often carries "").
        summary = event_obj.get("content_summary")
        if summary is None and "content_summary" not in event_obj:
            summary = event_obj.get("content")
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
        for dec in BEHAVIOR_DECISION_KEYS:
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

    for metric in (
        "conversational_intent_score",
        "grounding_confidence",
        "ambiguity_score",
        "continuity_confidence",
        "self_consistency_confidence",
        "world_model_belief_confidence",
        "context_load_ratio",
    ):
        if metric in trace and trace.get(metric) is not None and not _is_unit_number(trace.get(metric)):
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="warning",
                    code="invalid_numeric",
                    path=f"behavior.decision_trace.{metric}",
                    message=f"{metric} must be numeric in [0,1] when present.",
                    retry_recommended=True,
                )
            )

    intent_scores = trace.get("intent_scores")
    if intent_scores is not None:
        if not isinstance(intent_scores, Mapping):
            out.append(
                artifact_result(
                    validator="behavior_decision_trace_schema",
                    artifact_kind=artifact_kind,
                    status="failed",
                    severity="warning",
                    code="invalid_type",
                    path="behavior.decision_trace.intent_scores",
                    message="intent_scores must be mapping when present.",
                    retry_recommended=True,
                )
            )
        else:
            for k, v in intent_scores.items():
                if not _is_unit_number(v):
                    out.append(
                        artifact_result(
                            validator="behavior_decision_trace_schema",
                            artifact_kind=artifact_kind,
                            status="failed",
                            severity="warning",
                            code="invalid_numeric",
                            path=f"behavior.decision_trace.intent_scores.{k}",
                            message="intent_scores values must be in [0,1].",
                            retry_recommended=True,
                        )
                    )

    is_act = decision_is_act
    trace_is_act = fd == "act"
    policy_result = str(trace.get("policy_result") or "").strip().lower()
    contradiction = trace.get("contradiction_flag") is True
    conf = _to_float(trace.get("confidence")) or 0.0
    grounding_need = str(trace.get("grounding_need") or "unknown").strip().lower()
    grounding_available = bool(trace.get("grounding_available"))
    grounding_conf = _to_float(trace.get("grounding_confidence")) or 0.0
    ambiguity = _to_float(trace.get("ambiguity_score")) or 0.0
    ctx_ratio = _to_float(trace.get("context_load_ratio")) or 0.0

    if is_act and policy_result in {"denied", "approval_required"}:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="failed",
                severity="critical",
                code="act_policy_mismatch",
                path="behavior.decision_trace.policy_result",
                message="ACT cannot be final_decision when policy_result is denied/approval_required.",
                retry_recommended=False,
                escalation_recommended=True,
            )
        )
        reco = DecisionAction.ASK if policy_result == "approval_required" else DecisionAction.RECORD
    elif trace_is_act and policy_result in {"denied", "approval_required"}:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="warning",
                severity="warning",
                code="behavior_act_policy_mismatch_downgraded",
                path="behavior.decision_trace.policy_result",
                message="Behavior trace selected ACT while policy denied/required approval, but final decision was downgraded.",
            )
        )

    if is_act and contradiction:
        sev = "error" if conf >= 0.7 else "warning"
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="failed",
                severity=sev,
                code="act_with_contradiction",
                path="behavior.decision_trace.contradiction_flag",
                message="ACT selected while contradiction_flag is true.",
                retry_recommended=True,
                escalation_recommended=sev == "error",
            )
        )
        if sev == "error":
            reco = reco or DecisionAction.ASK

    if is_act and grounding_need not in {"none", "unknown"} and not grounding_available:
        sev = "error" if conf >= 0.7 else "warning"
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="failed",
                severity=sev,
                code="act_without_grounding",
                path="behavior.decision_trace.grounding_available",
                message="ACT selected while grounding is required but unavailable.",
                retry_recommended=True,
                escalation_recommended=False,
            )
        )
        if sev == "error":
            reco = reco or DecisionAction.ASK

    if conf >= 0.75 and grounding_conf <= 0.35:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="warning",
                severity="warning",
                code="high_confidence_low_grounding_confidence",
                path="behavior.decision_trace.grounding_confidence",
                message="High confidence with low grounding_confidence should be reviewed.",
                retry_recommended=True,
            )
        )

    if (is_act or trace_is_act) and ambiguity >= 0.7 and grounding_conf < 0.8:
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="warning",
                severity="warning",
                code="act_high_ambiguity",
                path="behavior.decision_trace.ambiguity_score",
                message="ACT under high ambiguity without strong grounding confidence.",
                retry_recommended=True,
            )
        )

    if is_act and ctx_ratio > ACT_OVERLOAD_THRESHOLD and grounding_conf < 0.5:
        sev = "error" if conf >= 0.7 else "warning"
        out.append(
            artifact_result(
                validator="behavior_decision_trace_consistency",
                artifact_kind=artifact_kind,
                status="failed",
                severity=sev,
                code="act_context_overload_low_grounding",
                path="behavior.decision_trace.context_load_ratio",
                message="ACT selected during context overload with weak grounding confidence.",
                retry_recommended=True,
            )
        )
        if sev == "error":
            reco = reco or DecisionAction.ASK

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


def validate_context_v2_packet(
    context_meta: Mapping[str, Any] | None,
    *,
    behavior_trace: Mapping[str, Any] | None,
    event: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not context_meta:
        return out
    kind = "context_v2_packet"
    strategy = str(context_meta.get("strategy") or "").strip().lower()
    if strategy and strategy not in VALID_CONTEXT_STRATEGIES:
        out.append(
            artifact_result(
                validator="context_v2_schema",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="invalid_context_strategy",
                path="context.metadata.strategy",
                message="Context strategy is not recognized.",
                retry_recommended=True,
            )
        )
    for key in ("context_budget", "budget_used", "item_count"):
        if key in context_meta and context_meta.get(key) is not None and _to_float(context_meta.get(key)) is None:
            out.append(
                artifact_result(
                    validator="context_v2_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="invalid_numeric",
                    path=f"context.metadata.{key}",
                    message=f"{key} should be numeric.",
                    retry_recommended=True,
                )
            )
    ict = context_meta.get("included_context_types")
    if ict is not None:
        if not isinstance(ict, list):
            out.append(
                artifact_result(
                    validator="context_v2_schema",
                    artifact_kind=kind,
                    status="failed",
                    severity="warning",
                    code="invalid_type",
                    path="context.metadata.included_context_types",
                    message="included_context_types should be a list.",
                )
            )
        else:
            bad = [x for x in ict if str(x) not in VALID_CONTEXT_TYPES]
            if bad:
                out.append(
                    artifact_result(
                        validator="context_v2_schema",
                        artifact_kind=kind,
                        status="failed",
                        severity="warning",
                        code="invalid_context_type",
                        path="context.metadata.included_context_types",
                        message=f"Unknown context types: {bad}.",
                    )
                )
    excluded = context_meta.get("excluded_context_items")
    if isinstance(excluded, list):
        for idx, row in enumerate(excluded[:64]):
            if isinstance(row, Mapping) and not str(row.get("reason") or "").strip():
                out.append(
                    artifact_result(
                        validator="context_v2_schema",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="excluded_item_missing_reason",
                        path=f"context.metadata.excluded_context_items[{idx}]",
                        message="Excluded context items should include reason.",
                    )
                )
    if strategy == PRESSURE_RELEVANCE_V2:
        for key in ("current_event_id", "context_budget", "excluded_context_items"):
            if key not in context_meta:
                out.append(
                    artifact_result(
                        validator="context_v2_consistency",
                        artifact_kind=kind,
                        status="warning",
                        severity="warning",
                        code="missing_pressure_relevance_metadata",
                        path=f"context.metadata.{key}",
                        message=f"{key} should be present for pressure_relevance_v2.",
                        retry_recommended=True,
                    )
                )
    if context_meta.get("working_memory_turn_count", 0) and not context_meta.get("included_working_memory_turns"):
        event_type = str((event or {}).get("event_type") or "").lower()
        if event_type == "user_message":
            out.append(
                artifact_result(
                    validator="context_v2_consistency",
                    artifact_kind=kind,
                    status="warning",
                    severity="warning",
                    code="missing_working_memory_inclusion",
                    path="context.metadata.included_working_memory_turns",
                    message="Recent working memory exists but no working-memory item was included.",
                    retry_recommended=True,
                )
            )
    if context_meta.get("active_unresolved_signal_count", 0) and not context_meta.get("included_lpb_entry_ids"):
        out.append(
            artifact_result(
                validator="context_v2_consistency",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="missing_lpb_context",
                path="context.metadata.included_lpb_entry_ids",
                message="Active unresolved signals exist but no LPB context item is included.",
                retry_recommended=True,
            )
        )
    cload = context_meta.get("context_load")
    is_overloaded = isinstance(cload, Mapping) and bool(cload.get("overloaded"))
    if is_overloaded and isinstance(behavior_trace, Mapping) and _coerce_decision_str(behavior_trace.get("final_decision")) == "act":
        out.append(
            artifact_result(
                validator="context_v2_consistency",
                artifact_kind=kind,
                status="failed",
                severity="warning",
                code="overloaded_context_act_retry",
                path="context.metadata.context_load.overloaded",
                message="Overloaded Context v2 with ACT should recommend compaction/retry.",
                retry_recommended=True,
            )
        )
    # Context v3 checks (compatible extension over v2 packet).
    if strategy == SELF_EDITING_V3:
        consolidated = context_meta.get("consolidated_context_items")
        if isinstance(consolidated, list):
            for idx, row in enumerate(consolidated[:32]):
                if not isinstance(row, Mapping):
                    continue
                if row.get("canonical") is not False:
                    out.append(
                        artifact_result(
                            validator="context_v3_schema",
                            artifact_kind=kind,
                            status="failed",
                            severity="warning",
                            code="consolidation_canonical_true",
                            path=f"context.metadata.consolidated_context_items[{idx}].canonical",
                            message="Consolidation must be non-canonical (canonical=false).",
                        )
                    )
                src_ids = row.get("source_item_ids")
                if not isinstance(src_ids, list) or not src_ids:
                    out.append(
                        artifact_result(
                            validator="context_v3_schema",
                            artifact_kind=kind,
                            status="warning",
                            severity="warning",
                            code="consolidation_missing_source_ids",
                            path=f"context.metadata.consolidated_context_items[{idx}].source_item_ids",
                            message="Consolidation should cite source item ids.",
                        )
                    )
        predictive = context_meta.get("predictive_context_items")
        if isinstance(predictive, list):
            for idx, row in enumerate(predictive[:64]):
                if not isinstance(row, Mapping):
                    continue
                meta = row.get("metadata")
                if not isinstance(meta, Mapping) or not bool(meta.get("predictive")):
                    out.append(
                        artifact_result(
                            validator="context_v3_schema",
                            artifact_kind=kind,
                            status="warning",
                            severity="warning",
                            code="predictive_item_unmarked",
                            path=f"context.metadata.predictive_context_items[{idx}]",
                            message="Predictive context item should be marked predictive=true.",
                        )
                    )
                score = (meta or {}).get("predictive_score") if isinstance(meta, Mapping) else None
                reason = (meta or {}).get("predictive_reason") if isinstance(meta, Mapping) else None
                if _to_float(score) is None or not str(reason or "").strip():
                    out.append(
                        artifact_result(
                            validator="context_v3_schema",
                            artifact_kind=kind,
                            status="warning",
                            severity="warning",
                            code="predictive_missing_reason_or_score",
                            path=f"context.metadata.predictive_context_items[{idx}].metadata",
                            message="Predictive item should include score and reason.",
                        )
                    )
        pressure = _to_float(context_meta.get("context_pressure"))
        components = context_meta.get("context_pressure_components")
        overloaded_v3 = bool(context_meta.get("context_overloaded"))
        if overloaded_v3 and pressure is None:
            out.append(
                artifact_result(
                    validator="context_v3_schema",
                    artifact_kind=kind,
                    status="warning",
                    severity="warning",
                    code="overloaded_without_context_pressure",
                    path="context.metadata.context_pressure",
                    message="Context overloaded but no context_pressure was emitted.",
                )
            )
        if isinstance(components, Mapping):
            for key, value in components.items():
                num = _to_float(value)
                if num is not None and key.endswith("_count"):
                    continue
                if num is not None and not (0.0 <= float(num) <= 1.0):
                    out.append(
                        artifact_result(
                            validator="context_v3_schema",
                            artifact_kind=kind,
                            status="warning",
                            severity="warning",
                            code="context_pressure_component_out_of_bounds",
                            path=f"context.metadata.context_pressure_components.{key}",
                            message="Context pressure component should be clamped to [0,1].",
                        )
                    )
        unresolved = context_meta.get("unresolved_references")
        reason = str(context_meta.get("context_pressure_reason") or "")
        if isinstance(unresolved, list) and len(unresolved) >= 2 and "unresolved" not in reason:
            out.append(
                artifact_result(
                    validator="context_v3_schema",
                    artifact_kind=kind,
                    status="warning",
                    severity="warning",
                    code="high_unresolved_without_clarification_pressure",
                    path="context.metadata.context_pressure_reason",
                    message="High unresolved references should influence pressure reason metadata.",
                )
            )
    out.append(
        artifact_result(
            validator="context_v2_schema",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="context.metadata",
            message="Context v2 packet checks complete.",
        )
    )
    return out


def validate_world_model_v1_artifacts(
    world_meta: Mapping[str, Any] | None,
    *,
    decision_is_act: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not world_meta:
        return out
    kind = "world_model_v1"
    relevant = world_meta.get("relevant_beliefs")
    if isinstance(relevant, list):
        by_norm: dict[str, int] = {}
        for idx, belief in enumerate(relevant):
            if not isinstance(belief, Mapping):
                continue
            if not belief.get("id"):
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="missing_belief_id", path=f"world_model.relevant_beliefs[{idx}].id", message="belief_id should be present."))
            norm = str(belief.get("normalized_key") or "").strip()
            if not norm and not str(belief.get("claim") or "").strip():
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="missing_content_or_normalized_key", path=f"world_model.relevant_beliefs[{idx}]", message="belief should include claim/content or normalized_key."))
            if not _is_unit_number(belief.get("confidence")):
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="invalid_confidence", path=f"world_model.relevant_beliefs[{idx}].confidence", message="belief confidence must be in [0,1]."))
            status = str(belief.get("status") or "").strip().lower()
            if status and status not in VALID_BELIEF_STATUS:
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="invalid_belief_status", path=f"world_model.relevant_beliefs[{idx}].status", message="belief status must be valid|contradicted|redundant."))
            sc = belief.get("support_count", 0)
            cc = belief.get("contradiction_count", 0)
            if _to_float(sc) is not None and float(sc) < 0:
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="negative_support_count", path=f"world_model.relevant_beliefs[{idx}].support_count", message="support_count must be >=0."))
            if _to_float(cc) is not None and float(cc) < 0:
                out.append(artifact_result(validator="world_model_belief_schema", artifact_kind=kind, status="failed", severity="warning", code="negative_contradiction_count", path=f"world_model.relevant_beliefs[{idx}].contradiction_count", message="contradiction_count must be >=0."))
            if status == "contradicted" and float(cc or 0) <= 0:
                out.append(artifact_result(validator="world_model_belief_consistency", artifact_kind=kind, status="failed", severity="warning", code="contradicted_without_contradiction_count", path=f"world_model.relevant_beliefs[{idx}].contradiction_count", message="Contradicted belief should have contradiction_count > 0."))
            if float(cc or 0) > 0 and float(belief.get("confidence") or 0.0) >= 0.9 and float(sc or 0) < float(cc):
                out.append(artifact_result(validator="world_model_belief_consistency", artifact_kind=kind, status="warning", severity="warning", code="contradiction_high_confidence_mismatch", path=f"world_model.relevant_beliefs[{idx}]", message="Belief has contradictions but remains very high confidence."))
            if status == "valid" and float(belief.get("confidence") or 0.0) < 0.2:
                out.append(artifact_result(validator="world_model_belief_consistency", artifact_kind=kind, status="warning", severity="warning", code="low_confidence_valid_status", path=f"world_model.relevant_beliefs[{idx}].status", message="Very low-confidence belief marked valid."))
            if status == "redundant" and norm:
                by_norm[norm] = by_norm.get(norm, 0) + 1
        for k, count in by_norm.items():
            if count > 1:
                out.append(artifact_result(validator="world_model_belief_consistency", artifact_kind=kind, status="warning", severity="warning", code="duplicate_redundant_normalized_key", path="world_model.relevant_beliefs", message=f"Redundant beliefs duplicate normalized_key {k!r}."))
    updates = world_meta.get("world_model_updates")
    if isinstance(updates, Mapping):
        edges = updates.get("edges")
        if isinstance(edges, list):
            for i, edge in enumerate(edges):
                if not isinstance(edge, Mapping):
                    continue
                src = str(edge.get("source_belief_id") or edge.get("source") or "")
                tgt = str(edge.get("target_belief_id") or edge.get("target") or "")
                et = str(edge.get("edge_type") or edge.get("type") or "")
                w = edge.get("weight")
                if not src or not tgt or not et or _to_float(w) is None:
                    out.append(artifact_result(validator="world_model_edge_schema", artifact_kind=kind, status="failed", severity="warning", code="invalid_edge_fields", path=f"world_model.world_model_updates.edges[{i}]", message="Belief edge requires source/target/type/weight."))
                if src and tgt and src == tgt and not bool(edge.get("allow_self_link")):
                    out.append(artifact_result(validator="world_model_edge_schema", artifact_kind=kind, status="failed", severity="warning", code="belief_edge_self_link", path=f"world_model.world_model_updates.edges[{i}]", message="Belief edge should not self-link unless explicitly allowed."))
    signals = world_meta.get("contradiction_signals")
    if isinstance(signals, list):
        has_contra = any(isinstance(s, Mapping) and s.get("entry_type") == "contradiction" for s in signals)
        if relevant and not has_contra:
            contradicted = any(isinstance(b, Mapping) and str(b.get("status") or "").lower() == "contradicted" for b in (relevant or []))
            if contradicted:
                out.append(artifact_result(validator="world_model_lpb_consistency", artifact_kind=kind, status="warning", severity="warning", code="missing_contradiction_pressure_signal", path="world_model.contradiction_signals", message="Contradicted belief present without contradiction pressure signal."))
    clusters = world_meta.get("active_contradiction_clusters")
    if isinstance(clusters, list):
        for idx, row in enumerate(clusters):
            if not isinstance(row, Mapping):
                continue
            ps = row.get("pressure_score")
            if ps is not None and not _is_unit_number(ps):
                out.append(
                    artifact_result(
                        validator="world_model_v2_cluster_schema",
                        artifact_kind=kind,
                        status="failed",
                        severity="warning",
                        code="invalid_cluster_pressure_score",
                        path=f"world_model.active_contradiction_clusters[{idx}].pressure_score",
                        message="Contradiction cluster pressure_score should be in [0,1] when present.",
                    )
                )
    for scalar_key in (
        "belief_graph_confidence",
        "top_contradiction_score",
        "top_belief_cluster_pressure",
    ):
        if scalar_key in world_meta and world_meta.get(scalar_key) is not None:
            if not _is_unit_number(world_meta.get(scalar_key)):
                out.append(
                    artifact_result(
                        validator="world_model_v2_schema",
                        artifact_kind=kind,
                        status="failed",
                        severity="warning",
                        code="invalid_world_model_scalar",
                        path=f"world_model.{scalar_key}",
                        message=f"{scalar_key} should be a unit number in [0,1] when present.",
                    )
                )
    if decision_is_act and bool(world_meta.get("requires_approval_due_to_contradiction")):
        out.append(
            artifact_result(
                validator="world_model_v2_behavior_consistency",
                artifact_kind=kind,
                status="warning",
                severity="warning",
                code="act_with_contradiction_approval_flag",
                path="world_model.requires_approval_due_to_contradiction",
                message="World model recommends approval for contradiction pressure while decision is ACT.",
            )
        )
    out.append(
        artifact_result(
            validator="world_model_v1_schema",
            artifact_kind=kind,
            status="passed",
            severity="info",
            code="ok",
            path="world_model.metadata",
            message="World Model v1 checks complete.",
        )
    )
    return out


def validate_output_metadata_and_capability_claims(
    output_meta: Mapping[str, Any] | None,
    *,
    available_traces: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not output_meta:
        return out
    kind = "output_metadata"
    for key in ("output_type",):
        if key in output_meta and not isinstance(output_meta.get(key), str):
            out.append(artifact_result(validator="output_metadata_schema", artifact_kind=kind, status="failed", severity="warning", code="invalid_type", path=f"output.{key}", message=f"{key} should be string when present.", retry_recommended=True))
    if "response_needed" in output_meta and not isinstance(output_meta.get("response_needed"), bool):
        out.append(artifact_result(validator="output_metadata_schema", artifact_kind=kind, status="failed", severity="warning", code="invalid_type", path="output.response_needed", message="response_needed should be boolean.", retry_recommended=True))
    text = str(output_meta.get("output_text") or output_meta.get("response_text") or output_meta.get("final_text") or "").lower()
    if text:
        traces = available_traces or {}
        has_exec = bool(traces.get("executor"))
        has_source_ids = bool(output_meta.get("source_ids") or output_meta.get("context_ids"))
        has_grounding = bool(output_meta.get("grounding_metadata") or output_meta.get("grounding_confidence") is not None)
        for code, patterns in UNSUPPORTED_CAPABILITY_PATTERNS:
            if any(p in text for p in patterns):
                missing = False
                row_code = "unsupported_capability_claim"
                if code == "tool_use_claim" and not has_exec:
                    missing = True
                    row_code = "missing_tool_or_executor_trace"
                elif code in {"file_or_database_access_claim", "external_access_claim"} and not has_exec:
                    missing = True
                elif code == "memory_source_claim" and not has_source_ids:
                    missing = True
                    row_code = "missing_source_ids"
                if "ground" in text and not has_grounding:
                    missing = True
                    row_code = "missing_grounding_metadata"
                out.append(
                    artifact_result(
                        validator="output_capability_claims",
                        artifact_kind=kind,
                        status="failed" if missing else "info",
                        severity="error" if missing else "info",
                        code=row_code if missing else "supported_capability_claim",
                        path="output.output_text",
                        message="Capability/source claim requires matching runtime metadata." if missing else "Capability/source claim has matching runtime metadata.",
                        retry_recommended=missing,
                        escalation_recommended=False,
                    )
                )
    out.append(artifact_result(validator="output_metadata_schema", artifact_kind=kind, status="passed", severity="info", code="ok", path="output", message="Output metadata checks complete."))
    return out


SKILL_VALIDATORS: dict[str, Any] = {}


def register_skill_validator(skill_name: str, validator: Any) -> None:
    SKILL_VALIDATORS[str(skill_name).strip().lower()] = validator


def validate_executor_skill_result_generic(
    executor_meta: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    if not executor_meta:
        return out, reco
    kind = "skill_executor_result"
    er = executor_meta.get("execution_result")
    if not isinstance(er, Mapping):
        return out, reco
    for idx, rec in enumerate(er.get("records", []) if isinstance(er.get("records"), list) else []):
        if not isinstance(rec, Mapping):
            continue
        st = str(rec.get("status") or "").lower()
        dry = rec.get("dry_run")
        md = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        if not isinstance(dry, bool):
            out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="warning", code="dry_run_not_explicit", path=f"executor.execution_result.records[{idx}].dry_run", message="dry_run/live mode should be explicit.", retry_recommended=True))
        action_type = md.get("action_type") or md.get("requested_action_type")
        target_type = md.get("target_type")
        if st == "success" and (not action_type or not target_type):
            out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="error", code="success_missing_action_or_target", path=f"executor.execution_result.records[{idx}].metadata", message="Successful record should include action_type and target_type.", retry_recommended=True))
            reco = DecisionAction.RECORD
        skill_name = rec.get("skill_name")
        skill_version = rec.get("skill_version")
        if not isinstance(skill_name, str) or not skill_name.strip():
            out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="warning", code="missing_skill_name", path=f"executor.execution_result.records[{idx}].skill_name", message="Execution record should include skill_name.", retry_recommended=True))
        if not isinstance(skill_version, str) or not skill_version.strip():
            out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="warning", code="missing_skill_version", path=f"executor.execution_result.records[{idx}].skill_version", message="Execution record should include skill_version.", retry_recommended=True))
        if st == "success" and str(target_type or "").strip().lower() == "file":
            sandbox_status = str(rec.get("sandbox_status") or "").strip().lower()
            if sandbox_status not in {"ok", "not_required"}:
                out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="critical", code="file_op_sandbox_status_missing", path=f"executor.execution_result.records[{idx}].sandbox_status", message="File operations require explicit sandbox_status.", escalation_recommended=True))
        if st == "failed" and not (md.get("reason_code") or md.get("reason")):
            out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="warning", severity="warning", code="failed_without_reason", path=f"executor.execution_result.records[{idx}].metadata", message="Failed execution should include reason/reason_code."))
        if st == "success" and dry is False:
            allowed = str(md.get("policy_status") or "").lower() in {"allowed", "allow"}
            if not allowed:
                out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="failed", severity="critical", code="live_execution_policy_mismatch", path=f"executor.execution_result.records[{idx}].metadata.policy_status", message="Live execution requires policy-allowed metadata.", escalation_recommended=True))
                reco = DecisionAction.RECORD
    if not any(r.get("status") == "failed" for r in out):
        out.append(artifact_result(validator="skill_result_generic", artifact_kind=kind, status="passed", severity="info", code="ok", path="executor.execution_result", message="Skill validator hooks passed for executor output."))
    return out, reco


def validate_policy_planner_executor_consistency(
    *,
    decision_action: DecisionAction | None,
    policy_meta: Mapping[str, Any] | None,
    behavior_trace: Mapping[str, Any] | None,
    planner_plan: Mapping[str, Any] | None,
    executor_meta: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], DecisionAction | None]:
    out: list[dict[str, Any]] = []
    reco: DecisionAction | None = None
    kind = "cross_artifact_consistency"
    final_decision = _coerce_decision_str((behavior_trace or {}).get("final_decision"))
    decision_is_act = decision_action == DecisionAction.ACT
    policy_status = str((policy_meta or {}).get("policy_status") or (behavior_trace or {}).get("policy_result") or "").lower()
    approval_token = (policy_meta or {}).get("approval_token_valid")
    if decision_is_act and policy_status == "denied":
        out.append(artifact_result(validator="policy_behavior_consistency", artifact_kind=kind, status="failed", severity="critical", code="act_while_policy_denied", path="behavior.decision_trace.final_decision", message="Behavior selected ACT while policy denied.", escalation_recommended=True))
        reco = DecisionAction.RECORD
    if decision_is_act and policy_status == "approval_required" and approval_token is not True:
        out.append(artifact_result(validator="policy_behavior_consistency", artifact_kind=kind, status="failed", severity="error", code="act_without_approval_token", path="policy.policy_status", message="ACT selected while approval required without approval token.", escalation_recommended=True))
        reco = DecisionAction.ASK
    if not decision_is_act and final_decision == "act" and policy_status in {"denied", "approval_required"}:
        out.append(artifact_result(validator="policy_behavior_consistency", artifact_kind=kind, status="info", severity="info", code="behavior_act_downgraded_by_policy", path="behavior.decision_trace.final_decision", message="Behavior trace ACT appears to be downgraded by policy/runtime."))
    if final_decision == "ask" and policy_status == "approval_required":
        out.append(artifact_result(validator="policy_behavior_consistency", artifact_kind=kind, status="info", severity="info", code="ask_for_approval_ok", path="behavior.decision_trace.final_decision", message="ASK is compatible with approval_required policy."))
    if isinstance(planner_plan, Mapping):
        steps = planner_plan.get("steps")
        executable_steps = [s for s in steps if isinstance(s, Mapping) and str((s.get("metadata") or {}).get("action_type") or "").lower() not in {"", "noop"}] if isinstance(steps, list) else []
        if executable_steps and not isinstance(policy_meta, Mapping):
            out.append(artifact_result(validator="planner_policy_consistency", artifact_kind=kind, status="warning", severity="warning", code="planner_missing_policy_evaluation", path="planner.plan.steps", message="Plan has executable steps but policy evaluation metadata is missing.", retry_recommended=True))
        for idx, step in enumerate(executable_steps):
            st = str(step.get("status") or "").lower()
            pol = str(step.get("policy_status") or "").lower()
            if pol == "denied" and st != "blocked":
                out.append(artifact_result(validator="planner_policy_consistency", artifact_kind=kind, status="failed", severity="error", code="denied_step_marked_executable", path=f"planner.plan.steps[{idx}].status", message="Denied step should be blocked/non-executable."))
                reco = reco or DecisionAction.ASK
            if pol == "approval_required" and not (step.get("requires_approval") or (step.get("metadata") or {}).get("approval")):
                out.append(artifact_result(validator="planner_policy_consistency", artifact_kind=kind, status="failed", severity="warning", code="approval_required_step_missing_metadata", path=f"planner.plan.steps[{idx}]", message="Approval-required step lacks approval metadata.", retry_recommended=True))
    if isinstance(executor_meta, Mapping):
        er = executor_meta.get("execution_result")
        if isinstance(er, Mapping):
            for idx, rec in enumerate(er.get("records", []) if isinstance(er.get("records"), list) else []):
                if not isinstance(rec, Mapping):
                    continue
                md = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
                dry = rec.get("dry_run")
                status = str(rec.get("status") or "").lower()
                pol = str(md.get("policy_status") or "").lower()
                if dry is False and pol not in {"allowed", "allow"}:
                    out.append(artifact_result(validator="executor_policy_consistency", artifact_kind=kind, status="failed", severity="critical", code="live_execution_without_policy_allowed", path=f"executor.execution_result.records[{idx}].metadata.policy_status", message="Live execution requires policy allowed.", escalation_recommended=True))
                    reco = DecisionAction.RECORD
                if status == "success" and (not md.get("action_type") or not md.get("target_type")):
                    out.append(artifact_result(validator="executor_policy_consistency", artifact_kind=kind, status="failed", severity="error", code="success_unsupported_action_target", path=f"executor.execution_result.records[{idx}].metadata", message="Success record missing supported action/target.", retry_recommended=True))
                    reco = reco or DecisionAction.RECORD
                if status == "failed" and not (md.get("reason_code") or md.get("reason")):
                    out.append(artifact_result(validator="executor_policy_consistency", artifact_kind=kind, status="warning", severity="warning", code="failed_execution_missing_reason", path=f"executor.execution_result.records[{idx}].metadata", message="Failed execution should include reason metadata."))
    if not any(str(r.get("status")) == "failed" for r in out):
        out.append(artifact_result(validator="cross_artifact_consistency", artifact_kind=kind, status="passed", severity="info", code="ok", path="consistency", message="Policy/Planner/Executor consistency checks complete."))
    return out, reco


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
    context_meta = None
    world_meta = None
    learning_meta = None
    plan = None
    executor_er = None
    executor_meta = None
    behavior_meta = None

    for fr in facet_results:
        name = getattr(fr, "facet_name", "")
        md = getattr(fr, "metadata", None)
        if not isinstance(md, Mapping):
            continue
        if name == "behavior":
            behavior_meta = md
            behavior_trace = md.get("decision_trace")
            context_load = md.get("context_load") or context_load
        if name == "context":
            context_meta = md
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
            executor_meta = md
        if name == "world_model":
            world_meta = md

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
    if isinstance(context_meta, Mapping):
        rows.extend(
            validate_context_v2_packet(
                context_meta,
                behavior_trace=behavior_trace if isinstance(behavior_trace, Mapping) else None,
                event={
                    "event_type": nexus_ctx.get("event_type") if isinstance(nexus_ctx, Mapping) else None
                },
            )
        )
    if isinstance(world_meta, Mapping):
        rows.extend(
            validate_world_model_v1_artifacts(
                world_meta,
                decision_is_act=decision_is_act,
            )
        )

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
    skill_rows, skill_reco = validate_executor_skill_result_generic(executor_meta)
    rows.extend(skill_rows)
    _merge_r(skill_reco)

    consistency_rows, consistency_reco = validate_policy_planner_executor_consistency(
        decision_action=decision_action,
        policy_meta=policy_meta if isinstance(policy_meta, Mapping) else None,
        behavior_trace=behavior_trace if isinstance(behavior_trace, Mapping) else None,
        planner_plan=plan if isinstance(plan, Mapping) else None,
        executor_meta=executor_meta if isinstance(executor_meta, Mapping) else None,
    )
    rows.extend(consistency_rows)
    _merge_r(consistency_reco)

    if isinstance(behavior_meta, Mapping):
        rows.extend(
            validate_output_metadata_and_capability_claims(
                behavior_meta,
                available_traces={
                    "executor": executor_meta,
                    "policy": policy_meta,
                    "world_model": world_meta,
                    "context": context_meta,
                },
            )
        )

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
