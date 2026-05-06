"""Deterministic verification checks for Fullerene Verifier (v0+v1 facets)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence as TypingSeq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fullerene.nexus.models import (
    DecisionAction,
    Event,
    FacetResult,
    NexusDecision,
    NexusState,
)
from fullerene.policy import (
    PolicyRuleType,
    PolicyStatus,
    PolicyTargetType,
    coerce_policy_target_type,
)
from fullerene.verifier.models import (
    VerificationResult,
    VerificationSeverity,
    VerificationStatus,
    VerificationSummary,
)
from fullerene.verifier.artifacts import run_all_artifact_validators

EXTERNAL_SIDE_EFFECT_TARGET_TYPES = frozenset(
    {
        PolicyTargetType.FILE_WRITE,
        PolicyTargetType.FILE_DELETE,
        PolicyTargetType.SHELL,
        PolicyTargetType.NETWORK,
        PolicyTargetType.MESSAGE,
        PolicyTargetType.GIT,
        PolicyTargetType.TOOL,
    }
)


@dataclass(slots=True)
class VerificationContext:
    event: Event
    state: NexusState
    facet_results: list[FacetResult]
    decision: NexusDecision | None
    state_dir: Path | None = None


class VerificationCheck(Protocol):
    name: str

    def run(self, context: VerificationContext) -> VerificationResult:
        """Run a deterministic verification check."""


class DecisionShapeCheck:
    name = "decision_shape"

    def run(self, context: VerificationContext) -> VerificationResult:
        if context.decision is None:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.CRITICAL,
                message="Final Nexus decision is missing.",
                metadata={"recommended_action": DecisionAction.RECORD.value},
            )

        action = _coerce_decision_action(getattr(context.decision, "action", None))
        if action is None:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.CRITICAL,
                message="Final Nexus decision action is invalid.",
                metadata={
                    "recommended_action": DecisionAction.RECORD.value,
                    "raw_action": getattr(context.decision, "action", None),
                },
            )

        reason = getattr(context.decision, "reason", None)
        if not isinstance(reason, str) or not reason.strip():
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message="Final Nexus decision is missing a non-empty reason.",
                metadata={
                    "recommended_action": DecisionAction.RECORD.value,
                    "action": action.value,
                },
            )

        raw_metadata = getattr(context.decision, "metadata", None)
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message="Final Nexus decision metadata must be dict-like when present.",
                metadata={
                    "recommended_action": DecisionAction.RECORD.value,
                    "action": action.value,
                },
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Final Nexus decision has a valid action and reason.",
            metadata={"action": action.value},
        )


class FacetResultShapeCheck:
    name = "facet_result_shape"

    def run(self, context: VerificationContext) -> VerificationResult:
        issues: list[str] = []

        for index, result in enumerate(context.facet_results):
            facet_name = getattr(result, "facet_name", None)
            label = (
                facet_name.strip()
                if isinstance(facet_name, str) and facet_name.strip()
                else f"facet_result[{index}]"
            )

            if not isinstance(facet_name, str) or not facet_name.strip():
                issues.append(f"{label}: missing facet_name")

            proposed_decision = getattr(result, "proposed_decision", None)
            if proposed_decision is not None and _coerce_decision_action(
                proposed_decision
            ) is None:
                issues.append(f"{label}: invalid proposed_decision")

            metadata = getattr(result, "metadata", None)
            if metadata is not None and not isinstance(metadata, Mapping):
                issues.append(f"{label}: metadata must be dict-like")

        if issues:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message="Facet results contain malformed structural fields.",
                metadata={
                    "issues": issues,
                    "recommended_action": DecisionAction.RECORD.value,
                },
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Facet results have valid names, decisions, and metadata shapes.",
            metadata={"facet_count": len(context.facet_results)},
        )


class PolicyComplianceCheck:
    name = "policy_compliance"

    def run(self, context: VerificationContext) -> VerificationResult:
        action = _decision_action(context.decision)
        policy_results = _policy_results(context.facet_results)

        if not policy_results:
            if action == DecisionAction.ACT:
                return VerificationResult(
                    check_name=self.name,
                    status=VerificationStatus.WARNING,
                    severity=VerificationSeverity.WARNING,
                    message=(
                        "No policy facet result was available while the final decision "
                        "is ACT."
                    ),
                )
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="No policy facet result was available to validate.",
            )

        if action != DecisionAction.ACT:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="Final decision is not ACT, so policy guardrails are satisfied.",
                metadata={"policy_result_count": len(policy_results)},
            )

        if _has_policy_status(policy_results, PolicyStatus.DENIED):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.CRITICAL,
                message="Policy denied the modeled action, but the final decision is ACT.",
                metadata={"recommended_action": DecisionAction.RECORD.value},
            )

        if _has_policy_status(policy_results, PolicyStatus.APPROVAL_REQUIRED):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.CRITICAL,
                message=(
                    "Policy requires approval for the modeled action, but the final "
                    "decision is ACT."
                ),
                metadata={"recommended_action": DecisionAction.ASK.value},
            )

        dangerous_results = [
            result for result in policy_results if _is_external_policy_target(result)
        ]
        if dangerous_results and not any(
            _has_explicit_policy_allow(result) for result in dangerous_results
        ):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message=(
                    "ACT targets an external side effect, but no explicit allow policy "
                    "matched."
                ),
                metadata={"recommended_action": DecisionAction.ASK.value},
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Policy results are consistent with the final decision.",
        )


class PlanSafetyCheck:
    name = "plan_safety"

    def run(self, context: VerificationContext) -> VerificationResult:
        planner_plan = _planner_plan(context.facet_results)
        if planner_plan is None:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="No planner output was available to validate.",
            )

        issues: list[str] = []
        plan_status = str(planner_plan.get("status", "") or "").strip().lower()
        raw_steps = planner_plan.get("steps", [])
        if not isinstance(raw_steps, list):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message="Planner metadata contained a malformed steps payload.",
                metadata={"recommended_action": DecisionAction.ASK.value},
            )

        for index, step in enumerate(raw_steps):
            if not isinstance(step, Mapping):
                issues.append(f"steps[{index}] is not dict-like")
                continue
            step_id = str(step.get("id") or f"steps[{index}]")
            risk_level = str(step.get("risk_level", "") or "").strip().lower()
            requires_approval = bool(step.get("requires_approval"))
            step_status = str(step.get("status", "") or "").strip().lower()

            if risk_level == "high" and not requires_approval:
                issues.append(f"{step_id}: high-risk step must require approval")
            if step_status == "blocked" and plan_status == "approved":
                issues.append(f"{step_id}: blocked step cannot appear in an approved plan")

        if issues:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                message="Planner output violates deterministic plan safety rules.",
                metadata={
                    "issues": issues,
                    "recommended_action": DecisionAction.ASK.value,
                },
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Planner output satisfies deterministic risk and approval rules.",
        )


class ArtifactSchemaCheck:
    name = "artifact_schemas"

    def run(self, context: VerificationContext) -> VerificationResult:
        nexus_bucket: dict[str, Any] = {}
        fs = getattr(context.state, "facet_state", None)
        if isinstance(fs, Mapping):
            raw = fs.get("nexus")
            if isinstance(raw, Mapping):
                nexus_bucket = dict(raw)
        nexus_ctx = nexus_bucket.get("verifier_cycle_context")
        if not isinstance(nexus_ctx, Mapping):
            nexus_ctx = {}

        prior_trace_raw = nexus_bucket.get("last_cycle_trace")
        prior_cycle_trace = (
            dict(prior_trace_raw)
            if isinstance(prior_trace_raw, Mapping)
            else None
        )
        artifact_rows, reco = run_all_artifact_validators(
            facet_results=context.facet_results,
            decision_action=_decision_action(context.decision),
            nexus_ctx=nexus_ctx,
            prior_cycle_trace=prior_cycle_trace,
        )

        interesting: list[dict[str, Any]] = []
        for row in artifact_rows:
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("code") or "")
            if row.get("status") == "passed" and code in {"ok"}:
                continue
            interesting.append(dict(row))

        severity_order = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        worst = 0
        for row in interesting:
            worst = max(worst, severity_order.get(str(row.get("severity") or "info"), 0))

        worst_sev = (
            VerificationSeverity.CRITICAL
            if worst >= 3
            else VerificationSeverity.ERROR
            if worst >= 2
            else VerificationSeverity.WARNING
            if worst >= 1
            else VerificationSeverity.INFO
        )

        has_warning_finding = any(
            str(row.get("severity") or "") == "warning" for row in interesting
        )
        has_hard_finding = any(
            str(row.get("severity") or "") in {"error", "critical"} for row in interesting
        )

        metadata: dict[str, Any] = {
            "artifact_checks": artifact_rows,
            "schema_checks": artifact_rows,
            "validated_artifact_kinds": sorted(
                {
                    str(row["artifact_kind"])
                    for row in artifact_rows
                    if isinstance(row, Mapping) and row.get("artifact_kind")
                }
            ),
            "artifact_worst_severity": worst_sev.value,
        }
        if reco is not None:
            metadata["recommended_action"] = reco.value

        if reco is not None or has_hard_finding:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.FAILED,
                severity=worst_sev,
                message="Verifier v1 deterministic artifact/schema checks failed.",
                metadata=metadata,
            )

        if interesting and has_warning_finding:
            metadata["recommended_action"] = None
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.WARNING,
                severity=VerificationSeverity.WARNING,
                message="Verifier v1 found deterministic schema/metadata warnings.",
                metadata=metadata,
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            message="Verifier v1 artifact/schema checks passed.",
            metadata=metadata,
        )


class ActRequiresApprovalCheck:
    name = "act_requires_approval"

    def run(self, context: VerificationContext) -> VerificationResult:
        if _decision_action(context.decision) != DecisionAction.ACT:
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="Final decision is not ACT.",
            )

        if _metadata_flag(context.event.metadata, "low_risk"):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="ACT is allowed because the event metadata marks it low-risk.",
            )

        if any(
            _has_explicit_policy_allow(result)
            for result in _policy_results(context.facet_results)
        ):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="ACT is allowed because an explicit policy allow matched.",
            )

        if _targets_text_response(context):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message="ACT is allowed because it only emits deterministic text output.",
            )

        if _targets_internal_state(context):
            return VerificationResult(
                check_name=self.name,
                status=VerificationStatus.PASSED,
                severity=VerificationSeverity.INFO,
                message=(
                    "ACT is allowed because it targets Fullerene internal state inside "
                    "the configured state-dir."
                ),
            )

        return VerificationResult(
            check_name=self.name,
            status=VerificationStatus.FAILED,
            severity=VerificationSeverity.CRITICAL,
            message=(
                "ACT requires explicit low_risk metadata, an explicit policy allow, "
                "or an internal_state target inside state-dir."
            ),
            metadata={"recommended_action": DecisionAction.ASK.value},
        )


DEFAULT_CHECKS: tuple[VerificationCheck, ...] = (
    DecisionShapeCheck(),
    FacetResultShapeCheck(),
    PolicyComplianceCheck(),
    PlanSafetyCheck(),
    ArtifactSchemaCheck(),
    ActRequiresApprovalCheck(),
)


def run_verification_checks(
    *,
    event: Event,
    state: NexusState,
    facet_results: Sequence[FacetResult],
    decision: NexusDecision | None,
    checks: Sequence[VerificationCheck] | None = None,
    state_dir: Path | str | None = None,
) -> VerificationSummary:
    resolved_state_dir = _coerce_state_dir(state_dir)
    context = VerificationContext(
        event=event,
        state=state,
        facet_results=list(facet_results),
        decision=decision,
        state_dir=resolved_state_dir,
    )
    active_checks = tuple(checks or DEFAULT_CHECKS)
    results = [check.run(context) for check in active_checks]
    failed_checks = [
        result.check_name
        for result in results
        if result.status == VerificationStatus.FAILED
    ]
    warnings = [
        result.message
        for result in results
        if result.status == VerificationStatus.WARNING
    ]

    overall_status = VerificationStatus.PASSED
    if failed_checks:
        overall_status = VerificationStatus.FAILED
    elif warnings:
        overall_status = VerificationStatus.WARNING

    decision_action = _decision_action(decision)
    metadata: dict[str, Any] = {
        "check_count": len(active_checks),
        "facet_count": len(context.facet_results),
        "event_id": event.event_id,
        "decision_action": decision_action.value if decision_action is not None else None,
    }
    if resolved_state_dir is not None:
        metadata["state_dir"] = str(resolved_state_dir)

    summary = VerificationSummary(
        overall_status=overall_status,
        results=results,
        failed_checks=failed_checks,
        warnings=warnings,
        metadata=metadata,
    )
    finalize_verifier_v1_summary(summary)
    return summary


def finalize_verifier_v1_summary(summary: VerificationSummary) -> None:
    """Attach Verifier v1 inspectable aggregates (deterministic JSON-friendly)."""

    summary.metadata.setdefault("verifier_version", "v1")
    artifact_checks: list[dict[str, Any]] = []

    failure_messages = [
        r.message for r in summary.results if r.status == VerificationStatus.FAILED
    ]
    for result in summary.results:
        if result.check_name == "artifact_schemas" and isinstance(result.metadata, dict):
            artifact_checks = list(result.metadata.get("artifact_checks", []))

    summary.metadata["artifact_checks"] = artifact_checks
    summary.metadata["schema_checks"] = list(artifact_checks)

    validation_codes: list[str] = []
    for row in artifact_checks:
        if not isinstance(row, Mapping):
            continue
        code = row.get("code")
        if code and code not in {"ok", "summary"}:
            validation_codes.append(str(code))
    summary.metadata["validation_codes"] = validation_codes

    kinds = sorted(
        {
            str(row["artifact_kind"])
            for row in artifact_checks
            if isinstance(row, Mapping) and row.get("artifact_kind")
        }
    )
    summary.metadata["validated_artifact_kinds"] = kinds

    retry_reasons = [
        str(row.get("message") or "")
        for row in artifact_checks
        if isinstance(row, Mapping)
        and row.get("retry_recommended")
        and row.get("message")
    ]
    escalation_reasons = [
        str(row.get("message") or "")
        for row in artifact_checks
        if isinstance(row, Mapping)
        and row.get("escalation_recommended")
        and row.get("message")
    ]
    for result in summary.results:
        if result.check_name != "artifact_schemas" and result.metadata.get(
            "recommended_action"
        ):
            meta = result.metadata if isinstance(result.metadata, dict) else {}
            reasons = meta.get("issues") or [result.message]
            if isinstance(reasons, TypingSeq) and not isinstance(reasons, (str, bytes)):
                escalation_reasons.extend(str(x) for x in reasons)

    summary.metadata["retry_recommended"] = bool(retry_reasons)
    summary.metadata["escalation_recommended"] = bool(
        escalation_reasons
        or any(
            r.status == VerificationStatus.FAILED
            and r.severity == VerificationSeverity.CRITICAL
            for r in summary.results
        )
    )

    summary.metadata["retry_reasons"] = list(dict.fromkeys(retry_reasons))
    summary.metadata["escalation_reasons"] = list(
        dict.fromkeys((*escalation_reasons, *failure_messages))
    )
    affected_artifact_ids: list[str] = []
    for row in artifact_checks:
        if not isinstance(row, Mapping):
            continue
        path = row.get("path")
        if isinstance(path, str) and path:
            affected_artifact_ids.append(path)
    summary.metadata["affected_artifact_ids"] = list(dict.fromkeys(affected_artifact_ids))
    suggested_safe_decision = None
    downgraded = verifier_downgraded_decision(summary)
    if isinstance(downgraded, dict):
        to_value = str(downgraded.get("to") or "").strip().lower()
        if to_value in {"wait", "record", "ask"}:
            suggested_safe_decision = to_value.upper()
    summary.metadata["suggested_safe_decision"] = suggested_safe_decision

def verifier_downgraded_decision(summary: VerificationSummary) -> dict[str, Any] | None:
    """Peek at aggregated recommended_action hints from failed verifier checks."""

    if summary.overall_status != VerificationStatus.FAILED:
        return None
    for result in reversed(summary.results):
        meta = result.metadata if isinstance(result.metadata, dict) else {}
        raw = meta.get("recommended_action")
        if raw:
            anchor = summary.metadata.get("decision_action")
            payload: dict[str, Any] = {"to": str(raw)}
            if anchor is not None:
                payload["from"] = str(anchor)
            return payload
    return None


def _coerce_state_dir(raw_state_dir: Path | str | None) -> Path | None:
    if raw_state_dir is None:
        return None
    return Path(raw_state_dir).expanduser()


def _coerce_decision_action(raw_action: Any) -> DecisionAction | None:
    if isinstance(raw_action, DecisionAction):
        return raw_action
    if not isinstance(raw_action, str):
        return None
    cleaned = raw_action.strip().lower()
    if not cleaned:
        return None
    try:
        return DecisionAction(cleaned)
    except ValueError:
        return None


def _decision_action(decision: NexusDecision | None) -> DecisionAction | None:
    if decision is None:
        return None
    return _coerce_decision_action(getattr(decision, "action", None))


def _policy_results(facet_results: Sequence[FacetResult]) -> list[FacetResult]:
    matches: list[FacetResult] = []
    for result in facet_results:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping) and "policy_status" in metadata:
            matches.append(result)
    return matches


def _planner_plan(facet_results: Sequence[FacetResult]) -> Mapping[str, Any] | None:
    for result in facet_results:
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        plan = metadata.get("plan")
        if isinstance(plan, Mapping):
            return plan
    return None


def _has_policy_status(
    policy_results: Sequence[FacetResult],
    status: PolicyStatus,
) -> bool:
    for result in policy_results:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping) and metadata.get("policy_status") == status.value:
            return True
    return False


def _is_external_policy_target(result: FacetResult) -> bool:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    target_type = coerce_policy_target_type(metadata.get("target_type"))
    return target_type in EXTERNAL_SIDE_EFFECT_TARGET_TYPES


def _has_explicit_policy_allow(result: FacetResult) -> bool:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("policy_status") != PolicyStatus.ALLOWED.value:
        return False

    effective_policy = metadata.get("effective_policy")
    if isinstance(effective_policy, Mapping):
        return (
            effective_policy.get("rule_type") == PolicyRuleType.ALLOW.value
            and not bool(effective_policy.get("built_in"))
        )

    matched_policies = metadata.get("matched_policies")
    if not isinstance(matched_policies, list):
        return False

    for policy in matched_policies:
        if not isinstance(policy, Mapping):
            continue
        if (
            policy.get("rule_type") == PolicyRuleType.ALLOW.value
            and not bool(policy.get("built_in"))
        ):
            return True
    return False


def _metadata_flag(metadata: dict[str, Any], key: str) -> bool:
    raw_value = metadata.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _targets_internal_state(context: VerificationContext) -> bool:
    for result in _policy_results(context.facet_results):
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("is_internal_state_action"):
            return True
        if metadata.get("target") == "state-dir":
            return True
        if metadata.get("within_state_dir"):
            return True

    raw_target_type = coerce_policy_target_type(context.event.metadata.get("target_type"))
    target = str(context.event.metadata.get("target", "") or "").strip()
    if raw_target_type == PolicyTargetType.INTERNAL_STATE and not context.event.metadata.get("path"):
        return True
    if target.casefold() == "state-dir":
        return True

    raw_path = context.event.metadata.get("path")
    if (
        raw_target_type == PolicyTargetType.INTERNAL_STATE
        and isinstance(raw_path, str)
        and context.state_dir is not None
    ):
        return _within_state_dir(raw_path, context.state_dir)

    return False


def _targets_text_response(context: VerificationContext) -> bool:
    for result in context.facet_results:
        if _coerce_decision_action(getattr(result, "proposed_decision", None)) != DecisionAction.ACT:
            continue
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        if (
            metadata.get("output_type") == "text"
            and metadata.get("tool") == "text"
            and bool(metadata.get("response_needed"))
        ):
            return True
    return False


def _within_state_dir(raw_path: str, state_dir: Path) -> bool:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = state_dir / candidate
    normalized_candidate = Path(os.path.normpath(str(candidate)))
    normalized_state_dir = Path(os.path.normpath(str(state_dir)))
    try:
        normalized_candidate.relative_to(normalized_state_dir)
        return True
    except ValueError:
        return False
