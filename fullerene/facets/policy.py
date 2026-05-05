"""Deterministic policy facet for Fullerene Policy v1 (compat with Policy v0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fullerene.memory import infer_tags, merge_tags, normalize_tags
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.policy import (
    PolicyApprovalScope,
    PolicyEvaluation,
    PolicyEffectiveAction,
    PolicyPrecedenceTraceEntry,
    PolicyRule,
    PolicyRuleType,
    PolicySource,
    PolicyStatus,
    PolicyRiskLevel,
    PolicyStore,
    PolicyTargetType,
    SQLitePolicyStore,
    coerce_policy_target_type,
)

INTERNAL_STATE_ALLOW_RULE = PolicyRule(
    id="builtin-allow-internal-state",
    name="Allow internal state CRUD",
    description=(
        "Fullerene may create, update, and delete its own state within the "
        "configured state directory."
    ),
    rule_type=PolicyRuleType.ALLOW,
    target_type=PolicyTargetType.INTERNAL_STATE,
    target="state-dir",
    conditions={"within_state_dir": True},
    priority=100.0,
    source=PolicySource.SYSTEM,
    metadata={"built_in": True, "baseline": True},
)

INTERNAL_STATE_LIVE_HIGH_RISK_REQUIRE_APPROVAL_RULE = PolicyRule(
    id="builtin-require-approval-internal-state-live-high-risk",
    name="Require approval for live internal high-risk",
    description=(
        "Live (non-dry-run) internal state actions that are high risk require "
        "explicit approval."
    ),
    rule_type=PolicyRuleType.REQUIRE_APPROVAL,
    target_type=PolicyTargetType.INTERNAL_STATE,
    target="state-dir",
    conditions={"within_state_dir": True},
    priority=50.0,
    source=PolicySource.SYSTEM,
    metadata={"built_in": True, "baseline": True, "fallback": True},
)

UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE = PolicyRule(
    id="builtin-unknown-target-type-require-approval",
    name="Require approval for unknown target type",
    description=(
        "When Fullerene cannot model the action's target type, execution is "
        "treated as requiring explicit approval."
    ),
    rule_type=PolicyRuleType.REQUIRE_APPROVAL,
    target_type=PolicyTargetType.GENERAL,
    target="*",
    priority=-101.0,
    source=PolicySource.SYSTEM,
    metadata={"built_in": True, "baseline": True, "fallback": True},
)

EXTERNAL_APPROVAL_RULES = {
    target_type: PolicyRule(
        id=f"builtin-require-approval-{target_type.value}",
        name=f"Require approval for {target_type.value}",
        description=(
            "External side effects require explicit approval or an explicit "
            "allow policy."
        ),
        rule_type=PolicyRuleType.REQUIRE_APPROVAL,
        target_type=target_type,
        target="*",
        priority=-100.0,
        source=PolicySource.SYSTEM,
        metadata={"built_in": True, "baseline": True, "fallback": True},
    )
    for target_type in (
        PolicyTargetType.FILE_WRITE,
        PolicyTargetType.FILE_DELETE,
        PolicyTargetType.SHELL,
        PolicyTargetType.NETWORK,
        PolicyTargetType.MESSAGE,
        PolicyTargetType.GIT,
        PolicyTargetType.TOOL,
    )
}


@dataclass(slots=True)
class _PolicyContext:
    metadata: dict[str, Any]
    explicit_action: bool
    current_decision: DecisionAction | None
    raw_target_type: PolicyTargetType | None
    target: str
    operation: str
    path: str | None
    resolved_path: Path | None
    within_state_dir: bool
    is_internal_state_action: bool
    tags: list[str]
    state_dir: Path
    action_type: str | None
    risk_level: PolicyRiskLevel
    source_facet: str | None
    plan_id: str | None
    plan_step_id: str | None
    skill_name: str | None
    dry_run: bool
    live_mode: bool
    external_side_effect: bool
    approval_token: dict[str, Any] | None

    @property
    def is_action_candidate(self) -> bool:
        return bool(
            self.explicit_action
            or self.raw_target_type is not None
            or self.path
            or self.current_decision == DecisionAction.ACT
        )


class PolicyFacet:
    """Evaluate explicit policy rules plus built-in sandbox defaults."""

    name = "policy"

    def __init__(self, store: PolicyStore, *, state_dir: Path | str) -> None:
        self.store = store
        self.state_dir = Path(state_dir).expanduser().resolve()

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        state_dir: Path | str,
    ) -> "PolicyFacet":
        return cls(SQLitePolicyStore(path), state_dir=state_dir)

    def process(self, event: Event, state: NexusState) -> FacetResult:
        context = self._build_context(event, state)
        evaluation_scope = (
            PolicyApprovalScope.PLAN_STEP
            if context.plan_step_id is not None
            else PolicyApprovalScope.ACTION
        )
        if self._metadata_flag(event.metadata, "execute_plan"):
            plan = self._get_planner_last_plan(state)
            if plan is not None:
                return self._evaluate_plan_v1(event=event, state=state, plan_dict=plan)
        if not context.is_action_candidate:
            result = self._build_result(
                status=PolicyStatus.NO_MATCH,
                summary="Policy facet found no modeled action or policy target to evaluate.",
                matched_policies=[],
                reasons=["no_policy_target"],
                context=context,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=[],
                effective_policy=None,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.NO_MATCH,
                    effective_action=PolicyEffectiveAction.NONE,
                    matched_policies=[],
                    effective_policy=None,
                    decisive_reason="no_policy_target",
                    reasons=["no_policy_target"],
                    warnings=[],
                    fallback_applied=False,
                    sandbox_scope="",
                    token_validation=None,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        explicit_matches = [
            rule
            for rule in self.store.list_enabled_policies()
            if self._rule_matches(rule, context)
        ]
        ordered_matches = self._sort_rules(explicit_matches)
        denied = self._top_rule(ordered_matches, PolicyRuleType.DENY)
        approval = self._top_rule(ordered_matches, PolicyRuleType.REQUIRE_APPROVAL)
        allowed = self._top_rule(ordered_matches, PolicyRuleType.ALLOW)
        preferred = self._rules_of_type(ordered_matches, PolicyRuleType.PREFER)

        if denied is not None:
            reasons = [
                f"deny:{denied.name}",
                "deny_rules_override_allow_and_prefer",
            ]
            result = self._build_result(
                status=PolicyStatus.DENIED,
                summary=f"Policy facet denied the modeled action via {denied.name!r}.",
                matched_policies=ordered_matches,
                reasons=reasons,
                context=context,
                proposed_decision=DecisionAction.RECORD,
                effective_policy=denied,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=ordered_matches,
                effective_policy=denied,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.DENIED,
                    effective_action=PolicyEffectiveAction.DENY,
                    matched_policies=ordered_matches,
                    effective_policy=denied,
                    decisive_reason="explicit_deny_wins",
                    reasons=reasons,
                    warnings=[],
                    fallback_applied=False,
                    sandbox_scope="",
                    token_validation=None,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        if approval is not None:
            reasons = [
                f"require_approval:{approval.name}",
                "approval_rules_override_allow_and_prefer",
            ]
            token_validation = self._validate_approval_token(
                context, evaluation_scope=evaluation_scope.value
            )
            token_valid = bool(token_validation.get("valid") is True)
            token_conversion_applied = token_valid
            if token_valid:
                result = self._build_result(
                    status=PolicyStatus.ALLOWED,
                    summary=(
                        "Policy facet allowed the modeled action via an explicit "
                        f"approval token for {approval.name!r}."
                    ),
                    matched_policies=ordered_matches,
                    reasons=["approval_token_converted_require_approval_to_allow"],
                    context=context,
                    proposed_decision=None,
                    effective_policy=approval,
                )
                trace = self._build_rule_precedence_trace_v1(
                    context=context,
                    matched_policies=ordered_matches,
                    effective_policy=approval,
                    token_conversion_applied=True,
                )
                result.metadata["policy_evaluation"] = (
                    self._build_policy_evaluation_v1_from_outcome(
                        context=context,
                        evaluation_scope=evaluation_scope,
                        status=PolicyStatus.ALLOWED,
                        effective_action=PolicyEffectiveAction.ALLOW,
                        matched_policies=ordered_matches,
                        effective_policy=approval,
                        decisive_reason="approval_converted_by_token",
                        reasons=list(result.metadata.get("reasons", [])),
                        warnings=[],
                        fallback_applied=False,
                        sandbox_scope="",
                        token_validation=token_validation,
                        token_conversion_applied=True,
                        rule_precedence_trace=trace,
                    )
                )
                return result

            result = self._build_result(
                status=PolicyStatus.APPROVAL_REQUIRED,
                summary=(
                    "Policy facet requires approval for the modeled action via "
                    f"{approval.name!r}."
                ),
                matched_policies=ordered_matches,
                reasons=reasons,
                context=context,
                proposed_decision=DecisionAction.ASK,
                effective_policy=approval,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=ordered_matches,
                effective_policy=approval,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                        evaluation_scope=evaluation_scope,
                    status=PolicyStatus.APPROVAL_REQUIRED,
                    effective_action=PolicyEffectiveAction.REQUIRE_APPROVAL,
                    matched_policies=ordered_matches,
                    effective_policy=approval,
                    decisive_reason="approval_required_rule",
                    reasons=reasons,
                    warnings=token_validation.get("warnings", []),
                    fallback_applied=False,
                    sandbox_scope="",
                    token_validation=token_validation,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        baseline_rule = self._baseline_rule_for_context(context)
        baseline_status = self._baseline_status_for_context(context)

        if allowed is not None:
            reasons = [f"allow:{allowed.name}"]
            if baseline_status == PolicyStatus.APPROVAL_REQUIRED:
                reasons.append("explicit_allow_overrides_default_external_approval")
            if preferred:
                reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            result = self._build_result(
                status=PolicyStatus.ALLOWED,
                summary=f"Policy facet allowed the modeled action via {allowed.name!r}.",
                matched_policies=ordered_matches,
                reasons=reasons,
                context=context,
                effective_policy=allowed,
                baseline_status=baseline_status,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=ordered_matches,
                effective_policy=allowed,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.ALLOWED,
                    effective_action=PolicyEffectiveAction.ALLOW,
                    matched_policies=ordered_matches,
                    effective_policy=allowed,
                    decisive_reason="explicit_allow_wins",
                    reasons=reasons,
                    warnings=[],
                    fallback_applied=False,
                    sandbox_scope="",
                    token_validation=None,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        if baseline_status == PolicyStatus.ALLOWED and baseline_rule is not None:
            baseline_matches = [baseline_rule, *preferred]
            reasons = ["baseline_internal_state_allow"]
            reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            result = self._build_result(
                status=PolicyStatus.ALLOWED,
                summary=(
                    "Policy facet allowed the modeled action because it stays inside "
                    "the configured state directory."
                ),
                matched_policies=baseline_matches,
                reasons=reasons,
                context=context,
                effective_policy=baseline_rule,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=baseline_matches,
                effective_policy=baseline_rule,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.ALLOWED,
                    effective_action=PolicyEffectiveAction.ALLOW,
                    matched_policies=baseline_matches,
                    effective_policy=baseline_rule,
                    decisive_reason="builtin_internal_state_allow",
                    reasons=reasons,
                    warnings=[],
                    fallback_applied=True,
                    sandbox_scope="internal_state_default_allow",
                    token_validation=None,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        if baseline_status == PolicyStatus.APPROVAL_REQUIRED and baseline_rule is not None:
            baseline_matches = [baseline_rule, *preferred]
            reasons = ["baseline_external_approval_required"]
            reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            token_validation = self._validate_approval_token(
                context, evaluation_scope=evaluation_scope.value
            )
            token_valid = bool(token_validation.get("valid") is True)
            token_conversion_applied = token_valid
            if token_valid:
                result = self._build_result(
                    status=PolicyStatus.ALLOWED,
                    summary=(
                        "Policy facet allowed the modeled action via an explicit "
                        "approval token for the built-in fallback rule."
                    ),
                    matched_policies=baseline_matches,
                    reasons=["approval_token_converted_builtin_approval_to_allow"],
                    context=context,
                    proposed_decision=None,
                    effective_policy=baseline_rule,
                )
                trace = self._build_rule_precedence_trace_v1(
                    context=context,
                    matched_policies=baseline_matches,
                    effective_policy=baseline_rule,
                    token_conversion_applied=True,
                )
                sandbox_scope = (
                    "internal_state_live_high_risk"
                    if context.is_internal_state_action
                    else "external_requires_approval_default"
                )
                result.metadata["policy_evaluation"] = (
                    self._build_policy_evaluation_v1_from_outcome(
                        context=context,
                        evaluation_scope=evaluation_scope,
                        status=PolicyStatus.ALLOWED,
                        effective_action=PolicyEffectiveAction.ALLOW,
                        matched_policies=baseline_matches,
                        effective_policy=baseline_rule,
                        decisive_reason="builtin_approval_converted_by_token",
                        reasons=list(result.metadata.get("reasons", [])),
                        warnings=[],
                        fallback_applied=True,
                        sandbox_scope=sandbox_scope,
                        token_validation=token_validation,
                        token_conversion_applied=True,
                        rule_precedence_trace=trace,
                    )
                )
                return result

            result = self._build_result(
                status=PolicyStatus.APPROVAL_REQUIRED,
                summary=(
                    "Policy facet requires approval because the modeled action has "
                    "external side effects and no explicit allow rule matched."
                ),
                matched_policies=baseline_matches,
                reasons=reasons,
                context=context,
                proposed_decision=DecisionAction.ASK,
                effective_policy=baseline_rule,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=baseline_matches,
                effective_policy=baseline_rule,
                token_conversion_applied=False,
            )
            sandbox_scope = (
                "internal_state_live_high_risk"
                if context.is_internal_state_action
                else "external_requires_approval_default"
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.APPROVAL_REQUIRED,
                    effective_action=PolicyEffectiveAction.REQUIRE_APPROVAL,
                    matched_policies=baseline_matches,
                    effective_policy=baseline_rule,
                    decisive_reason="builtin_fallback_requires_approval",
                    reasons=reasons,
                    warnings=token_validation.get("warnings", []),
                    fallback_applied=True,
                    sandbox_scope=sandbox_scope,
                    token_validation=token_validation,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        if preferred:
            result = self._build_result(
                status=PolicyStatus.PREFERRED,
                summary=(
                    "Policy facet matched preference-only policies and recorded the "
                    "preference metadata without changing the decision."
                ),
                matched_policies=preferred,
                reasons=[f"prefer:{rule.name}" for rule in preferred],
                context=context,
                effective_policy=preferred[0],
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=preferred,
                effective_policy=preferred[0],
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.PREFERRED,
                    effective_action=PolicyEffectiveAction.PREFER,
                    matched_policies=preferred,
                    effective_policy=preferred[0],
                    decisive_reason="preference_only",
                    reasons=[f"prefer:{rule.name}" for rule in preferred],
                    warnings=[],
                    fallback_applied=False,
                    sandbox_scope="",
                    token_validation=None,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        if context.raw_target_type is None and (
            evaluation_scope == PolicyApprovalScope.PLAN_STEP
            or (
                context.current_decision == DecisionAction.ACT
                and (
                    context.explicit_action
                    or bool(context.target)
                    or bool(context.operation)
                    or context.path is not None
                )
            )
        ):
            # Conservative v1 fallback for unknown target types.
            token_validation = self._validate_approval_token(
                context, evaluation_scope=evaluation_scope.value
            )
            result = self._build_result(
                status=PolicyStatus.APPROVAL_REQUIRED,
                summary="Policy facet fell back to approval for unknown target type.",
                matched_policies=[UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE],
                reasons=["unknown_target_type_fallback_requires_approval"],
                context=context,
                proposed_decision=DecisionAction.ASK,
                effective_policy=UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE,
            )
            trace = self._build_rule_precedence_trace_v1(
                context=context,
                matched_policies=[UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE],
                effective_policy=UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE,
                token_conversion_applied=False,
            )
            result.metadata["policy_evaluation"] = (
                self._build_policy_evaluation_v1_from_outcome(
                    context=context,
                    evaluation_scope=evaluation_scope,
                    status=PolicyStatus.APPROVAL_REQUIRED,
                    effective_action=PolicyEffectiveAction.REQUIRE_APPROVAL,
                    matched_policies=[UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE],
                    effective_policy=UNKNOWN_TARGET_TYPE_REQUIRE_APPROVAL_RULE,
                    decisive_reason="unknown_target_type_for_execution",
                    reasons=["unknown_target_type_fallback_requires_approval"],
                    warnings=token_validation.get("warnings", []),
                    fallback_applied=True,
                    sandbox_scope="unknown_target_type_fallback",
                    token_validation=token_validation,
                    token_conversion_applied=False,
                    rule_precedence_trace=trace,
                )
            )
            return result

        result = self._build_result(
            status=PolicyStatus.NO_MATCH,
            summary="Policy facet found no matching policy rules for the modeled action.",
            matched_policies=[],
            reasons=["no_matching_policy_rule"],
            context=context,
        )
        trace = self._build_rule_precedence_trace_v1(
            context=context,
            matched_policies=[],
            effective_policy=None,
            token_conversion_applied=False,
        )
        result.metadata["policy_evaluation"] = (
            self._build_policy_evaluation_v1_from_outcome(
                context=context,
                evaluation_scope=evaluation_scope,
                status=PolicyStatus.NO_MATCH,
                effective_action=PolicyEffectiveAction.NONE,
                matched_policies=[],
                effective_policy=None,
                decisive_reason="no_matching_policy_rule",
                reasons=["no_matching_policy_rule"],
                warnings=[],
                fallback_applied=False,
                sandbox_scope="",
                token_validation=None,
                token_conversion_applied=False,
                rule_precedence_trace=trace,
            )
        )
        return result

    def _get_planner_last_plan(self, state: NexusState) -> dict[str, Any] | None:
        planner_state = state.facet_state.get("planner") if isinstance(
            state.facet_state, dict
        ) else None
        if not isinstance(planner_state, dict):
            return None
        last_plan = planner_state.get("last_plan")
        return last_plan if isinstance(last_plan, dict) else None

    def _evaluate_plan_v1(
        self,
        *,
        event: Event,
        state: NexusState,
        plan_dict: dict[str, Any],
    ) -> FacetResult:
        """Aggregate step-level `policy_evaluation` into a plan decision."""
        steps_raw = plan_dict.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            # No steps -> no modeled target.
            policy_eval = self._build_policy_evaluation_v1_from_outcome(  # type: ignore[call-arg]
                context=self._build_context(event, state),
                evaluation_scope=PolicyApprovalScope.ACTION,
                status=PolicyStatus.NO_MATCH,
                effective_action=PolicyEffectiveAction.NONE,
                matched_policies=[],
                effective_policy=None,
                decisive_reason="no_plan_steps",
                reasons=["no_plan_steps"],
                warnings=[],
                fallback_applied=False,
                sandbox_scope="",
                token_validation=None,
                token_conversion_applied=False,
                rule_precedence_trace=[],
            )
            result = self._build_result(
                status=PolicyStatus.NO_MATCH,
                summary="Policy facet found no plan steps to evaluate.",
                matched_policies=[],
                reasons=["no_plan_steps"],
                context=self._build_context(event, state),
            )
            result.metadata["policy_evaluation"] = policy_eval
            result.metadata["plan_policy_evaluation"] = policy_eval
            result.metadata["step_policy_evaluations"] = {}
            result.metadata["denied_step_ids"] = []
            result.metadata["approval_required_step_ids"] = []
            result.metadata["allowed_step_ids"] = []
            result.metadata["preferred_step_ids"] = []
            return result

        step_policy_evaluations: dict[str, Any] = {}
        denied_step_ids: list[str] = []
        approval_required_step_ids: list[str] = []
        allowed_step_ids: list[str] = []
        preferred_step_ids: list[str] = []

        # Decide which step is decisive for plan-level reporting.
        decisive_step_id: str | None = None
        decisive_step_eval: dict[str, Any] | None = None

        plan_reasons: list[str] = []
        plan_warnings: list[str] = []

        for step_raw in steps_raw:
            if not isinstance(step_raw, dict):
                continue
            step_id = str(step_raw.get("id") or "")
            step_meta = step_raw.get("metadata") if isinstance(step_raw.get("metadata"), dict) else {}
            step_eval = step_meta.get("policy_evaluation")
            if not isinstance(step_eval, dict):
                step_eval = {}
            step_status = str(
                step_eval.get("status") or step_raw.get("policy_status") or ""
            ).strip().lower()
            if not step_status:
                step_status = "no_match"

            # Conservative plan aggregation for step `no_match`.
            if step_status == PolicyStatus.NO_MATCH.value:
                target_type = str(step_raw.get("target_type") or "").strip().lower()
                # Built-in fallback: only external side-effect target types
                # conservatively become approval-required. Everything else is
                # treated as safe/allowed.
                external_side_effect_target_types = {
                    t.value for t in EXTERNAL_APPROVAL_RULES.keys()
                }
                if target_type == PolicyTargetType.INTERNAL_STATE.value:
                    step_status = PolicyStatus.ALLOWED.value
                    step_eval.setdefault("status", PolicyStatus.ALLOWED.value)
                    step_eval.setdefault(
                        "effective_action", PolicyEffectiveAction.ALLOW.value
                    )
                elif target_type in external_side_effect_target_types:
                    step_status = PolicyStatus.APPROVAL_REQUIRED.value
                    step_eval.setdefault(
                        "status", PolicyStatus.APPROVAL_REQUIRED.value
                    )
                    step_eval.setdefault(
                        "effective_action",
                        PolicyEffectiveAction.REQUIRE_APPROVAL.value,
                    )
                else:
                    step_status = PolicyStatus.ALLOWED.value
                    step_eval.setdefault("status", PolicyStatus.ALLOWED.value)
                    step_eval.setdefault(
                        "effective_action", PolicyEffectiveAction.ALLOW.value
                    )

            step_policy_evaluations[step_id] = step_eval

            if step_status == PolicyStatus.DENIED.value:
                denied_step_ids.append(step_id)
                if decisive_step_id is None:
                    decisive_step_id = step_id
                    decisive_step_eval = step_eval
            elif step_status == PolicyStatus.APPROVAL_REQUIRED.value:
                approval_required_step_ids.append(step_id)
                if decisive_step_id is None:
                    decisive_step_id = step_id
                    decisive_step_eval = step_eval
            elif step_status == PolicyStatus.PREFERRED.value:
                preferred_step_ids.append(step_id)
                if decisive_step_id is None:
                    decisive_step_id = step_id
                    decisive_step_eval = step_eval
            elif step_status == PolicyStatus.ALLOWED.value:
                allowed_step_ids.append(step_id)
                if decisive_step_id is None:
                    decisive_step_id = step_id
                    decisive_step_eval = step_eval

            # Aggregate top-level reasons/warnings when available.
            if isinstance(step_eval.get("reasons"), list):
                plan_reasons.extend([str(r) for r in step_eval["reasons"]])
            if isinstance(step_eval.get("warnings"), list):
                plan_warnings.extend([str(w) for w in step_eval["warnings"]])

        if denied_step_ids:
            plan_status = PolicyStatus.DENIED.value
            proposed_decision = DecisionAction.RECORD
        elif approval_required_step_ids:
            plan_status = PolicyStatus.APPROVAL_REQUIRED.value
            proposed_decision = DecisionAction.ASK
        else:
            # Prefer if any step is preferred and none are allowed.
            plan_status = (
                PolicyStatus.PREFERRED.value
                if preferred_step_ids and not allowed_step_ids
                else PolicyStatus.ALLOWED.value
            )
            proposed_decision = None

        # Pick a decisive step evaluation that matches the final aggregate.
        if plan_status == PolicyStatus.DENIED.value and denied_step_ids:
            decisive_step_id = denied_step_ids[0]
            decisive_step_eval = step_policy_evaluations.get(decisive_step_id)
        elif (
            plan_status == PolicyStatus.APPROVAL_REQUIRED.value
            and approval_required_step_ids
        ):
            decisive_step_id = approval_required_step_ids[0]
            decisive_step_eval = step_policy_evaluations.get(decisive_step_id)
        elif plan_status == PolicyStatus.PREFERRED.value and preferred_step_ids:
            decisive_step_id = preferred_step_ids[0]
            decisive_step_eval = step_policy_evaluations.get(decisive_step_id)
        elif plan_status == PolicyStatus.ALLOWED.value and allowed_step_ids:
            decisive_step_id = allowed_step_ids[0]
            decisive_step_eval = step_policy_evaluations.get(decisive_step_id)

        if not isinstance(decisive_step_eval, dict):
            decisive_step_eval = {}
        # Ensure plan-level status/effective_action match the aggregate outcome.
        plan_policy_evaluation = dict(decisive_step_eval)
        plan_policy_evaluation["status"] = plan_status
        if plan_status == PolicyStatus.DENIED.value:
            plan_policy_evaluation["effective_action"] = PolicyEffectiveAction.DENY.value
        elif plan_status == PolicyStatus.APPROVAL_REQUIRED.value:
            plan_policy_evaluation["effective_action"] = (
                PolicyEffectiveAction.REQUIRE_APPROVAL.value
            )
        elif plan_status == PolicyStatus.PREFERRED.value:
            plan_policy_evaluation["effective_action"] = PolicyEffectiveAction.PREFER.value
        elif plan_status == PolicyStatus.ALLOWED.value:
            plan_policy_evaluation["effective_action"] = PolicyEffectiveAction.ALLOW.value

        policy_status = plan_status
        policy_reason = str(
            plan_policy_evaluation.get("decisive_reason") or ""
        ).strip()

        context = self._build_context(event, state)
        result = self._build_result(
            status=PolicyStatus(policy_status),
            summary="Policy facet aggregated plan step policy outcomes (v1).",
            matched_policies=[],
            reasons=list(set(plan_reasons)) or ["plan_policy_aggregated"],
            context=context,
            proposed_decision=proposed_decision,
            effective_policy=None,
        )

        result.metadata["policy_evaluation"] = plan_policy_evaluation
        result.metadata["plan_policy_evaluation"] = plan_policy_evaluation
        result.metadata["step_policy_evaluations"] = step_policy_evaluations
        result.metadata["denied_step_ids"] = denied_step_ids
        result.metadata["approval_required_step_ids"] = approval_required_step_ids
        result.metadata["allowed_step_ids"] = allowed_step_ids
        result.metadata["preferred_step_ids"] = preferred_step_ids
        result.metadata["warnings"] = list(set(plan_warnings))
        return result

    def _build_policy_evaluation_v1_from_outcome(
        self,
        *,
        context: _PolicyContext,
        evaluation_scope: PolicyApprovalScope,
        status: PolicyStatus,
        effective_action: PolicyEffectiveAction,
        matched_policies: list[PolicyRule],
        effective_policy: PolicyRule | None,
        decisive_reason: str,
        reasons: list[str],
        warnings: list[str],
        fallback_applied: bool,
        sandbox_scope: str,
        token_validation: dict[str, Any] | None,
        token_conversion_applied: bool,
        rule_precedence_trace: list[PolicyPrecedenceTraceEntry],
    ) -> dict[str, Any]:
        token_valid = bool(token_validation and token_validation.get("valid") is True)
        approval_token_valid = bool(token_conversion_applied and token_valid)
        approval_token_required = status == PolicyStatus.APPROVAL_REQUIRED and not token_conversion_applied

        approval_scope = (
            str(token_validation.get("scope"))
            if token_validation and token_validation.get("scope") is not None
            else evaluation_scope.value
        )

        decisive_rule_id = effective_policy.id if effective_policy is not None else None
        decisive_rule_name = effective_policy.name if effective_policy is not None else None

        evaluation = PolicyEvaluation(
            status=status.value,
            effective_action=effective_action.value,
            target_type=context.raw_target_type.value
            if context.raw_target_type is not None
            else None,
            target=context.target or None,
            risk_level=context.risk_level.value,
            matched_rule_ids=[r.id for r in matched_policies],
            matched_rule_names=[r.name for r in matched_policies],
            decisive_rule_id=decisive_rule_id,
            decisive_rule_name=decisive_rule_name,
            decisive_reason=str(decisive_reason or ""),
            approval_required=status == PolicyStatus.APPROVAL_REQUIRED,
            approval_token_required=bool(approval_token_required),
            approval_token_valid=bool(approval_token_valid),
            approval_scope=approval_scope,
            rule_precedence_trace=rule_precedence_trace,
            fallback_applied=bool(fallback_applied),
            sandbox_scope=str(sandbox_scope or ""),
            reasons=list(reasons or []),
            warnings=list(warnings or []),
        )
        return evaluation.to_dict()

    def _build_rule_precedence_trace_v1(
        self,
        *,
        context: _PolicyContext,
        matched_policies: list[PolicyRule],
        effective_policy: PolicyRule | None,
        token_conversion_applied: bool,
    ) -> list[PolicyPrecedenceTraceEntry]:
        decisive_id = effective_policy.id if effective_policy is not None else None
        out: list[PolicyPrecedenceTraceEntry] = []
        for rule in matched_policies:
            target_match, target_info = self._target_matches_v1_info(rule, context)
            conditions_matched = (
                []
                if rule.metadata.get("built_in")
                else list(rule.conditions.keys())
            )
            # Effect reflects the *effective* outcome when tokens convert
            # require_approval into allow.
            effect: str
            if rule.rule_type == PolicyRuleType.DENY:
                effect = PolicyEffectiveAction.DENY.value
            elif rule.rule_type == PolicyRuleType.REQUIRE_APPROVAL:
                if (
                    token_conversion_applied
                    and effective_policy is not None
                    and rule.id == effective_policy.id
                ):
                    effect = PolicyEffectiveAction.ALLOW.value
                else:
                    effect = PolicyEffectiveAction.REQUIRE_APPROVAL.value
            elif rule.rule_type == PolicyRuleType.ALLOW:
                effect = PolicyEffectiveAction.ALLOW.value
            elif rule.rule_type == PolicyRuleType.PREFER:
                effect = PolicyEffectiveAction.PREFER.value
            else:
                effect = PolicyEffectiveAction.NONE.value

            out.append(
                PolicyPrecedenceTraceEntry(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.rule_type.value,
                    priority=rule.priority,
                    match_score=target_info.get("match_score"),
                    match_reason=(
                        target_info.get("match_reason")
                        if not rule.metadata.get("built_in")
                        else "built_in_fallback"
                    ),
                    conditions_matched=conditions_matched,
                    effect=effect,
                    decisive=bool(decisive_id and rule.id == decisive_id),
                )
            )
        return out

    def _build_context(self, event: Event, state: NexusState) -> _PolicyContext:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        raw_target_type = coerce_policy_target_type(metadata.get("target_type"))
        target = str(metadata.get("target", "") or "").strip()
        path = self._coerce_string(metadata.get("path"))
        current_decision = self._extract_current_decision(state)
        explicit_tags = normalize_tags(metadata.get("tags", []))
        tags = merge_tags(explicit_tags, infer_tags(event.content))
        operation = self._coerce_string(metadata.get("operation")) or ""
        resolved_path = self._resolve_path(
            path,
            raw_target_type=raw_target_type,
        )
        within_state_dir = self._is_within_state_dir(resolved_path)
        is_internal_state_action = self._is_internal_state_action(
            raw_target_type=raw_target_type,
            target=target,
            within_state_dir=within_state_dir,
        )

        action_type = (
            str(metadata.get("action_type")).strip()
            if isinstance(metadata.get("action_type"), str)
            else None
        )
        source_facet = (
            str(metadata.get("source_facet")).strip()
            if isinstance(metadata.get("source_facet"), str)
            else None
        )
        plan_id = (
            str(metadata.get("plan_id")).strip()
            if isinstance(metadata.get("plan_id"), str)
            else None
        )
        plan_step_id = (
            str(metadata.get("plan_step_id")).strip()
            if isinstance(metadata.get("plan_step_id"), str)
            else None
        )
        skill_name = (
            str(metadata.get("skill_name")).strip()
            if isinstance(metadata.get("skill_name"), str)
            else None
        )

        raw_risk_level = metadata.get("risk_level")
        if isinstance(raw_risk_level, str):
            try:
                risk_level = PolicyRiskLevel(raw_risk_level.strip().lower())
            except ValueError:
                risk_level = PolicyRiskLevel.UNKNOWN
        else:
            # v0 callers use low_risk boolean; map to low when true, unknown otherwise.
            risk_level = (
                PolicyRiskLevel.LOW
                if self._metadata_flag(metadata, "low_risk")
                else PolicyRiskLevel.UNKNOWN
            )

        # Default to dry-run unless explicitly set to false.
        dry_run = metadata.get("dry_run")
        if isinstance(dry_run, bool):
            dry_run_flag = dry_run
        elif dry_run is None:
            dry_run_flag = True
        else:
            dry_run_flag = True
        live_mode = not dry_run_flag

        external_side_effect = (
            raw_target_type
            in {
                PolicyTargetType.FILE_WRITE,
                PolicyTargetType.FILE_DELETE,
                PolicyTargetType.SHELL,
                PolicyTargetType.NETWORK,
                PolicyTargetType.MESSAGE,
                PolicyTargetType.GIT,
                PolicyTargetType.TOOL,
            }
        )

        approval_token = metadata.get("approval")
        if not isinstance(approval_token, dict):
            approval_token = None

        return _PolicyContext(
            metadata=metadata,
            explicit_action=self._metadata_flag(metadata, "explicit_action"),
            current_decision=current_decision,
            raw_target_type=raw_target_type,
            target=target,
            operation=operation,
            path=path,
            resolved_path=resolved_path,
            within_state_dir=within_state_dir,
            is_internal_state_action=is_internal_state_action,
            tags=tags,
            state_dir=self.state_dir,
            action_type=action_type,
            risk_level=risk_level,
            source_facet=source_facet,
            plan_id=plan_id,
            plan_step_id=plan_step_id,
            skill_name=skill_name,
            dry_run=dry_run_flag,
            live_mode=live_mode,
            external_side_effect=bool(external_side_effect),
            approval_token=approval_token,
        )

    def _rule_matches(self, rule: PolicyRule, context: _PolicyContext) -> bool:
        if not self._target_type_matches(rule.target_type, context):
            return False
        if not self._target_matches(rule, context):
            return False
        return self._conditions_match(rule.conditions, context)

    @staticmethod
    def _sort_rules(rules: list[PolicyRule]) -> list[PolicyRule]:
        return sorted(
            rules,
            key=lambda rule: (
                rule.priority,
                rule.updated_at.timestamp(),
                rule.created_at.timestamp(),
                rule.id,
            ),
            reverse=True,
        )

    @staticmethod
    def _top_rule(
        rules: list[PolicyRule],
        rule_type: PolicyRuleType,
    ) -> PolicyRule | None:
        for rule in rules:
            if rule.rule_type == rule_type:
                return rule
        return None

    @staticmethod
    def _rules_of_type(
        rules: list[PolicyRule],
        rule_type: PolicyRuleType,
    ) -> list[PolicyRule]:
        return [rule for rule in rules if rule.rule_type == rule_type]

    def _baseline_status_for_context(
        self,
        context: _PolicyContext,
    ) -> PolicyStatus:
        if context.is_internal_state_action:
            if context.live_mode and context.risk_level == PolicyRiskLevel.HIGH:
                return PolicyStatus.APPROVAL_REQUIRED
            return PolicyStatus.ALLOWED
        if context.raw_target_type in EXTERNAL_APPROVAL_RULES:
            return PolicyStatus.APPROVAL_REQUIRED
        return PolicyStatus.NO_MATCH

    def _baseline_rule_for_context(
        self,
        context: _PolicyContext,
    ) -> PolicyRule | None:
        if context.is_internal_state_action:
            if context.live_mode and context.risk_level == PolicyRiskLevel.HIGH:
                return INTERNAL_STATE_LIVE_HIGH_RISK_REQUIRE_APPROVAL_RULE
            return INTERNAL_STATE_ALLOW_RULE
        if context.raw_target_type in EXTERNAL_APPROVAL_RULES:
            return EXTERNAL_APPROVAL_RULES[context.raw_target_type]
        return None

    def _target_type_matches(
        self,
        target_type: PolicyTargetType,
        context: _PolicyContext,
    ) -> bool:
        if target_type == PolicyTargetType.INTERNAL_STATE:
            return context.is_internal_state_action
        if target_type == PolicyTargetType.GENERAL:
            return True
        if target_type == PolicyTargetType.DECISION:
            return context.current_decision is not None
        if target_type == PolicyTargetType.TAG:
            return bool(context.tags)
        return context.raw_target_type == target_type

    def _target_matches(self, rule: PolicyRule, context: _PolicyContext) -> bool:
        matched, _ = self._target_matches_v1_info(rule, context)
        return matched

    def _target_matches_v1_info(
        self,
        rule: PolicyRule,
        context: _PolicyContext,
    ) -> tuple[bool, dict[str, Any]]:
        candidates = self._target_candidates(rule.target_type, context)

        if not candidates:
            return (
                rule.target == "*",
                {
                    "match_score": 0.5 if rule.target == "*" else None,
                    "match_reason": "target_wildcard_no_candidates"
                    if rule.target == "*"
                    else None,
                },
            )

        pattern = str(rule.target or "").strip()
        if not pattern:
            return False, {}
        if pattern == "*":
            return True, {"match_score": 0.5, "match_reason": "target_wildcard"}

        pattern_cf = pattern.casefold()
        # Exact match.
        for candidate in candidates:
            if candidate.casefold() == pattern_cf:
                return True, {"match_score": 1.0, "match_reason": "target_exact"}

        # Simple prefix match for sandbox-ish strings.
        for candidate in candidates:
            if (
                "/" in candidate
                or "\\" in candidate
                or "/" in pattern
                or "\\" in pattern
            ):
                if candidate.casefold().startswith(pattern_cf):
                    return (
                        True,
                        {
                            "match_score": 0.8,
                            "match_reason": "target_prefix_path",
                        },
                    )

        return False, {}

    def _target_candidates(
        self,
        target_type: PolicyTargetType,
        context: _PolicyContext,
    ) -> list[str]:
        candidates: list[str] = []

        if target_type == PolicyTargetType.INTERNAL_STATE:
            candidates.append("state-dir")
            if context.target:
                candidates.append(context.target)
            if context.path:
                candidates.append(context.path)
            if context.resolved_path is not None:
                candidates.append(str(context.resolved_path))
            return self._dedupe(candidates)

        if target_type == PolicyTargetType.DECISION:
            if context.current_decision is not None:
                candidates.append(context.current_decision.value)
            return candidates

        if target_type == PolicyTargetType.TAG:
            return list(context.tags)

        if target_type == PolicyTargetType.GENERAL:
            if context.raw_target_type is not None:
                candidates.append(context.raw_target_type.value)
            if context.target:
                candidates.append(context.target)
            if context.path:
                candidates.append(context.path)
            if context.current_decision is not None:
                candidates.append(context.current_decision.value)
            candidates.extend(context.tags)
            if context.is_internal_state_action:
                candidates.append("state-dir")
            return self._dedupe(candidates)

        if context.target:
            candidates.append(context.target)
        if context.path:
            candidates.append(context.path)
        if context.resolved_path is not None:
            candidates.append(str(context.resolved_path))
        if not candidates and context.raw_target_type is not None:
            candidates.append(context.raw_target_type.value)
        return self._dedupe(candidates)

    def _conditions_match(
        self,
        conditions: dict[str, Any],
        context: _PolicyContext,
    ) -> bool:
        if not conditions:
            return True

        context_map = {
            "explicit_action": context.explicit_action,
            "current_decision": (
                context.current_decision.value
                if context.current_decision is not None
                else None
            ),
            "target_type": (
                context.raw_target_type.value if context.raw_target_type else None
            ),
            "target": context.target or None,
            "path": context.path,
            "within_state_dir": context.within_state_dir,
            "is_internal_state_action": context.is_internal_state_action,
            "operation": context.operation or None,
            "tags": list(context.tags),
            # v1 structured action context keys
            "action_type": context.action_type,
            "risk_level": context.risk_level.value,
            "source_facet": context.source_facet,
            "plan_id": context.plan_id,
            "plan_step_id": context.plan_step_id,
            "skill_name": context.skill_name,
            "dry_run": context.dry_run,
            "live_mode": context.live_mode,
            "external_side_effect": context.external_side_effect,
            "state_dir": str(self.state_dir),
            "sandbox_root": str(self.state_dir),
        }
        for key, expected in conditions.items():
            actual = context_map.get(key)
            if isinstance(expected, (list, tuple, set, frozenset)):
                if isinstance(actual, list):
                    actual_values = {str(item).casefold() for item in actual}
                    expected_values = {str(item).casefold() for item in expected}
                    if not actual_values & expected_values:
                        return False
                    continue
                if actual not in expected:
                    return False
                continue
            if isinstance(expected, str) and isinstance(actual, str):
                if actual.casefold() != expected.casefold():
                    return False
                continue
            if actual != expected:
                return False
        return True

    def _build_result(
        self,
        *,
        status: PolicyStatus,
        summary: str,
        matched_policies: list[PolicyRule],
        reasons: list[str],
        context: _PolicyContext,
        proposed_decision: DecisionAction | None = None,
        effective_policy: PolicyRule | None = None,
        baseline_status: PolicyStatus | None = None,
    ) -> FacetResult:
        matched_payload = [self._describe_policy(rule) for rule in matched_policies]
        effective_payload = (
            self._describe_policy(effective_policy) if effective_policy is not None else None
        )
        metadata = {
            "policy_status": status.value,
            "matched_policies": matched_payload,
            "matched_policy_ids": [policy["id"] for policy in matched_payload],
            "matched_policy_names": [policy["name"] for policy in matched_payload],
            "reasons": list(reasons),
            "current_decision": (
                context.current_decision.value
                if context.current_decision is not None
                else None
            ),
            "target_type": (
                context.raw_target_type.value if context.raw_target_type is not None else None
            ),
            "target": context.target or None,
            "path": context.path,
            "within_state_dir": context.within_state_dir,
            "is_internal_state_action": context.is_internal_state_action,
            "explicit_action": context.explicit_action,
            "operation": context.operation or None,
            "tags": list(context.tags),
            "approval_required": status == PolicyStatus.APPROVAL_REQUIRED,
            "denied": status == PolicyStatus.DENIED,
            "effective_policy": effective_payload,
        }
        if baseline_status is not None:
            metadata["baseline_status"] = baseline_status.value

        return FacetResult(
            facet_name=self.name,
            summary=summary,
            proposed_decision=proposed_decision,
            state_updates={
                "last_policy_status": status.value,
                "last_matched_policy_ids": [policy["id"] for policy in matched_payload],
                "last_effective_policy_id": (
                    effective_payload["id"] if effective_payload is not None else None
                ),
                "last_approval_required": status == PolicyStatus.APPROVAL_REQUIRED,
                "last_denied": status == PolicyStatus.DENIED,
            },
            metadata=metadata,
        )

    @staticmethod
    def _describe_policy(rule: PolicyRule) -> dict[str, object]:
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type.value,
            "target_type": rule.target_type.value,
            "target": rule.target,
            "priority": rule.priority,
            "enabled": rule.enabled,
            "source": rule.source.value,
            "built_in": bool(rule.metadata.get("built_in")),
        }

    def _extract_current_decision(self, state: NexusState) -> DecisionAction | None:
        behavior_state = state.facet_state.get("behavior")
        if not isinstance(behavior_state, dict):
            return None
        raw_decision = behavior_state.get("last_selected_decision")
        if not isinstance(raw_decision, str):
            return None
        try:
            return DecisionAction(raw_decision)
        except ValueError:
            return None

    def _resolve_path(
        self,
        raw_path: str | None,
        *,
        raw_target_type: PolicyTargetType | None,
    ) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute() and raw_target_type == PolicyTargetType.INTERNAL_STATE:
            candidate = self.state_dir / candidate
        return candidate.resolve()

    def _is_within_state_dir(self, path: Path | None) -> bool:
        if path is None:
            return False
        try:
            path.relative_to(self.state_dir)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_internal_state_action(
        *,
        raw_target_type: PolicyTargetType | None,
        target: str,
        within_state_dir: bool,
    ) -> bool:
        if raw_target_type == PolicyTargetType.INTERNAL_STATE:
            return True
        if target.strip().casefold() == "state-dir":
            return True
        return within_state_dir

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

    def _validate_approval_token(
        self, context: _PolicyContext, *, evaluation_scope: str
    ) -> dict[str, Any]:
        """Validate explicit approval metadata locally (no persistence)."""
        token = context.approval_token
        if token is None:
            return {
                "valid": False,
                "scope": evaluation_scope,
                "warnings": ["missing_approval_token"],
            }
        if not isinstance(token.get("approved"), bool) or token.get("approved") is not True:
            return {
                "valid": False,
                "scope": str(token.get("scope") or evaluation_scope),
                "warnings": ["approval_token_not_approved_true"],
            }

        warnings: list[str] = []
        expires_at_raw = token.get("expires_at")
        if isinstance(expires_at_raw, str) and expires_at_raw.strip():
            try:
                exp = datetime.fromisoformat(expires_at_raw.strip())
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp:
                    warnings.append("approval_token_expired")
            except Exception:  # noqa: BLE001 deliberate deterministic parsing warning
                warnings.append("approval_token_expires_at_invalid")

        token_scope = token.get("scope")
        token_scope_clean = (
            str(token_scope).strip().lower() if isinstance(token_scope, str) else ""
        )
        evaluation_scope_clean = evaluation_scope.strip().lower()

        # Accept matching scope, session-wide scope, or wildcard scope.
        scope_valid = token_scope_clean in {
            evaluation_scope_clean,
            PolicyApprovalScope.SESSION.value,
            "*",
        }

        # Determine matching target/type requirements.
        token_target_type = token.get("target_type")
        token_target = token.get("target")
        ctx_target_type = context.raw_target_type.value if context.raw_target_type else None
        ctx_target = context.target or ""

        type_valid = (
            isinstance(token_target_type, str)
            and (
                token_target_type.strip() == "*"
                or (ctx_target_type is not None and token_target_type.strip() == ctx_target_type)
            )
        )
        target_valid = (
            isinstance(token_target, str)
            and (
                token_target.strip() == "*"
                or token_target.strip().casefold() == ctx_target.casefold()
            )
        )

        valid = (not warnings) and scope_valid and type_valid and target_valid
        if not valid and not warnings:
            warnings.append("approval_token_scope_or_target_mismatch")

        return {
            "valid": bool(valid),
            "scope": token_scope_clean or evaluation_scope_clean,
            "warnings": warnings,
        }

    @staticmethod
    def _coerce_string(raw_value: Any) -> str | None:
        if not isinstance(raw_value, str):
            return None
        cleaned = raw_value.strip()
        return cleaned or None

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            deduped.append(cleaned)
            seen.add(key)
        return deduped
