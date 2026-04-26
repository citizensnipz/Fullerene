"""Deterministic policy facet for Fullerene Policy v0."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from fullerene.memory import infer_tags, merge_tags, normalize_tags
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicySource,
    PolicyStatus,
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
        if not context.is_action_candidate:
            return self._build_result(
                status=PolicyStatus.NO_MATCH,
                summary="Policy facet found no modeled action or policy target to evaluate.",
                matched_policies=[],
                reasons=["no_policy_target"],
                context=context,
            )

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
            return self._build_result(
                status=PolicyStatus.DENIED,
                summary=f"Policy facet denied the modeled action via {denied.name!r}.",
                matched_policies=ordered_matches,
                reasons=reasons,
                context=context,
                proposed_decision=DecisionAction.RECORD,
                effective_policy=denied,
            )

        if approval is not None:
            reasons = [
                f"require_approval:{approval.name}",
                "approval_rules_override_allow_and_prefer",
            ]
            return self._build_result(
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

        baseline_rule = self._baseline_rule_for_context(context)
        baseline_status = self._baseline_status_for_context(context)

        if allowed is not None:
            reasons = [f"allow:{allowed.name}"]
            if baseline_status == PolicyStatus.APPROVAL_REQUIRED:
                reasons.append("explicit_allow_overrides_default_external_approval")
            if preferred:
                reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            return self._build_result(
                status=PolicyStatus.ALLOWED,
                summary=f"Policy facet allowed the modeled action via {allowed.name!r}.",
                matched_policies=ordered_matches,
                reasons=reasons,
                context=context,
                effective_policy=allowed,
                baseline_status=baseline_status,
            )

        if baseline_status == PolicyStatus.ALLOWED and baseline_rule is not None:
            baseline_matches = [baseline_rule, *preferred]
            reasons = ["baseline_internal_state_allow"]
            reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            return self._build_result(
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

        if baseline_status == PolicyStatus.APPROVAL_REQUIRED and baseline_rule is not None:
            baseline_matches = [baseline_rule, *preferred]
            reasons = ["baseline_external_approval_required"]
            reasons.extend(f"prefer:{rule.name}" for rule in preferred)
            return self._build_result(
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

        if preferred:
            return self._build_result(
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

        return self._build_result(
            status=PolicyStatus.NO_MATCH,
            summary="Policy facet found no matching policy rules for the modeled action.",
            matched_policies=[],
            reasons=["no_matching_policy_rule"],
            context=context,
        )

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
            return PolicyStatus.ALLOWED
        if context.raw_target_type in EXTERNAL_APPROVAL_RULES:
            return PolicyStatus.APPROVAL_REQUIRED
        return PolicyStatus.NO_MATCH

    def _baseline_rule_for_context(
        self,
        context: _PolicyContext,
    ) -> PolicyRule | None:
        if context.is_internal_state_action:
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
        candidates = self._target_candidates(rule.target_type, context)
        if not candidates:
            return rule.target == "*"
        pattern = rule.target.casefold()
        return any(fnmatch(candidate.casefold(), pattern) for candidate in candidates)

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
