"""Minimal Nexus runtime loop."""

from __future__ import annotations

from typing import Iterable

from fullerene.facets.base import Facet
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusState,
)
from fullerene.policy.models import PolicyStatus
from fullerene.state.store import InMemoryStateStore, StateStore

# Higher score wins when multiple facets explicitly propose a decision.
# ACT > ASK > RECORD > WAIT.
DECISION_PRIORITY = {
    DecisionAction.WAIT: 0,
    DecisionAction.RECORD: 1,
    DecisionAction.ASK: 2,
    DecisionAction.ACT: 3,
}


class Nexus:
    """Central interpreter/integrator loop for Fullerene v0."""

    def __init__(
        self,
        facets: Iterable[Facet] | None = None,
        store: StateStore | None = None,
        initial_state: NexusState | None = None,
    ) -> None:
        self._store = store or InMemoryStateStore()
        self._facets: list[Facet] = list(facets or [])
        self.state = initial_state or self._store.load_state() or NexusState()

    @property
    def facets(self) -> tuple[Facet, ...]:
        return tuple(self._facets)

    def register_facet(self, facet: Facet) -> None:
        self._facets.append(facet)

    def process_event(self, event: Event) -> NexusRecord:
        working_state = NexusState.from_dict(self.state.to_dict())
        facet_results: list[FacetResult] = []
        verifier_facets: list[Facet] = []
        for facet in self._facets:
            if self._is_post_decision_verifier(facet):
                verifier_facets.append(facet)
                continue
            result = self._run_facet(facet, event, working_state)
            facet_results.append(result)
            self._apply_result_to_state(working_state, result)
        decision = self._integrate(event, facet_results)
        for verifier in verifier_facets:
            verifier_result = self._run_verifier(
                verifier,
                event,
                working_state,
                facet_results,
                decision,
            )
            facet_results.append(verifier_result)
            self._apply_result_to_state(working_state, verifier_result)
            decision = self._apply_verifier_decision(decision, verifier_result)
        self.state.apply(event, facet_results, decision)

        record = NexusRecord(
            event=event,
            facet_results=facet_results,
            decision=decision,
        )
        self._store.save_state(self.state)
        self._store.append_record(record)
        return record

    def _run_facet(
        self,
        facet: Facet,
        event: Event,
        state: NexusState,
    ) -> FacetResult:
        try:
            return facet.process(event, state)
        except Exception as exc:
            facet_name = self._facet_name(facet)
            error_message = str(exc) or "Facet raised without an error message."
            return FacetResult(
                facet_name=facet_name,
                summary=(
                    f"Facet '{facet_name}' failed while processing the event: "
                    f"{error_message}"
                ),
                proposed_decision=DecisionAction.RECORD,
                metadata={
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message,
                },
            )

    @staticmethod
    def _apply_result_to_state(state: NexusState, result: FacetResult) -> None:
        if not result.state_updates:
            return
        facet_bucket = state.facet_state.setdefault(result.facet_name, {})
        facet_bucket.update(result.state_updates)

    def _facet_name(self, facet: Facet) -> str:
        raw_name = getattr(facet, "name", "") or facet.__class__.__name__
        return str(raw_name)

    @staticmethod
    def _is_post_decision_verifier(facet: Facet) -> bool:
        return (
            callable(getattr(facet, "verify", None))
            and str(getattr(facet, "name", "") or "").strip().casefold() == "verifier"
        )

    def _run_verifier(
        self,
        facet: Facet,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
        decision: NexusDecision,
    ) -> FacetResult:
        verify = getattr(facet, "verify", None)
        if not callable(verify):
            return self._run_facet(facet, event, state)
        try:
            return verify(event, state, list(facet_results), decision)
        except Exception as exc:
            facet_name = self._facet_name(facet)
            error_message = str(exc) or "Verifier raised without an error message."
            return FacetResult(
                facet_name=facet_name,
                summary=(
                    f"Verifier '{facet_name}' failed while validating the decision: "
                    f"{error_message}"
                ),
                proposed_decision=DecisionAction.RECORD,
                metadata={
                    "verification_status": "failed",
                    "failed_checks": ["verifier_runtime_error"],
                    "warnings": [],
                    "results": [
                        {
                            "check_name": "verifier_runtime_error",
                            "status": "failed",
                            "severity": "critical",
                            "message": error_message,
                            "metadata": {
                                "recommended_action": DecisionAction.RECORD.value
                            },
                        }
                    ],
                    "reasons": [error_message],
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message,
                },
            )

    @staticmethod
    def _apply_verifier_decision(
        decision: NexusDecision,
        verifier_result: FacetResult,
    ) -> NexusDecision:
        metadata = (
            verifier_result.metadata if isinstance(verifier_result.metadata, dict) else {}
        )
        if metadata.get("verification_status") != "failed":
            metadata["override_applied"] = False
            metadata["override_reason"] = "verification_did_not_fail"
            return decision
        proposed_decision = verifier_result.proposed_decision
        if proposed_decision is None:
            metadata["override_applied"] = False
            metadata["override_reason"] = "no_verifier_proposal"
            return decision
        metadata["current_decision"] = decision.action.value
        metadata["proposed_override_decision"] = proposed_decision.value

        current_priority = DECISION_PRIORITY[decision.action]
        proposed_priority = DECISION_PRIORITY[proposed_decision]
        if proposed_priority > current_priority:
            metadata["override_applied"] = False
            metadata["override_reason"] = "ignored_higher_priority_verifier_proposal"
            return decision
        if proposed_priority == current_priority:
            metadata["override_applied"] = False
            if proposed_decision == decision.action:
                metadata["override_reason"] = "proposed_decision_matches_current"
            else:
                metadata["override_reason"] = "ignored_same_priority_verifier_proposal"
            return decision

        metadata["override_applied"] = True
        metadata["override_reason"] = "risk_reducing_downgrade"
        source_facets = list(decision.source_facets)
        if verifier_result.facet_name not in source_facets:
            source_facets.append(verifier_result.facet_name)
        return NexusDecision(
            action=proposed_decision,
            reason=(
                f"Verifier downgraded {decision.action.value.upper()} to "
                f"{proposed_decision.value.upper()}: {verifier_result.summary}"
            ),
            source_facets=source_facets,
        )

    def _integrate(
        self,
        event: Event,
        facet_results: list[FacetResult],
    ) -> NexusDecision:
        denied_policy_results = self._policy_results(
            facet_results,
            status=PolicyStatus.DENIED,
        )
        if denied_policy_results:
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason=self._policy_reason(
                    denied_policy_results,
                    default="Selected RECORD because policy denied the modeled action.",
                ),
                source_facets=[result.facet_name for result in denied_policy_results],
            )

        approval_policy_results = self._policy_results(
            facet_results,
            status=PolicyStatus.APPROVAL_REQUIRED,
        )
        if approval_policy_results:
            return NexusDecision(
                action=DecisionAction.ASK,
                reason=self._policy_reason(
                    approval_policy_results,
                    default=(
                        "Selected ASK because policy requires approval before the "
                        "modeled action."
                    ),
                ),
                source_facets=[result.facet_name for result in approval_policy_results],
            )

        explicit_results = [
            result for result in facet_results if result.proposed_decision is not None
        ]
        if explicit_results:
            selected_action = max(
                (result.proposed_decision for result in explicit_results),
                key=lambda action: DECISION_PRIORITY[action],
            )
            source_facets = [
                result.facet_name
                for result in explicit_results
                if result.proposed_decision == selected_action
            ]
            reason = (
                f"Selected {selected_action.value.upper()} from facet proposals: "
                f"{', '.join(source_facets)}."
            )
            return NexusDecision(
                action=selected_action,
                reason=reason,
                source_facets=source_facets,
            )

        if event.event_type == EventType.USER_MESSAGE:
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD for a user message event.",
            )

        if any(result.state_updates for result in facet_results):
            return NexusDecision(
                action=DecisionAction.RECORD,
                reason="Defaulted to RECORD because facets produced state updates.",
            )

        return NexusDecision(
            action=DecisionAction.WAIT,
            reason="Defaulted to WAIT because no facet proposed or updated anything.",
        )

    @staticmethod
    def _policy_results(
        facet_results: list[FacetResult],
        *,
        status: PolicyStatus,
    ) -> list[FacetResult]:
        matches: list[FacetResult] = []
        for result in facet_results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if metadata.get("policy_status") == status.value:
                matches.append(result)
        return matches

    @staticmethod
    def _policy_reason(
        policy_results: list[FacetResult],
        *,
        default: str,
    ) -> str:
        policy_names: list[str] = []
        for result in policy_results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            matched_policies = metadata.get("matched_policies")
            if not isinstance(matched_policies, list):
                continue
            for policy in matched_policies:
                if not isinstance(policy, dict):
                    continue
                name = policy.get("name")
                if isinstance(name, str) and name not in policy_names:
                    policy_names.append(name)
        if not policy_names:
            return default
        return f"{default} Matched policies: {', '.join(policy_names)}."


class NexusRuntime(Nexus):
    """Explicit runtime alias for callers that prefer a runtime-oriented name."""
