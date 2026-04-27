"""Deterministic Verifier facet for post-decision Nexus checks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fullerene.nexus.models import (
    DecisionAction,
    Event,
    FacetResult,
    NexusDecision,
    NexusState,
)
from fullerene.verifier import (
    DEFAULT_CHECKS,
    VerificationCheck,
    VerificationStatus,
    VerificationSummary,
    run_verification_checks,
)


class VerifierFacet:
    """Run deterministic post-decision verification checks."""

    name = "verifier"

    def __init__(
        self,
        *,
        checks: Sequence[VerificationCheck] | None = None,
        state_dir: Path | str | None = None,
    ) -> None:
        self.checks = tuple(checks or DEFAULT_CHECKS)
        self.state_dir = (
            Path(state_dir).expanduser().resolve() if state_dir is not None else None
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        return FacetResult(
            facet_name=self.name,
            summary="VerifierFacet runs after Nexus aggregates an initial decision.",
            metadata={"post_processing_only": True},
        )

    def verify(
        self,
        event: Event,
        state: NexusState,
        facet_results: list[FacetResult],
        decision: NexusDecision | None,
    ) -> FacetResult:
        summary = run_verification_checks(
            event=event,
            state=state,
            facet_results=facet_results,
            decision=decision,
            checks=self.checks,
            state_dir=self.state_dir,
        )
        proposed_decision = self._proposed_decision(summary)
        reasons = self._reasons(summary)
        initial_action = decision.action.value if decision is not None else None
        initial_reason = decision.reason if decision is not None else None

        return FacetResult(
            facet_name=self.name,
            summary=self._summary_text(summary),
            proposed_decision=proposed_decision,
            state_updates={
                "last_verification_status": summary.overall_status.value,
                "last_failed_checks": list(summary.failed_checks),
                "last_warnings": list(summary.warnings),
                "last_checked_action": initial_action,
            },
            metadata={
                "verification_status": summary.overall_status.value,
                "failed_checks": list(summary.failed_checks),
                "warnings": list(summary.warnings),
                "results": [result.to_dict() for result in summary.results],
                "reasons": reasons,
                "summary_metadata": dict(summary.metadata),
                "initial_decision": initial_action,
                "initial_reason": initial_reason,
                "recommended_decision": (
                    proposed_decision.value if proposed_decision is not None else None
                ),
            },
        )

    @staticmethod
    def _proposed_decision(
        summary: VerificationSummary,
    ) -> DecisionAction | None:
        if summary.overall_status != VerificationStatus.FAILED:
            return None

        recommended_actions: list[DecisionAction] = []
        for result in summary.results:
            if result.status != VerificationStatus.FAILED:
                continue
            raw_action = result.metadata.get("recommended_action")
            if raw_action == DecisionAction.RECORD.value:
                recommended_actions.append(DecisionAction.RECORD)
            elif raw_action == DecisionAction.ASK.value:
                recommended_actions.append(DecisionAction.ASK)

        if DecisionAction.RECORD in recommended_actions:
            return DecisionAction.RECORD
        if DecisionAction.ASK in recommended_actions:
            return DecisionAction.ASK
        return DecisionAction.RECORD

    @staticmethod
    def _reasons(summary: VerificationSummary) -> list[str]:
        reasons = [
            result.message
            for result in summary.results
            if result.status != VerificationStatus.PASSED
        ]
        if reasons:
            return reasons
        return ["All deterministic verifier checks passed."]

    @staticmethod
    def _summary_text(summary: VerificationSummary) -> str:
        if summary.overall_status == VerificationStatus.FAILED:
            return (
                "Verifier failed deterministic checks: "
                + "; ".join(summary.failed_checks)
                + "."
            )
        if summary.overall_status == VerificationStatus.WARNING:
            return (
                "Verifier completed with warnings: "
                + "; ".join(summary.warnings)
                + "."
            )
        return "Verifier passed all deterministic checks."
