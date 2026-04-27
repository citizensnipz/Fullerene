"""Deterministic Learning v0 signal classification."""

from __future__ import annotations

import re
from typing import Any

from fullerene.learning.models import LearningSignal, SignalSource, SignalType
from fullerene.nexus.models import Event, NexusState

POSITIVE_FEEDBACK_PHRASES = (
    "that worked",
    "worked",
    "good",
    "correct",
    "yes",
    "nice",
    "perfect",
)
NEGATIVE_FEEDBACK_PHRASES = (
    "that was wrong",
    "wrong",
    "bad",
    "incorrect",
    "no",
    "failed",
    "not right",
)
APPROVAL_NEUTRAL_REASONS = frozenset({"requires_approval", "blocked_by_policy"})
GOAL_NEGATIVE_STATUSES = frozenset({"stale", "abandoned"})


def classify_user_feedback_signal(event: Event) -> LearningSignal | None:
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    raw_feedback = metadata.get("feedback")
    if isinstance(raw_feedback, str):
        cleaned = raw_feedback.strip().lower()
        if cleaned == "positive":
            return LearningSignal(
                signal_type=SignalType.POSITIVE,
                source=SignalSource.USER_FEEDBACK,
                magnitude=1.0,
                source_event_id=event.event_id,
                metadata={"feedback": cleaned},
                reasons=["explicit_feedback_positive"],
            )
        if cleaned == "negative":
            return LearningSignal(
                signal_type=SignalType.NEGATIVE,
                source=SignalSource.USER_FEEDBACK,
                magnitude=1.0,
                source_event_id=event.event_id,
                metadata={"feedback": cleaned},
                reasons=["explicit_feedback_negative"],
            )

    normalized = _normalize_text(event.content)
    if not normalized:
        return None
    if _matches_phrase(normalized, POSITIVE_FEEDBACK_PHRASES):
        return LearningSignal(
            signal_type=SignalType.POSITIVE,
            source=SignalSource.USER_FEEDBACK,
            magnitude=1.0,
            source_event_id=event.event_id,
            metadata={"content": event.content},
            reasons=["feedback_phrase_positive"],
        )
    if _matches_phrase(normalized, NEGATIVE_FEEDBACK_PHRASES):
        return LearningSignal(
            signal_type=SignalType.NEGATIVE,
            source=SignalSource.USER_FEEDBACK,
            magnitude=1.0,
            source_event_id=event.event_id,
            metadata={"content": event.content},
            reasons=["feedback_phrase_negative"],
        )
    return None


def classify_execution_result_signal(
    event: Event,
    state: NexusState,
) -> LearningSignal | None:
    payload = _extract_execution_payload(event, state)
    if payload is None:
        return None

    status = _normalize_text(payload.get("overall_status"))
    if not status:
        return None

    reasons = payload.get("reasons", [])
    normalized_reasons = [
        _normalize_text(reason)
        for reason in reasons
        if isinstance(reason, str) and _normalize_text(reason)
    ]
    source_record_id = _extract_execution_record_id(payload)
    metadata = {
        "execution_status": status,
        "dry_run": bool(payload.get("dry_run", True)),
        "reasons": list(normalized_reasons),
        "plan_id": payload.get("plan_id"),
    }

    if status == "success":
        return LearningSignal(
            signal_type=SignalType.POSITIVE,
            source=SignalSource.EXECUTION_RESULT,
            magnitude=0.3,
            source_event_id=event.event_id,
            source_record_id=source_record_id,
            metadata=metadata,
            reasons=["execution_status_success"],
        )
    if status == "failed":
        return LearningSignal(
            signal_type=SignalType.FAILURE,
            source=SignalSource.EXECUTION_RESULT,
            magnitude=0.7,
            source_event_id=event.event_id,
            source_record_id=source_record_id,
            metadata=metadata,
            reasons=["execution_status_failed"],
        )
    if status == "skipped":
        signal_type = (
            SignalType.NEUTRAL
            if APPROVAL_NEUTRAL_REASONS & set(normalized_reasons)
            else SignalType.NEGATIVE
        )
        return LearningSignal(
            signal_type=signal_type,
            source=SignalSource.EXECUTION_RESULT,
            magnitude=0.2,
            source_event_id=event.event_id,
            source_record_id=source_record_id,
            metadata=metadata,
            reasons=["execution_status_skipped", *normalized_reasons],
        )
    return None


def classify_goal_lifecycle_signal(event: Event) -> LearningSignal | None:
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    raw_status = metadata.get("goal_status")
    if not isinstance(raw_status, str):
        return None

    status = _normalize_text(raw_status)
    if status == "completed":
        return LearningSignal(
            signal_type=SignalType.POSITIVE,
            source=SignalSource.GOAL_LIFECYCLE,
            magnitude=0.6,
            source_event_id=event.event_id,
            metadata={"goal_status": status},
            reasons=["goal_completed"],
        )
    if status in GOAL_NEGATIVE_STATUSES:
        return LearningSignal(
            signal_type=SignalType.NEGATIVE,
            source=SignalSource.GOAL_LIFECYCLE,
            magnitude=0.3,
            source_event_id=event.event_id,
            metadata={"goal_status": status},
            reasons=[f"goal_{status}"],
        )
    return None


def collect_learning_signals(event: Event, state: NexusState) -> list[LearningSignal]:
    signals: list[LearningSignal] = []
    for classifier in (
        classify_user_feedback_signal,
        lambda current_event: classify_execution_result_signal(current_event, state),
        classify_goal_lifecycle_signal,
    ):
        signal = classifier(event)
        if signal is not None:
            signals.append(signal)
    return signals


def _extract_execution_payload(
    event: Event,
    state: NexusState,
) -> dict[str, Any] | None:
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    event_payload = metadata.get("execution_result")
    if isinstance(event_payload, dict):
        return event_payload

    executor_state = state.facet_state.get("executor")
    if not isinstance(executor_state, dict):
        return None
    state_payload = executor_state.get("last_execution_result")
    return state_payload if isinstance(state_payload, dict) else None


def _extract_execution_record_id(payload: dict[str, Any]) -> str | None:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None
    first_record = records[0]
    if not isinstance(first_record, dict):
        return None
    record_id = first_record.get("id")
    return record_id if isinstance(record_id, str) and record_id.strip() else None


def _matches_phrase(content: str, phrases: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(phrase)}\b", content) for phrase in phrases)


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().strip().split())
