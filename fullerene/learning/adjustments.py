"""Learning v0 adjustment generation and safe application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullerene.goals import Goal, GoalStore
from fullerene.learning.models import (
    AdjustmentRecord,
    AdjustmentStatus,
    AdjustmentTarget,
    LearningResult,
    LearningSignal,
    SignalSource,
    SignalType,
)
from fullerene.learning.signals import collect_learning_signals
from fullerene.memory import MemoryStore
from fullerene.nexus.models import Event, NexusState

LEARNING_ALPHA = 0.1
MINOR_NUDGE = 0.05
DEFAULT_PROPOSAL_THRESHOLD = 0.1


@dataclass(slots=True)
class _ResolvedTarget:
    target: AdjustmentTarget
    target_id: str | None
    target_facet: str
    field: str
    old_value: float | None
    store_available: bool
    no_target: bool = False
    unknown_target: bool = False
    current_value_unavailable: bool = False
    metadata: dict[str, Any] | None = None


def build_learning_result(
    event: Event,
    state: NexusState,
    *,
    memory_store: MemoryStore | None = None,
    goal_store: GoalStore | None = None,
    alpha: float = LEARNING_ALPHA,
    minor_nudge: float = MINOR_NUDGE,
    proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD,
) -> LearningResult:
    signals = collect_learning_signals(event, state)
    adjustments = generate_adjustments(
        signals,
        event=event,
        state=state,
        memory_store=memory_store,
        goal_store=goal_store,
        alpha=alpha,
        minor_nudge=minor_nudge,
        proposal_threshold=proposal_threshold,
    )
    proposals = [
        record for record in adjustments if record.status == AdjustmentStatus.PROPOSED
    ]
    applied = [
        record for record in adjustments if record.status == AdjustmentStatus.APPLIED
    ]
    skipped = [
        record for record in adjustments if record.status == AdjustmentStatus.SKIPPED
    ]
    return LearningResult(
        signals=signals,
        adjustments=adjustments,
        proposals=proposals,
        applied=applied,
        overall_status=_overall_status(signals, adjustments),
        metadata={
            "signal_count": len(signals),
            "adjustment_count": len(adjustments),
            "proposal_count": len(proposals),
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "alpha": round(float(alpha), 3),
            "minor_nudge": round(float(minor_nudge), 3),
            "proposal_threshold": round(float(proposal_threshold), 3),
            "skipped": [record.to_dict() for record in skipped],
        },
    )


def generate_adjustments(
    signals: list[LearningSignal],
    *,
    event: Event,
    state: NexusState,
    memory_store: MemoryStore | None = None,
    goal_store: GoalStore | None = None,
    alpha: float = LEARNING_ALPHA,
    minor_nudge: float = MINOR_NUDGE,
    proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD,
) -> list[AdjustmentRecord]:
    adjustments: list[AdjustmentRecord] = []
    for signal in signals:
        direction = _signal_direction(signal.signal_type)
        if direction == 0:
            continue
        resolved_targets = _resolve_targets(
            signal,
            event=event,
            state=state,
            memory_store=memory_store,
            goal_store=goal_store,
        )
        if not resolved_targets:
            adjustments.append(_make_skipped_record(signal))
            continue
        for target in resolved_targets:
            adjustments.append(
                _build_adjustment_record(
                    signal,
                    target=target,
                    direction=direction,
                    alpha=alpha,
                    minor_nudge=minor_nudge,
                    proposal_threshold=proposal_threshold,
                    memory_store=memory_store,
                    goal_store=goal_store,
                )
            )
    return adjustments


def _resolve_targets(
    signal: LearningSignal,
    *,
    event: Event,
    state: NexusState,
    memory_store: MemoryStore | None,
    goal_store: GoalStore | None,
) -> list[_ResolvedTarget]:
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    resolved: list[_ResolvedTarget] = []

    if isinstance(metadata.get("target_memory_id"), str) and metadata["target_memory_id"].strip():
        resolved.append(
            _resolve_memory_target(
                memory_id=metadata["target_memory_id"].strip(),
                memory_store=memory_store,
                metadata=metadata,
            )
        )

    if isinstance(metadata.get("target_goal_id"), str) and metadata["target_goal_id"].strip():
        resolved.append(
            _resolve_goal_target(
                goal_id=metadata["target_goal_id"].strip(),
                goal_store=goal_store,
                metadata=metadata,
            )
        )

    behavior_target = _resolve_behavior_target(signal, metadata=metadata, state=state)
    explicit_behavior_requested = behavior_target is not None and (
        "target_behavior_field" in metadata
        or "adjustment_target" in metadata
        or "behavior_threshold" in metadata
        or "behavior_confidence" in metadata
        or "current_behavior_threshold" in metadata
        or "current_behavior_confidence" in metadata
    )

    if behavior_target is not None and (explicit_behavior_requested or not resolved):
        resolved.append(behavior_target)

    if resolved:
        return resolved

    return [_fallback_target_for_signal(signal)]


def _resolve_memory_target(
    *,
    memory_id: str,
    memory_store: MemoryStore | None,
    metadata: dict[str, Any],
) -> _ResolvedTarget:
    old_value = _numeric_unit_value(metadata.get("current_memory_salience"))
    unknown_target = False
    if memory_store is not None:
        memory = memory_store.get_memory(memory_id)
        if memory is None:
            unknown_target = True
            old_value = None
        else:
            old_value = memory.salience
    return _ResolvedTarget(
        target=AdjustmentTarget.MEMORY_SALIENCE,
        target_id=memory_id,
        target_facet="memory",
        field="salience",
        old_value=old_value,
        store_available=memory_store is not None and hasattr(memory_store, "update_memory_salience"),
        unknown_target=unknown_target,
        current_value_unavailable=old_value is None and not unknown_target,
        metadata={"owner": "memory"},
    )


def _resolve_goal_target(
    *,
    goal_id: str,
    goal_store: GoalStore | None,
    metadata: dict[str, Any],
) -> _ResolvedTarget:
    old_value = _numeric_unit_value(metadata.get("current_goal_priority"))
    unknown_target = False
    if goal_store is not None:
        goal = goal_store.get_goal(goal_id)
        if goal is None:
            unknown_target = True
            old_value = None
        else:
            old_value = goal.priority
    return _ResolvedTarget(
        target=AdjustmentTarget.GOAL_PRIORITY,
        target_id=goal_id,
        target_facet="goals",
        field="priority",
        old_value=old_value,
        store_available=goal_store is not None,
        unknown_target=unknown_target,
        current_value_unavailable=old_value is None and not unknown_target,
        metadata={"owner": "goals"},
    )


def _resolve_behavior_target(
    signal: LearningSignal,
    *,
    metadata: dict[str, Any],
    state: NexusState,
) -> _ResolvedTarget | None:
    explicit_target = _coerce_behavior_target(metadata.get("target_behavior_field"))
    adjustment_target = _coerce_adjustment_target(metadata.get("adjustment_target"))
    if explicit_target is None and adjustment_target in {
        AdjustmentTarget.BEHAVIOR_CONFIDENCE,
        AdjustmentTarget.BEHAVIOR_THRESHOLD,
    }:
        explicit_target = adjustment_target

    if explicit_target == AdjustmentTarget.BEHAVIOR_THRESHOLD:
        old_value = _numeric_unit_value(
            metadata.get("current_behavior_threshold", metadata.get("behavior_threshold"))
        )
        return _ResolvedTarget(
            target=AdjustmentTarget.BEHAVIOR_THRESHOLD,
            target_id=None,
            target_facet="behavior",
            field="threshold",
            old_value=old_value,
            store_available=False,
            current_value_unavailable=old_value is None,
            metadata={"owner": "behavior", "signal_source": signal.source.value},
        )

    behavior_state = state.facet_state.get("behavior")
    old_value = _numeric_unit_value(metadata.get("current_behavior_confidence"))
    if old_value is None and isinstance(behavior_state, dict):
        old_value = _numeric_unit_value(behavior_state.get("last_confidence"))
    if old_value is None:
        old_value = _numeric_unit_value(metadata.get("behavior_confidence"))

    if explicit_target == AdjustmentTarget.BEHAVIOR_CONFIDENCE or old_value is not None:
        return _ResolvedTarget(
            target=AdjustmentTarget.BEHAVIOR_CONFIDENCE,
            target_id=None,
            target_facet="behavior",
            field="confidence",
            old_value=old_value,
            store_available=False,
            current_value_unavailable=old_value is None,
            metadata={"owner": "behavior", "signal_source": signal.source.value},
        )
    return None


def _build_adjustment_record(
    signal: LearningSignal,
    *,
    target: _ResolvedTarget,
    direction: float,
    alpha: float,
    minor_nudge: float,
    proposal_threshold: float,
    memory_store: MemoryStore | None,
    goal_store: GoalStore | None,
) -> AdjustmentRecord:
    reasons = list(signal.reasons)
    metadata = dict(target.metadata or {})
    metadata["source"] = signal.source.value

    if target.no_target:
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            delta=0.0,
            status=AdjustmentStatus.SKIPPED,
            source_signal_id=signal.id,
            reasons=[*reasons, "no_target"],
            metadata=metadata,
        )
    if target.unknown_target:
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            delta=0.0,
            status=AdjustmentStatus.SKIPPED,
            source_signal_id=signal.id,
            reasons=[*reasons, "unknown_target"],
            metadata=metadata,
        )

    if target.old_value is None:
        proposed_delta = round(direction * minor_nudge, 3)
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=None,
            new_value=None,
            delta=proposed_delta,
            status=AdjustmentStatus.PROPOSED,
            source_signal_id=signal.id,
            reasons=[*reasons, "current_value_unavailable"],
            metadata=metadata,
        )

    anchor = 1.0 if direction > 0 else 0.0
    ema_value = _clamp_unit((alpha * anchor) + ((1.0 - alpha) * target.old_value))
    raw_delta = round(ema_value - target.old_value, 3)
    if raw_delta == 0.0:
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=target.old_value,
            new_value=target.old_value,
            delta=0.0,
            status=AdjustmentStatus.SKIPPED,
            source_signal_id=signal.id,
            reasons=[*reasons, "no_change"],
            metadata=metadata,
        )

    if abs(raw_delta) >= proposal_threshold:
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=target.old_value,
            new_value=ema_value,
            delta=raw_delta,
            status=AdjustmentStatus.PROPOSED,
            source_signal_id=signal.id,
            reasons=[*reasons, "proposal_threshold_exceeded"],
            metadata=metadata,
        )

    safe_delta = round(max(-minor_nudge, min(raw_delta, minor_nudge)), 3)
    new_value = _clamp_unit(target.old_value + safe_delta)
    delta = round(new_value - target.old_value, 3)
    if delta == 0.0:
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=target.old_value,
            new_value=target.old_value,
            delta=0.0,
            status=AdjustmentStatus.SKIPPED,
            source_signal_id=signal.id,
            reasons=[*reasons, "clamped_to_no_change"],
            metadata=metadata,
        )

    if target.target == AdjustmentTarget.MEMORY_SALIENCE and target.store_available:
        assert target.target_id is not None
        assert memory_store is not None
        memory_store.update_memory_salience(target.target_id, new_value)
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=target.old_value,
            new_value=new_value,
            delta=delta,
            status=AdjustmentStatus.APPLIED,
            source_signal_id=signal.id,
            reasons=[*reasons, "applied_minor_nudge"],
            metadata=metadata,
        )

    if target.target == AdjustmentTarget.GOAL_PRIORITY and target.store_available:
        assert target.target_id is not None
        assert goal_store is not None
        goal = goal_store.get_goal(target.target_id)
        if goal is None:
            return AdjustmentRecord(
                target=target.target,
                target_id=target.target_id,
                target_facet=target.target_facet,
                field=target.field,
                delta=0.0,
                status=AdjustmentStatus.SKIPPED,
                source_signal_id=signal.id,
                reasons=[*reasons, "unknown_target"],
                metadata=metadata,
            )
        updated_goal = Goal.from_dict(goal.to_dict())
        updated_goal.priority = _clamp_unit(new_value)
        goal_store.update_goal(updated_goal)
        return AdjustmentRecord(
            target=target.target,
            target_id=target.target_id,
            target_facet=target.target_facet,
            field=target.field,
            old_value=target.old_value,
            new_value=updated_goal.priority,
            delta=round(updated_goal.priority - target.old_value, 3),
            status=AdjustmentStatus.APPLIED,
            source_signal_id=signal.id,
            reasons=[*reasons, "applied_minor_nudge"],
            metadata=metadata,
        )

    return AdjustmentRecord(
        target=target.target,
        target_id=target.target_id,
        target_facet=target.target_facet,
        field=target.field,
        old_value=target.old_value,
        new_value=new_value,
        delta=delta,
        status=AdjustmentStatus.PROPOSED,
        source_signal_id=signal.id,
        reasons=[*reasons, "store_unavailable"],
        metadata=metadata,
    )


def _make_skipped_record(signal: LearningSignal) -> AdjustmentRecord:
    fallback = _fallback_target_for_signal(signal)
    return AdjustmentRecord(
        target=fallback.target,
        target_id=fallback.target_id,
        target_facet=fallback.target_facet,
        field=fallback.field,
        delta=0.0,
        status=AdjustmentStatus.SKIPPED,
        source_signal_id=signal.id,
        reasons=[*signal.reasons, "no_target"],
        metadata={"source": signal.source.value},
    )


def _fallback_target_for_signal(signal: LearningSignal) -> _ResolvedTarget:
    if signal.source == SignalSource.GOAL_LIFECYCLE:
        return _ResolvedTarget(
            target=AdjustmentTarget.GOAL_PRIORITY,
            target_id=None,
            target_facet="goals",
            field="priority",
            old_value=None,
            store_available=False,
            no_target=True,
        )
    return _ResolvedTarget(
        target=AdjustmentTarget.BEHAVIOR_CONFIDENCE,
        target_id=None,
        target_facet="behavior",
        field="confidence",
        old_value=None,
        store_available=False,
        no_target=True,
    )


def _signal_direction(signal_type: SignalType) -> float:
    if signal_type == SignalType.POSITIVE:
        return 1.0
    if signal_type in {SignalType.NEGATIVE, SignalType.FAILURE}:
        return -1.0
    return 0.0


def _overall_status(
    signals: list[LearningSignal],
    adjustments: list[AdjustmentRecord],
) -> str:
    if not signals:
        return "no_signal"
    if not adjustments:
        return "signals_only"
    statuses = {record.status for record in adjustments}
    if statuses == {AdjustmentStatus.APPLIED}:
        return "applied"
    if statuses == {AdjustmentStatus.PROPOSED}:
        return "proposed"
    if statuses == {AdjustmentStatus.SKIPPED}:
        return "skipped"
    return "mixed"


def _clamp_unit(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


def _numeric_unit_value(raw_value: Any) -> float | None:
    if not isinstance(raw_value, (int, float)):
        return None
    return _clamp_unit(raw_value)


def _coerce_adjustment_target(raw_value: Any) -> AdjustmentTarget | None:
    if isinstance(raw_value, AdjustmentTarget):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    if not cleaned:
        return None
    try:
        return AdjustmentTarget(cleaned)
    except ValueError:
        return None


def _coerce_behavior_target(raw_value: Any) -> AdjustmentTarget | None:
    if isinstance(raw_value, AdjustmentTarget):
        if raw_value in {
            AdjustmentTarget.BEHAVIOR_CONFIDENCE,
            AdjustmentTarget.BEHAVIOR_THRESHOLD,
        }:
            return raw_value
        return None
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().lower()
    mapping = {
        "behavior_confidence": AdjustmentTarget.BEHAVIOR_CONFIDENCE,
        "confidence": AdjustmentTarget.BEHAVIOR_CONFIDENCE,
        "behavior_threshold": AdjustmentTarget.BEHAVIOR_THRESHOLD,
        "threshold": AdjustmentTarget.BEHAVIOR_THRESHOLD,
    }
    return mapping.get(cleaned)
