"""Latent Pressure Buffer (LPB) support infrastructure."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from fullerene.nexus.models import Event, EventType, FacetResult, NexusState
from fullerene.signals.latent_pressure.models import (
    LatentPressureEntry,
    LatentPressureResult,
    utcnow,
)

DEFAULT_DECAY_RATE = 0.05
DEFAULT_ESCALATION_RATE = 0.08
TOP_ENTRY_LIMIT = 5
MAX_RESOLVED_ENTRIES = 20
OVERLOAD_RATIO_THRESHOLD = 0.85
IDLE_TICK_DECAY_MULTIPLIER = 2.0
MIN_IDLE_DECAY = 0.05

INITIAL_INTENSITIES = {
    "verifier_failure": 0.7,
    "policy_block": 0.65,
    "approval_required": 0.5,
    "contradiction": 0.6,
    "uncertainty": 0.45,
    "context_overload": 0.4,
    "planner_conflict": 0.45,
    "attention_conflict": 0.3,
    "interrupt_recommendation": 0.55,
    "learning_route": 0.35,
}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _dict(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _extract_facet_metadata(
    facet_results: list[FacetResult], facet_name: str
) -> dict[str, Any]:
    for result in reversed(facet_results):
        if result.facet_name != facet_name:
            continue
        if isinstance(result.metadata, dict):
            return result.metadata
    return {}


def _stable_key(source: str, entry_type: str, source_id: str | None, description: str) -> str:
    payload = f"{source}|{entry_type}|{source_id or ''}|{description.strip().lower()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324 deterministic key


def _merge_metadata(original: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(original)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
    return merged


def _compute_total(active: list[LatentPressureEntry]) -> float:
    weights = (0.55, 0.30, 0.15, 0.08, 0.04)
    ranked = sorted(active, key=lambda item: item.intensity, reverse=True)[: len(weights)]
    total = 0.0
    for idx, entry in enumerate(ranked):
        type_factor = 1.0
        if idx > 0 and entry.entry_type == ranked[0].entry_type:
            type_factor = 0.25
        total += entry.intensity * weights[idx] * type_factor
    return _clamp01(total)


def _ignition_info(
    active: list[LatentPressureEntry],
    total: float,
    *,
    is_system_tick: bool,
    has_critical_signal: bool,
    has_non_suppressed_signal: bool,
) -> tuple[bool, str | None, str | None, str | None]:
    if not active:
        return False, None, None, None
    top = max(active, key=lambda item: item.intensity)
    if top.intensity >= 0.75:
        return True, "top_entry_intensity_high", top.id, top.entry_type
    if total >= 0.85:
        if is_system_tick:
            if top.intensity >= 0.75:
                return True, "latent_pressure_total_high_tick_top_intense", top.id, top.entry_type
            if top.retrigger_count >= 3 and has_non_suppressed_signal:
                return True, "latent_pressure_total_high_tick_retrigger", top.id, top.entry_type
            if has_critical_signal:
                return True, "latent_pressure_total_high_tick_critical", top.id, top.entry_type
            return False, None, None, None
        return True, "latent_pressure_total_high", top.id, top.entry_type
    if top.retrigger_count >= 3 and top.intensity >= 0.55:
        return True, "retrigger_threshold", top.id, top.entry_type
    return False, None, None, None


def _signals_from_behavior(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    trace = _dict(metadata.get("decision_trace"))
    source_id = trace.get("event", {}).get("id") if isinstance(trace.get("event"), dict) else None
    if bool(trace.get("contradiction_flag")):
        out.append(
            {
                "source": "behavior",
                "entry_type": "contradiction",
                "source_id": source_id,
                "description": "Behavior decision trace contradiction flag.",
                "metadata": {"trace_key": "contradiction_flag"},
            }
        )
    elif bool(metadata.get("belief_contradiction")):
        out.append(
            {
                "source": "behavior",
                "entry_type": "contradiction",
                "source_id": source_id,
                "description": "Behavior metadata reports belief contradiction.",
                "metadata": {"trace_key": "belief_contradiction"},
            }
        )
    if bool(trace.get("interrupt_recommended")) or bool(metadata.get("interrupt_recommended")):
        out.append(
            {
                "source": "behavior",
                "entry_type": "interrupt_recommendation",
                "source_id": source_id,
                "description": "Behavior recommends interrupt.",
                "metadata": {"interrupt_reason": metadata.get("interrupt_reason")},
            }
        )
    policy_result = str(
        trace.get("policy_result") or metadata.get("policy_result") or ""
    ).strip().lower()
    if policy_result == "denied":
        out.append(
            {
                "source": "behavior",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Behavior policy result denied action.",
                "metadata": {"policy_result": policy_result},
            }
        )
    if policy_result == "approval_required":
        out.append(
            {
                "source": "behavior",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Behavior policy result requires approval.",
                "metadata": {"policy_result": policy_result},
            }
        )
    if bool(metadata.get("policy_blocks_act")):
        out.append(
            {
                "source": "behavior",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Behavior metadata indicates policy blocks ACT.",
                "metadata": {"policy_blocks_act": True},
            }
        )
    if bool(metadata.get("policy_requires_approval")):
        out.append(
            {
                "source": "behavior",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Behavior metadata indicates policy requires approval.",
                "metadata": {"policy_requires_approval": True},
            }
        )
    ratio = _clamp01(
        trace.get("context_load_ratio")
        if "context_load_ratio" in trace
        else _dict(metadata.get("context_load")).get("load_ratio")
    )
    if ratio >= OVERLOAD_RATIO_THRESHOLD:
        out.append(
            {
                "source": "behavior",
                "entry_type": "context_overload",
                "source_id": source_id,
                "description": "Behavior reports context overload.",
                "metadata": {"context_load_ratio": ratio},
            }
        )
    confidence = _clamp01(trace.get("world_model_belief_confidence"))
    if 0.0 < confidence < 0.4:
        out.append(
            {
                "source": "world_model",
                "entry_type": "uncertainty",
                "source_id": source_id,
                "description": "Low world-model belief confidence in behavior trace.",
                "metadata": {"belief_confidence": confidence},
            }
        )
    return out


def _signals_from_nexus(signal_map: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_id = signal_map.get("event_id")
    if bool(signal_map.get("belief_contradiction")):
        out.append(
            {
                "source": "nexus",
                "entry_type": "contradiction",
                "source_id": source_id,
                "description": "Nexus cycle signal map reports belief contradiction.",
                "metadata": {},
            }
        )
    if bool(signal_map.get("context_overloaded")):
        out.append(
            {
                "source": "nexus",
                "entry_type": "context_overload",
                "source_id": source_id,
                "description": "Nexus cycle signal map reports context overload.",
                "metadata": {},
            }
        )
    if bool(signal_map.get("interrupt_recommended")):
        out.append(
            {
                "source": "nexus",
                "entry_type": "interrupt_recommendation",
                "source_id": source_id,
                "description": "Nexus cycle signal map recommends interrupt.",
                "metadata": {"interrupt_reason": signal_map.get("interrupt_reason")},
            }
        )
    if bool(signal_map.get("policy_blocks_act")):
        out.append(
            {
                "source": "nexus",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Nexus policy blocks act.",
                "metadata": {"policy_status": signal_map.get("policy_status")},
            }
        )
    if bool(signal_map.get("policy_requires_approval")):
        out.append(
            {
                "source": "nexus",
                "entry_type": "policy_block",
                "source_id": source_id,
                "description": "Nexus policy requires approval.",
                "metadata": {"policy_status": signal_map.get("policy_status")},
            }
        )
    return out


def _signals_from_attention(metadata: dict[str, Any], event_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if bool(metadata.get("attention_conflict")):
        out.append(
            {
                "source": "attention",
                "entry_type": "attention_conflict",
                "source_id": event_id,
                "description": "Attention reports close-score conflict.",
                "metadata": {"conflict_items": metadata.get("conflict_items", [])},
            }
        )
    contribution = _clamp01(metadata.get("pressure_contribution"))
    if contribution >= 0.25:
        out.append(
            {
                "source": "attention",
                "entry_type": "attention_conflict",
                "source_id": event_id,
                "description": "Attention repeated pressure contribution elevated.",
                "metadata": {"pressure_contribution": contribution},
            }
        )
    return out


def _signals_from_planner(metadata: dict[str, Any], event_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    conflict_report = _dict(metadata.get("conflict_report"))
    if bool(conflict_report.get("has_conflicts")):
        out.append(
            {
                "source": "planner",
                "entry_type": "planner_conflict",
                "source_id": event_id,
                "description": "Planner conflict report contains conflicts.",
                "metadata": {"conflicts": conflict_report.get("conflicts", [])},
            }
        )
    grounding = str(metadata.get("grounding_status") or "").strip().lower()
    if grounding in {"failed", "weak", "ungrounded", "insufficient_context"}:
        out.append(
            {
                "source": "planner",
                "entry_type": "goal_block",
                "source_id": event_id,
                "description": f"Planner grounding status is {grounding}.",
                "metadata": {"grounding_status": grounding},
            }
        )
    if metadata.get("blocked_steps"):
        out.append(
            {
                "source": "planner",
                "entry_type": "planner_conflict",
                "source_id": event_id,
                "description": "Planner produced blocked steps.",
                "metadata": {"blocked_steps": metadata.get("blocked_steps")},
            }
        )
    if metadata.get("approval_required_steps"):
        out.append(
            {
                "source": "planner",
                "entry_type": "policy_block",
                "source_id": event_id,
                "description": "Planner produced approval-required steps.",
                "metadata": {"approval_required_steps": metadata.get("approval_required_steps")},
            }
        )
    return out


def _signals_from_verifier(metadata: dict[str, Any], event_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in metadata.get("artifact_checks", []):
        if not isinstance(row, dict):
            continue
        severity = str(row.get("severity") or "").lower()
        status = str(row.get("status") or "").lower()
        if severity in {"critical", "error"} or status == "failed":
            out.append(
                {
                    "source": "verifier",
                    "entry_type": "verifier_failure",
                    "source_id": event_id,
                    "description": f"Verifier check {row.get('code', 'unknown')} failed.",
                    "metadata": {
                        "code": row.get("code"),
                        "severity": severity,
                        "retry_recommended": bool(row.get("retry_recommended")),
                        "escalation_recommended": bool(
                            row.get("escalation_recommended")
                        ),
                    },
                }
            )
    return out


def _signals_from_learning(metadata: dict[str, Any], event_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for route in metadata.get("cross_facet_routes", []):
        if not isinstance(route, dict):
            continue
        target = str(route.get("target_facet") or route.get("target") or "")
        if target in {"nexus", "context", "behavior", "world_model"}:
            out.append(
                {
                    "source": "learning",
                    "entry_type": "learning_route",
                    "source_id": event_id,
                    "description": "Learning emitted unresolved cross-facet route.",
                    "metadata": {"route": route},
                }
            )
            if route.get("signal") == "latent_pressure_interrupt":
                out.append(
                    {
                        "source": "learning",
                        "entry_type": "interrupt_recommendation",
                        "source_id": event_id,
                        "description": "Learning route recommends latent-pressure interrupt.",
                        "metadata": {"route": route},
                    }
                )
    return out


def _signals_from_world_model(metadata: dict[str, Any], event_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = metadata.get("contradiction_signals")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        payload.setdefault("source", "world_model")
        payload.setdefault("source_id", event_id)
        payload.setdefault("entry_type", "uncertainty")
        payload.setdefault("description", "World model signal.")
        payload.setdefault("metadata", {})
        out.append(payload)
    return out


def _resolve_entries(
    entries: list[LatentPressureEntry],
    event: Event,
    *,
    reactivated_ids: set[str],
    now: datetime,
    is_system_tick: bool,
) -> list[dict[str, Any]]:
    resolved_entries: list[dict[str, Any]] = []
    event_resolve = event.metadata.get("resolve_latent_pressure")
    if isinstance(event_resolve, list):
        targets = {str(item).strip().lower() for item in event_resolve}
    else:
        targets = set()
    for entry in entries:
        if entry.status != "active":
            continue
        entry_key = entry.id.strip().lower()
        if entry_key in targets or entry.entry_type in targets:
            entry.status = "resolved"
            entry.last_decayed_at = now
            resolved_entries.append(entry.to_dict())
        elif entry.id not in reactivated_ids:
            decay_amount = entry.decay_rate
            if is_system_tick:
                decay_amount = max(
                    entry.decay_rate * IDLE_TICK_DECAY_MULTIPLIER, MIN_IDLE_DECAY
                )
            entry.intensity = _clamp01(entry.intensity - decay_amount)
            entry.last_decayed_at = now
            if entry.intensity <= 0.05:
                entry.status = "resolved"
                resolved_entries.append(entry.to_dict())
    return resolved_entries


def should_ingest_signal_on_tick(
    signal: dict[str, Any], event: Event, state: NexusState
) -> tuple[bool, str]:
    if bool(event.metadata.get("force_latent_pressure_ingest")):
        return True, "force_latent_pressure_ingest"
    if event.event_type != EventType.SYSTEM_TICK:
        return True, "non_system_tick"

    source = str(signal.get("source") or "unknown").strip().lower()
    entry_type = str(signal.get("entry_type") or "unknown").strip().lower()
    metadata = _dict(signal.get("metadata"))
    source_id = (
        str(signal.get("source_id")).strip()
        if signal.get("source_id") is not None
        else None
    )

    if source == "verifier" and entry_type == "verifier_failure":
        severity = str(metadata.get("severity") or "").lower()
        if severity in {"critical", "error"} or bool(metadata.get("escalation_recommended")):
            return True, "critical_verifier"

    if entry_type == "policy_block":
        p = str(
            metadata.get("policy_result")
            or metadata.get("policy_status")
            or ("denied" if metadata.get("policy_blocks_act") else "")
        ).lower()
        if p in {"denied", "approval_required"}:
            existing = _dict(_dict(state.facet_state.get("signals")).get("latent_pressure"))
            for row in existing.get("entries", []):
                if not isinstance(row, dict) or row.get("status") != "active":
                    continue
                if (
                    str(row.get("entry_type") or "") == "policy_block"
                    and str(row.get("source") or "") == source
                    and str(row.get("source_id") or "") == str(source_id or "")
                ):
                    return False, "policy_block_repeat_same_source"
            return True, "policy_block_new_source"

    if source == "nexus" and entry_type == "interrupt_recommendation":
        reason = str(metadata.get("interrupt_reason") or "").lower()
        if "latent_pressure" in reason or "lpb" in reason or "ignition" in reason:
            return False, "nexus_lpb_interrupt_echo"

    if source == "behavior" and entry_type == "interrupt_recommendation":
        reason = str(metadata.get("interrupt_reason") or "").lower()
        if any(tok in reason for tok in ("latent_pressure", "lpb", "ignition")):
            return False, "behavior_lpb_interrupt_echo"
        if str(metadata.get("severity") or "").lower() in {"high", "critical"}:
            return True, "behavior_interrupt_high_severity"
        last_source = _dict(state.facet_state.get("nexus")).get("last_interrupt_source_id")
        if source_id and str(last_source or "") != source_id:
            return True, "behavior_interrupt_new_source"
        return False, "behavior_interrupt_echo"

    if source == "learning":
        route = _dict(metadata.get("route"))
        sig = str(route.get("signal") or "")
        if sig == "latent_pressure_interrupt":
            if source_id and source_id != str(event.event_id):
                return True, "learning_lpb_interrupt_new"
            return False, "learning_lpb_interrupt_echo"

    if entry_type == "context_overload":
        ratio = _clamp01(metadata.get("context_load_ratio"))
        if ratio >= 0.95:
            return True, "context_overload_critical_ratio"
        nstate = _dict(state.facet_state.get("nexus"))
        if bool(nstate.get("last_context_overloaded_changed")):
            return True, "context_overload_new_change"
        return False, "context_overload_echo"

    if source == "world_model" and entry_type == "uncertainty":
        belief_id = str(metadata.get("belief_id") or "")
        if belief_id:
            return True, "uncertainty_new_belief"
        return False, "uncertainty_echo"

    if entry_type == "attention_conflict":
        conflict_items = metadata.get("conflict_items")
        if isinstance(conflict_items, list) and conflict_items:
            return True, "attention_conflict_new"
        return False, "attention_conflict_echo"

    return False, "system_tick_routine_signal_suppressed"


def update_latent_pressure(
    *,
    event: Event,
    state: NexusState,
    facet_results: list[FacetResult],
) -> tuple[dict[str, Any], LatentPressureResult]:
    now = utcnow()
    existing_state = _dict(
        _dict(state.facet_state.get("signals")).get("latent_pressure")
    )
    entries = [
        LatentPressureEntry.from_dict(item)
        for item in existing_state.get("entries", [])
        if isinstance(item, dict)
    ]
    active_by_key: dict[str, LatentPressureEntry] = {}
    for entry in entries:
        if entry.status == "active":
            key = _stable_key(entry.source, entry.entry_type, entry.source_id, entry.description)
            active_by_key[key] = entry

    behavior_meta = _extract_facet_metadata(facet_results, "behavior")
    attention_meta = _extract_facet_metadata(facet_results, "attention")
    planner_meta = _extract_facet_metadata(facet_results, "planner")
    verifier_meta = _extract_facet_metadata(facet_results, "verifier")
    learning_meta = _extract_facet_metadata(facet_results, "learning")
    world_model_meta = _extract_facet_metadata(facet_results, "world_model")
    signal_map = _dict(_dict(state.facet_state.get("nexus")).get("current_cycle_signal_map"))

    is_system_tick = event.event_type == EventType.SYSTEM_TICK
    incoming = []
    incoming.extend(_signals_from_behavior(behavior_meta))
    incoming.extend(_signals_from_nexus(signal_map))
    incoming.extend(_signals_from_attention(attention_meta, event.event_id))
    incoming.extend(_signals_from_planner(planner_meta, event.event_id))
    incoming.extend(_signals_from_verifier(verifier_meta, event.event_id))
    incoming.extend(_signals_from_learning(learning_meta, event.event_id))
    incoming.extend(_signals_from_world_model(world_model_meta, event.event_id))

    created_entries: list[dict[str, Any]] = []
    updated_entries: list[dict[str, Any]] = []
    reactivated_ids: set[str] = set()
    skipped_signals: list[dict[str, Any]] = []
    skip_reasons: list[str] = []
    reasons: list[str] = []

    filtered_incoming: list[dict[str, Any]] = []
    for signal in incoming:
        allowed, why = should_ingest_signal_on_tick(signal, event, state)
        if allowed:
            filtered_incoming.append(signal)
            continue
        skipped = dict(signal)
        skipped["skip_reason"] = why
        skipped_signals.append(skipped)
        skip_reasons.append(why)
    incoming = filtered_incoming

    for signal in incoming:
        source = str(signal.get("source", "unknown"))
        entry_type = str(signal.get("entry_type", "unknown"))
        source_id = (
            str(signal["source_id"]) if signal.get("source_id") is not None else None
        )
        description = str(signal.get("description", "")).strip()
        metadata = _dict(signal.get("metadata"))
        key = _stable_key(source, entry_type, source_id, description)
        source_boost = _clamp01(metadata.get("source_boost", 0.0))
        event_source_id = (
            str(signal.get("source_id")).strip()
            if signal.get("source_id") is not None
            else None
        )
        if key in active_by_key:
            entry = active_by_key[key]
            escalation = entry.escalation_rate
            if is_system_tick and (
                (event_source_id and entry.last_reactivation_source_id == event_source_id)
                or entry.last_reactivation_event_type == event.event_type.value
            ):
                escalation = entry.escalation_rate * 0.25
            entry.intensity = _clamp01(entry.intensity + escalation + source_boost)
            entry.retrigger_count += 1
            entry.last_activated_at = now
            entry.last_reactivation_event_type = event.event_type.value
            entry.last_reactivation_source_id = event_source_id
            if is_system_tick:
                entry.tick_reactivation_count += 1
            entry.metadata = _merge_metadata(entry.metadata, metadata)
            updated_entries.append(entry.to_dict())
            reactivated_ids.add(entry.id)
        else:
            entry = LatentPressureEntry(
                source=source,
                source_id=source_id,
                entry_type=entry_type,
                description=description,
                intensity=INITIAL_INTENSITIES.get(entry_type, 0.35),
                decay_rate=_clamp01(metadata.get("decay_rate", DEFAULT_DECAY_RATE)),
                escalation_rate=_clamp01(
                    metadata.get("escalation_rate", DEFAULT_ESCALATION_RATE)
                ),
                retrigger_count=1,
                created_at=now,
                last_activated_at=now,
                last_decayed_at=now,
                last_reactivation_event_type=event.event_type.value,
                last_reactivation_source_id=event_source_id,
                tick_reactivation_count=1 if is_system_tick else 0,
                status="active",
                metadata=metadata,
            )
            entries.append(entry)
            active_by_key[key] = entry
            created_entries.append(entry.to_dict())
            reactivated_ids.add(entry.id)

    resolved_entries = _resolve_entries(
        entries,
        event,
        reactivated_ids=reactivated_ids,
        now=now,
        is_system_tick=is_system_tick,
    )
    decayed_entries = [
        entry.to_dict()
        for entry in entries
        if entry.status == "active" and entry.id not in reactivated_ids
    ]

    active_entries = [entry for entry in entries if entry.status == "active"]
    resolved = [entry for entry in entries if entry.status == "resolved"]
    resolved = sorted(resolved, key=lambda item: item.last_decayed_at, reverse=True)[
        :MAX_RESOLVED_ENTRIES
    ]
    entries = active_entries + resolved
    top_entries = sorted(active_entries, key=lambda item: item.intensity, reverse=True)[:TOP_ENTRY_LIMIT]
    total = _compute_total(active_entries)
    has_critical = any(
        e.entry_type in {"verifier_failure", "policy_block"}
        and any(
            str(_dict(e.metadata).get(k, "")).lower() in {"critical", "error", "denied"}
            for k in ("severity", "policy_result", "policy_status")
        )
        for e in active_entries
    )
    ign_rec, ign_reason, ign_id, ign_et = _ignition_info(
        active_entries,
        total,
        is_system_tick=is_system_tick,
        has_critical_signal=has_critical,
        has_non_suppressed_signal=bool(incoming),
    )
    reasons.append(f"signals_ingested={len(incoming)}")
    reasons.append(f"signals_skipped={len(skipped_signals)}")
    if ign_rec and ign_reason:
        reasons.append(f"ignition_reason={ign_reason}")

    result = LatentPressureResult(
        latent_pressure_total=total,
        active_entries=[item.to_dict() for item in active_entries],
        top_entries=[item.to_dict() for item in top_entries],
        created_entries=created_entries,
        updated_entries=updated_entries,
        decayed_entries=decayed_entries,
        resolved_entries=resolved_entries,
        ignition_recommended=ign_rec,
        ignition_reason=ign_reason,
        ignition_entry_id=ign_id,
        ignition_entry_type=ign_et,
        skipped_signals=skipped_signals,
        skipped_signal_count=len(skipped_signals),
        skip_reasons=sorted(set(skip_reasons)),
        reasons=reasons,
    )
    state_blob = {
        "version": "v1",
        "updated_at": now.isoformat(),
        "entries": [item.to_dict() for item in entries],
        "last_result": result.to_dict(),
    }
    return state_blob, result

