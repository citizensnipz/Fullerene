"""Nexus v2 bounded interrupt candidates, scoring, and suppression (deterministic, JSON-safe)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from fullerene.nexus.models import Event, EventType, FacetResult, NexusState, utcnow
from fullerene.policy.models import PolicyStatus

# --- Constants ---

MAX_INTERRUPT_QUEUE = 5
LOW_PRIORITY_THRESHOLD = 0.55
COOLDOWN_BYPASS_PRIORITY = 0.85
COOLDOWN_PRIORITY_PENALTY = 0.08
CONTEXT_OVERLOAD_PRIORITY_PENALTY = 0.15
COOLDOWN_RECENT_EVENT_WINDOW = 3

INTERRUPT_SOURCES = frozenset(
    {
        "behavior",
        "latent_pressure",
        "verifier",
        "learning",
        "policy",
        "planner",
        "attention",
        "nexus",
        "unknown",
    }
)
INTERRUPT_TYPES = frozenset(
    {
        "behavior_interrupt",
        "latent_pressure_ignition",
        "latent_pressure_interrupt",
        "verifier_escalation",
        "verifier_retry",
        "policy_block",
        "approval_required",
        "planner_conflict",
        "attention_conflict",
        "learning_route",
        "system_pressure",
        "unknown",
    }
)

SOURCE_SEVERITY: dict[str, float] = {
    "verifier_escalation": 0.9,
    "policy_block": 0.8,
    "latent_pressure_ignition": 0.75,
    "latent_pressure_interrupt": 0.5,
    "approval_required": 0.65,
    "behavior_interrupt": 0.6,
    "planner_conflict": 0.55,
    "learning_route": 0.45,
    "system_pressure": 0.4,
    "attention_conflict": 0.35,
    "verifier_retry": 0.55,
    "unknown": 0.3,
}

SYSTEM_PRESSURE_INTERRUPT_THRESHOLD = 0.82


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _json_safe(value: Any, *, max_depth: int = 4, _d: int = 0) -> Any:
    if _d > max_depth:
        return None
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if value != value:  # noqa: PLC3002
            return 0.0
        return round(value, 6)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v, max_depth=max_depth, _d=_d + 1)
            for k, v in list(value.items())[:48]
            if isinstance(k, (str, int))
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v, max_depth=max_depth, _d=_d + 1)
            for v in value[:32]
        ]
    return str(value)


def _stable_candidate_id(parts: Sequence[str]) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"ic-{h}"


def _norm_enum(value: Any, allowed: frozenset[str], default: str) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in allowed else default


@dataclass(slots=True)
class InterruptCandidate:
    id: str
    source: str
    source_id: str | None
    interrupt_type: str
    priority: float
    pressure: float
    confidence: float
    novelty: float
    reason: str
    payload: dict[str, Any]
    created_at: str
    parent_event_id: str
    suppressible: bool
    requires_user_attention: bool
    cooldown_key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "interrupt_type": self.interrupt_type,
            "priority": _clamp01(self.priority),
            "pressure": _clamp01(self.pressure),
            "confidence": _clamp01(self.confidence),
            "novelty": _clamp01(self.novelty),
            "reason": str(self.reason or ""),
            "payload": _json_safe(dict(self.payload)),
            "created_at": self.created_at,
            "parent_event_id": self.parent_event_id,
            "suppressible": bool(self.suppressible),
            "requires_user_attention": bool(self.requires_user_attention),
            "cooldown_key": self.cooldown_key,
            "metadata": _json_safe(dict(self.metadata)),
        }


@dataclass(slots=True)
class SuppressionDecision:
    candidate_id: str
    suppressed: bool
    suppression_reason: str
    suppression_rules_triggered: list[str]
    cooldown_applied: bool
    cooldown_key: str
    previous_count: int
    priority_before: float
    priority_after: float
    allowed_internal_event: bool
    allowed_user_expression: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "suppressed": bool(self.suppressed),
            "suppression_reason": str(self.suppression_reason or ""),
            "suppression_rules_triggered": list(self.suppression_rules_triggered),
            "cooldown_applied": bool(self.cooldown_applied),
            "cooldown_key": self.cooldown_key,
            "previous_count": int(max(0, self.previous_count)),
            "priority_before": _clamp01(self.priority_before),
            "priority_after": _clamp01(self.priority_after),
            "allowed_internal_event": bool(self.allowed_internal_event),
            "allowed_user_expression": bool(self.allowed_user_expression),
            "reasons": list(self.reasons),
        }


def _latest_facet_metadata(
    facet_results: list[FacetResult], facet_name: str
) -> dict[str, Any]:
    for result in reversed(facet_results):
        if result.facet_name != facet_name:
            continue
        if isinstance(result.metadata, dict):
            return result.metadata
    return {}


def _artifact_critical_or_error(metadata: Mapping[str, Any]) -> bool:
    for row in metadata.get("artifact_checks", []) or []:
        if not isinstance(row, dict):
            continue
        sev = str(row.get("severity") or "").lower()
        st = str(row.get("status") or "").lower()
        if sev in {"critical", "error"} or st == "failed":
            return True
    return False


def interrupt_cooldown_key(source: str, interrupt_type: str, stable_hint: str) -> str:
    raw = f"{source}|{interrupt_type}|{stable_hint.strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def score_interrupt_priority(*, cand: InterruptCandidate, raw: Mapping[str, Any]) -> InterruptCandidate:
    """Apply deterministic Nexus v2 scoring; store components in cand.metadata."""
    pressure = _clamp01(cand.pressure)
    source_severity = SOURCE_SEVERITY.get(cand.interrupt_type, 0.3)
    novelty = _clamp01(cand.novelty)
    confidence = _clamp01(cand.confidence)
    recurrence_bonus = _clamp01(float(raw.get("recurrence_bonus", cand.metadata.get("recurrence_bonus", 0.0))))
    user_attention_bonus = (
        _clamp01(1.0) if cand.requires_user_attention else 0.0
    )

    priority = _clamp01(
        pressure * 0.30
        + source_severity * 0.30
        + novelty * 0.15
        + confidence * 0.10
        + recurrence_bonus * 0.10
        + user_attention_bonus * 0.05
    )
    components = {
        "pressure_term": round(pressure * 0.30, 6),
        "severity_term": round(source_severity * 0.30, 6),
        "novelty_term": round(novelty * 0.15, 6),
        "confidence_term": round(confidence * 0.10, 6),
        "recurrence_term": round(recurrence_bonus * 0.10, 6),
        "user_attention_term": round(user_attention_bonus * 0.05, 6),
        "source_severity_base": source_severity,
    }
    meta = dict(cand.metadata)
    meta["priority_components"] = components
    return InterruptCandidate(
        id=cand.id,
        source=cand.source,
        source_id=cand.source_id,
        interrupt_type=cand.interrupt_type,
        priority=priority,
        pressure=pressure,
        confidence=confidence,
        novelty=novelty,
        reason=cand.reason,
        payload=cand.payload,
        created_at=cand.created_at,
        parent_event_id=cand.parent_event_id,
        suppressible=cand.suppressible,
        requires_user_attention=cand.requires_user_attention,
        cooldown_key=cand.cooldown_key,
        metadata=meta,
    )


def _mk_candidate(
    *,
    interrupt_type: str,
    source: str,
    source_id: str | None,
    parent_event_id: str,
    reason: str,
    pressure: float,
    confidence: float,
    novelty: float,
    payload: Mapping[str, Any],
    suppressible: bool,
    requires_user_attention: bool,
    cooldown_hint: str,
    recurrence_bonus: float = 0.0,
    implies_act_denied: bool = False,
) -> InterruptCandidate:
    src = _norm_enum(source, INTERRUPT_SOURCES, "unknown")
    itype = _norm_enum(interrupt_type, INTERRUPT_TYPES, "unknown")
    ck = interrupt_cooldown_key(src, itype, cooldown_hint or reason)
    cid = _stable_candidate_id([src, itype, ck, parent_event_id, reason])
    meta: dict[str, Any] = {
        "recurrence_bonus": _clamp01(recurrence_bonus),
    }
    if implies_act_denied:
        meta["implies_act_execution_denied_by_policy"] = True
    return InterruptCandidate(
        id=cid,
        source=src,
        source_id=source_id,
        interrupt_type=itype,
        priority=0.0,
        pressure=_clamp01(pressure),
        confidence=_clamp01(confidence),
        novelty=_clamp01(novelty),
        reason=str(reason),
        payload=dict(payload),
        created_at=_iso(utcnow()),
        parent_event_id=parent_event_id,
        suppressible=suppressible,
        requires_user_attention=requires_user_attention,
        cooldown_key=ck,
        metadata=meta,
    )


def extract_interrupt_candidates(
    *,
    event: Event,
    facet_results: list[FacetResult],
    signal_map_dict: Mapping[str, Any],
    latent_pressure_total: float,
    lpb_ignition_recommended: bool,
    lpb_ignition_reason: str | None,
    lpb_ignition_entry_id: str | None,
    lpb_ignition_entry_type: str | None,
) -> list[InterruptCandidate]:
    """Collect interrupt candidates from current-cycle artifacts."""
    candidates: list[InterruptCandidate] = []
    meta = event.metadata if isinstance(event.metadata, dict) else {}
    novelty = float(meta.get("novelty") or 0.0)
    evt_pressure = float(meta.get("pressure") or 0.0)
    confidence_default = float(
        _latest_facet_metadata(facet_results, "behavior").get("confidence") or 0.5
    )

    behavior_md = _latest_facet_metadata(facet_results, "behavior")
    trace = behavior_md.get("decision_trace")
    trace = trace if isinstance(trace, dict) else {}

    interrupt_rec = bool(
        behavior_md.get("interrupt_recommended") or trace.get("interrupt_recommended")
    )
    if interrupt_rec:
        ir = behavior_md.get("interrupt_reason") or trace.get("interrupt_reason")
        lp_contrib = _clamp01(float(behavior_md.get("latent_pressure") or 0.0))
        pressure_hint = max(lp_contrib, evt_pressure, 0.78)
        recur = _clamp01(trace.get("retrigger_bonus", 0.0)) if "retrigger_bonus" in trace else 0.0
        if recur == 0.0:
            recur = 0.55
        conf_i = float(
            behavior_md.get("confidence") or trace.get("confidence") or confidence_default or 0.75
        )
        cand = _mk_candidate(
            interrupt_type="behavior_interrupt",
            source="behavior",
            source_id=trace.get("event", {}).get("id")
            if isinstance(trace.get("event"), dict)
            else event.event_id,
            parent_event_id=event.event_id,
            reason=str(ir or "behavior_interrupt_recommended"),
            pressure=max(pressure_hint, lp_contrib, 0.78),
            confidence=max(conf_i, 0.75),
            novelty=novelty,
            payload={
                "behavior_summary": behavior_md.get("selected_decision"),
                "interrupt_reason": ir,
            },
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint=f"behavior_interrupt|{ir or 'default'}",
            recurrence_bonus=recur,
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    if lpb_ignition_recommended:
        cand = _mk_candidate(
            interrupt_type="latent_pressure_ignition",
            source="latent_pressure",
            source_id=lpb_ignition_entry_id,
            parent_event_id=event.event_id,
            reason=str(lpb_ignition_reason or "latent_pressure_ignition"),
            pressure=max(latent_pressure_total, 0.5),
            confidence=confidence_default,
            novelty=novelty,
            payload={
                "latent_pressure_total": latent_pressure_total,
                "ignition_entry_id": lpb_ignition_entry_id,
                "ignition_entry_type": lpb_ignition_entry_type,
                "ignition_reason": lpb_ignition_reason,
            },
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint=f"lpb_ignition|{lpb_ignition_entry_id or lpb_ignition_reason}",
            recurrence_bonus=min(1.0, latent_pressure_total * 0.4),
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    verifier_md = _latest_facet_metadata(facet_results, "verifier")
    crit = _artifact_critical_or_error(verifier_md)
    if bool(verifier_md.get("escalation_recommended")) or crit:
        cand = _mk_candidate(
            interrupt_type="verifier_escalation",
            source="verifier",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason=str(
                (verifier_md.get("escalation_reasons") or ["verifier_escalation"])[0]
                if isinstance(verifier_md.get("escalation_reasons"), list)
                else "verifier_escalation"
            ),
            pressure=max(evt_pressure, 0.55 if crit else 0.45),
            confidence=0.8,
            novelty=novelty,
            payload={
                "escalation_recommended": bool(verifier_md.get("escalation_recommended")),
                "artifact_critical_or_error": crit,
            },
            suppressible=not crit,
            requires_user_attention=crit,
            cooldown_hint="verifier_escalation",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    elif bool(verifier_md.get("retry_recommended")):
        cand = _mk_candidate(
            interrupt_type="verifier_retry",
            source="verifier",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="verifier_retry_recommended",
            pressure=0.4,
            confidence=0.7,
            novelty=novelty,
            payload={"retry_recommended": True},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="verifier_retry",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    policy_md = _latest_facet_metadata(facet_results, "policy")
    pe = policy_md.get("policy_evaluation")
    pe = pe if isinstance(pe, dict) else {}
    st = str(pe.get("status") or policy_md.get("policy_status") or "").lower()
    if st == PolicyStatus.DENIED.value:
        cand = _mk_candidate(
            interrupt_type="policy_block",
            source="policy",
            source_id=str(pe.get("effective_rule_id") or event.event_id),
            parent_event_id=event.event_id,
            reason="policy_denied_act",
            pressure=0.75,
            confidence=0.75,
            novelty=novelty,
            payload={"policy_status": st, "policy_evaluation": _json_safe(pe)},
            suppressible=False,
            requires_user_attention=True,
            cooldown_hint=f"policy_denied|{st}",
            implies_act_denied=True,
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    elif st == PolicyStatus.APPROVAL_REQUIRED.value or bool(
        behavior_md.get("policy_requires_approval")
    ):
        cand = _mk_candidate(
            interrupt_type="approval_required",
            source="policy",
            source_id=str(pe.get("effective_rule_id") or event.event_id),
            parent_event_id=event.event_id,
            reason="policy_approval_required",
            pressure=0.55,
            confidence=0.65,
            novelty=novelty,
            payload={"policy_status": st or "approval_required"},
            suppressible=True,
            requires_user_attention=True,
            cooldown_hint=f"policy_approval|{st}",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    warns = policy_md.get("warnings") if isinstance(policy_md.get("warnings"), list) else []
    if any("approval" in str(w).lower() and "invalid" in str(w).lower() for w in warns):
        cand = _mk_candidate(
            interrupt_type="approval_required",
            source="policy",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="invalid_approval_token",
            pressure=0.5,
            confidence=0.6,
            novelty=novelty,
            payload={"warnings": [str(w) for w in warns][:6]},
            suppressible=True,
            requires_user_attention=True,
            cooldown_hint="invalid_approval",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    learning_md = _latest_facet_metadata(facet_results, "learning")
    routes = learning_md.get("cross_facet_routes") if isinstance(
        learning_md.get("cross_facet_routes"), list
    ) else []
    for route in routes:
        if not isinstance(route, dict):
            continue
        tgt = str(route.get("target_facet") or route.get("target") or "").lower()
        sig = str(route.get("signal") or "")
        strength = float(route.get("strength") or 0.35)
        if tgt == "nexus" or sig == "latent_pressure_interrupt":
            itype = (
                "latent_pressure_interrupt"
                if sig == "latent_pressure_interrupt"
                else "learning_route"
            )
            cand = _mk_candidate(
                interrupt_type=itype,
                source="learning",
                source_id=event.event_id,
                parent_event_id=event.event_id,
                reason=f"learning_route:{sig or 'to_nexus'}",
                pressure=_clamp01(strength + 0.2),
                confidence=confidence_default,
                novelty=novelty,
                payload={"cross_facet_route": _json_safe(route)},
                suppressible=True,
                requires_user_attention=False,
                cooldown_hint=f"learning|{tgt}|{sig}",
                recurrence_bonus=0.15 if sig == "context_overload" else 0.0,
            )
            candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    planner_md = _latest_facet_metadata(facet_results, "planner")
    cr = planner_md.get("conflict_report")
    cr = cr if isinstance(cr, dict) else {}
    if bool(cr.get("has_conflicts")):
        cand = _mk_candidate(
            interrupt_type="planner_conflict",
            source="planner",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="planner_conflict_detected",
            pressure=0.5,
            confidence=confidence_default,
            novelty=novelty,
            payload={"conflict_report": _json_safe(cr)},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="planner_conflict",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    gstat = str(planner_md.get("grounding_status") or "").lower()
    if gstat in {"failed", "weak", "ungrounded", "insufficient_context"}:
        cand = _mk_candidate(
            interrupt_type="planner_conflict",
            source="planner",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason=f"planner_weak_grounding:{gstat}",
            pressure=0.45,
            confidence=confidence_default,
            novelty=novelty,
            payload={"grounding_status": gstat},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint=f"planner_grounding|{gstat}",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    blocked = planner_md.get("blocked_steps")
    apr = planner_md.get("approval_required_steps")
    if blocked:
        cand = _mk_candidate(
            interrupt_type="planner_conflict",
            source="planner",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="planner_blocked_steps",
            pressure=0.55,
            confidence=confidence_default,
            novelty=novelty,
            payload={"blocked_steps": _json_safe(blocked)},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="planner_blocked",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    if apr:
        cand = _mk_candidate(
            interrupt_type="approval_required",
            source="planner",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="planner_approval_required_steps",
            pressure=0.52,
            confidence=confidence_default,
            novelty=novelty,
            payload={"approval_required_steps": _json_safe(apr)},
            suppressible=True,
            requires_user_attention=True,
            cooldown_hint="planner_approval_steps",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    attn_md = _latest_facet_metadata(facet_results, "attention")
    if bool(attn_md.get("attention_conflict")):
        contrib = float(attn_md.get("pressure_contribution") or 0.0)
        cand = _mk_candidate(
            interrupt_type="attention_conflict",
            source="attention",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="attention_close_score_conflict",
            pressure=max(_clamp01(contrib + 0.25), 0.35),
            confidence=confidence_default,
            novelty=novelty,
            payload={
                "conflict_items": _json_safe(attn_md.get("conflict_items", [])),
                "pressure_contribution": contrib,
            },
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="attention_conflict",
            recurrence_bonus=0.2 if contrib >= 0.25 else 0.05,
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    elif float(attn_md.get("pressure_contribution") or 0.0) >= 0.25:
        c = float(attn_md.get("pressure_contribution") or 0.0)
        cand = _mk_candidate(
            interrupt_type="attention_conflict",
            source="attention",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="attention_repeated_pressure",
            pressure=_clamp01(c + 0.2),
            confidence=confidence_default,
            novelty=novelty,
            payload={"pressure_contribution": c},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="attention_pressure_repeat",
            recurrence_bonus=0.15,
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    ctx_overloaded = bool(signal_map_dict.get("context_overloaded"))
    sys_press = float(signal_map_dict.get("system_pressure") or 0.0)
    if sys_press >= SYSTEM_PRESSURE_INTERRUPT_THRESHOLD:
        cand = _mk_candidate(
            interrupt_type="system_pressure",
            source="nexus",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="system_pressure_threshold",
            pressure=sys_press,
            confidence=0.55,
            novelty=novelty,
            payload={"system_pressure": sys_press},
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="system_pressure",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))
    elif ctx_overloaded and evt_pressure >= 0.55:
        cand = _mk_candidate(
            interrupt_type="system_pressure",
            source="nexus",
            source_id=event.event_id,
            parent_event_id=event.event_id,
            reason="context_overloaded_with_pressure",
            pressure=max(evt_pressure, 0.5),
            confidence=0.5,
            novelty=novelty,
            payload={
                "context_overloaded": True,
                "event_pressure": evt_pressure,
            },
            suppressible=True,
            requires_user_attention=False,
            cooldown_hint="nexus_ctx_overload_pressure",
        )
        candidates.append(score_interrupt_priority(cand=cand, raw=cand.metadata))

    return candidates


def _is_safety_critical(cand: InterruptCandidate) -> bool:
    if cand.interrupt_type == "verifier_escalation" and not cand.suppressible:
        return True
    if cand.interrupt_type == "policy_block":
        return True
    return False


def apply_suppression(
    *,
    scored_sorted: Sequence[InterruptCandidate],
    cycle_duplicate_keys: set[str],
    context_overloaded: bool,
    cooldowns: Mapping[str, Any],
    current_event_count: int,
) -> tuple[list[SuppressionDecision], InterruptCandidate | None, list[str]]:
    """
    Returns decisions, winning candidate (first allowed internal), suppressed summaries.
    Mutates cycle_duplicate_keys for duplicate detection across calls in the same cycle.
    """
    decisions: list[SuppressionDecision] = []
    suppressed_summary: list[str] = []

    dup_scope = cycle_duplicate_keys
    ordered = sorted(scored_sorted, key=lambda c: c.priority, reverse=True)[
        :MAX_INTERRUPT_QUEUE
    ]
    winner: InterruptCandidate | None = None

    def make_decision(
        cand: InterruptCandidate,
        *,
        final_suppressed: bool,
        suppression_primary: str,
        rules: list[str],
        reasons: list[str],
        cooldown_applied: bool,
        dk: str,
        prev_count: int,
        priority_before: float,
        priority_after: float,
        allowed_internal: bool,
    ) -> SuppressionDecision:
        ae = allowed_internal and not final_suppressed
        return SuppressionDecision(
            candidate_id=cand.id,
            suppressed=bool(final_suppressed),
            suppression_reason=(
                suppression_primary if final_suppressed else ("none" if ae else "blocked")
            ),
            suppression_rules_triggered=list(rules),
            cooldown_applied=bool(cooldown_applied),
            cooldown_key=dk,
            previous_count=max(0, int(prev_count)),
            priority_before=_clamp01(priority_before),
            priority_after=_clamp01(priority_after),
            allowed_internal_event=bool(ae),
            allowed_user_expression=False,
            reasons=list(reasons),
        )

    for cand in ordered:
        rules: list[str] = []
        reasons: list[str] = []
        priority_before = _clamp01(cand.priority)
        priority_after = priority_before
        dk = cand.cooldown_key
        dedup_signature = f"{cand.source}|{cand.interrupt_type}|{dk}"
        if dedup_signature in dup_scope:
            decisions.append(
                make_decision(
                    cand,
                    final_suppressed=True,
                    suppression_primary="duplicate_same_cycle",
                    rules=rules + ["B_duplicate_same_cycle"],
                    reasons=["duplicate_equivalent_candidate"],
                    cooldown_applied=False,
                    dk=dk,
                    prev_count=0,
                    priority_before=priority_before,
                    priority_after=priority_after,
                    allowed_internal=False,
                )
            )
            suppressed_summary.append(f"{cand.id}:duplicate_same_cycle")
            dup_scope.add(dedup_signature)
            continue
        dup_scope.add(dedup_signature)

        cd_entry_raw = cooldowns.get(dk) if isinstance(cooldowns, dict) else None
        cd_entry = cd_entry_raw if isinstance(cd_entry_raw, dict) else {}
        prev_count = int(cd_entry.get("trigger_count") or 0)
        cooldown_applied = False
        lte_raw = cd_entry.get("last_triggered_event_count")
        try:
            lte = (
                int(lte_raw)
                if lte_raw is not None and not isinstance(lte_raw, bool)
                else None
            )
        except (TypeError, ValueError):
            lte = None

        if (
            lte is not None
            and lte > 0
            and (current_event_count - lte) <= COOLDOWN_RECENT_EVENT_WINDOW
        ):
            cooldown_applied = True
            if priority_before < COOLDOWN_BYPASS_PRIORITY:
                decisions.append(
                    make_decision(
                        cand,
                        final_suppressed=True,
                        suppression_primary="cooldown_throttle_below_bypass",
                        rules=rules + ["A_cooldown_throttle"],
                        reasons=[f"cooldown_key_recent:{dk}"],
                        cooldown_applied=True,
                        dk=dk,
                        prev_count=prev_count,
                        priority_before=priority_before,
                        priority_after=priority_after,
                        allowed_internal=False,
                    )
                )
                suppressed_summary.append(f"{cand.id}:cooldown")
                continue
            priority_after = _clamp01(priority_before - COOLDOWN_PRIORITY_PENALTY)
            rules.append("A_cooldown_penalty_above_bypass")
            reasons.append(f"cooldown_penalty:{COOLDOWN_PRIORITY_PENALTY}")

        if winner is not None:
            decisions.append(
                make_decision(
                    cand,
                    final_suppressed=True,
                    suppression_primary="max_one_internal_per_cycle",
                    rules=rules + ["H_max_one_internal_event"],
                    reasons=["winner_already_selected"],
                    cooldown_applied=cooldown_applied,
                    dk=dk,
                    prev_count=prev_count,
                    priority_before=priority_before,
                    priority_after=priority_after,
                    allowed_internal=False,
                )
            )
            suppressed_summary.append(f"{cand.id}:cap")
            continue

        suppressed = False
        suppression_primary = ""

        if priority_after < LOW_PRIORITY_THRESHOLD:
            suppressed = True
            suppression_primary = "low_priority"
            rules.append("C_low_priority")
            reasons.append("priority_below_threshold")

        if (
            not suppressed
            and context_overloaded
            and not _is_safety_critical(cand)
        ):
            priority_after = _clamp01(priority_after - CONTEXT_OVERLOAD_PRIORITY_PENALTY)
            rules.append("D_context_overload_downgrade")
            reasons.append("context_overloaded_downgrade")
            if priority_after < LOW_PRIORITY_THRESHOLD:
                suppressed = True
                suppression_primary = "context_overload_suppress"
                rules.append("D_context_overload_suppress")

        if not suppressed and cand.metadata.get(
            "implies_act_execution_denied_by_policy"
        ):
            suppressed = True
            suppression_primary = "policy_denied_execution_path"
            rules.append("E_policy_act_denied_internal_suppressed")

        rules.append("F_user_expression_blocked_v2_always")
        reasons.append("user_expression_future_expression_gate_only")
        if cand.requires_user_attention:
            rules.append("F_requires_user_attention_ok_internal")

        if suppressed:
            decisions.append(
                make_decision(
                    cand,
                    final_suppressed=True,
                    suppression_primary=suppression_primary,
                    rules=rules,
                    reasons=reasons,
                    cooldown_applied=cooldown_applied,
                    dk=dk,
                    prev_count=prev_count,
                    priority_before=priority_before,
                    priority_after=priority_after,
                    allowed_internal=False,
                )
            )
            suppressed_summary.append(f"{cand.id}:{suppression_primary}")
            continue

        winner = cand
        decisions.append(
            make_decision(
                cand,
                final_suppressed=False,
                suppression_primary="none",
                rules=rules,
                reasons=reasons,
                cooldown_applied=cooldown_applied,
                dk=dk,
                prev_count=prev_count,
                priority_before=priority_before,
                priority_after=priority_after,
                allowed_internal=True,
            )
        )

    return decisions, winner, suppressed_summary


def build_nexus_internal_event(parent: Event, cand: InterruptCandidate) -> Event:
    bounded_payload = {
        k: cand.payload.get(k)
        for k in (
            "latent_pressure_total",
            "ignition_entry_id",
            "ignition_reason",
            "policy_status",
        )
        if k in cand.payload
    }
    return Event(
        event_type=EventType.INTERNAL,
        content="nexus_interrupt",
        metadata={
            "interrupt_candidate_id": cand.id,
            "interrupt_type": cand.interrupt_type,
            "source": cand.source,
            "reason": cand.reason,
            "parent_event_id": cand.parent_event_id,
            "priority": _clamp01(cand.priority),
            "pressure": _clamp01(cand.pressure),
            "payload": _json_safe(bounded_payload),
        },
    )


def serialization_roundtrip(candidate_dict: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(candidate_dict, sort_keys=True)
    return json.loads(raw)


def update_cooldown_entry(
    *,
    cooldown_key: str,
    candidate_id: str,
    reason: str,
    event_count: int,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prev = dict(prior) if isinstance(prior, dict) else {}
    triggers = int(prev.get("trigger_count") or 0) + 1
    return {
        "cooldown_key": cooldown_key,
        "last_triggered_at": datetime.now(timezone.utc).isoformat(),
        "trigger_count": triggers,
        "last_candidate_id": candidate_id,
        "last_reason": str(reason),
        "last_triggered_event_count": int(event_count),
    }
