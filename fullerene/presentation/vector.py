"""Derive Presentation Vector v0 from Nexus records (read-only; no side effects)."""

from __future__ import annotations

from typing import Any

from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusState,
)
from fullerene.presentation.mapping import (
    animation_hint_for,
    eye_state_for,
    face_state_for,
    mouth_state_for,
    motion_for_mode,
)
from fullerene.presentation.models import (
    PresentationChannel,
    PresentationMode,
    PresentationVector,
    clamp01,
)


def _coerce_record(record: NexusRecord | dict[str, Any]) -> NexusRecord:
    if isinstance(record, NexusRecord):
        return record
    return NexusRecord.from_dict(record)


def _verifier_scan(facet_results: list[FacetResult]) -> tuple[bool, bool, float]:
    """Returns (critical_like_failure, verifyingSignals, verifier_severity_score)."""
    critical = False
    verifying = False
    worst_sev = 0.0

    severity_unit = {"info": 0.15, "warning": 0.55, "error": 0.85, "critical": 1.0}

    for fr in facet_results:
        if fr.facet_name != "verifier":
            continue
        md = fr.metadata if isinstance(fr.metadata, dict) else {}
        if md.get("retry_recommended"):
            verifying = True
        if md.get("escalation_recommended"):
            verifying = True
        if str(md.get("verification_status") or "").lower() == "failed":
            verifying = True
        for key in ("artifact_checks", "schema_checks"):
            rows = md.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                st = str(row.get("status") or "").lower()
                sev = str(row.get("severity") or "").lower()
                worst_sev = max(worst_sev, severity_unit.get(sev, 0.25))
                if st == "failed" and sev in {"critical", "error"}:
                    critical = True
    return critical, verifying, clamp01(worst_sev)


def _interrupt_priority_score(
    *,
    md: dict[str, Any],
) -> tuple[float, bool]:
    """Max priority score from candidates plus whether any unresolved candidate exists."""
    cands = md.get("interrupt_candidates")
    scores: list[float] = []
    has_any = False
    any_unsuppressed = False
    if isinstance(cands, list):
        for c in cands:
            if not isinstance(c, dict):
                continue
            has_any = True
            pri = clamp01(c.get("priority", 0.0))
            scores.append(pri)
            cid = str(c.get("id") or "")
            sup = md.get("suppression_decisions")
            suppressed = False
            if isinstance(sup, list) and cid:
                for sd in sup:
                    if (
                        isinstance(sd, dict)
                        and str(sd.get("candidate_id")) == cid
                        and sd.get("suppressed") is True
                    ):
                        suppressed = True
                        break
            if not suppressed:
                any_unsuppressed = True
    score = max(scores) if scores else 0.0
    allowed = md.get("allowed_interrupt_candidate")
    if isinstance(allowed, dict):
        score = max(score, clamp01(allowed.get("priority", 0.0)))
        has_any = True
        any_unsuppressed = True
    return clamp01(score), bool(has_any and any_unsuppressed)


def _learning_active(facet_results: list[FacetResult]) -> tuple[bool, bool]:
    applied = False
    routes = False
    for fr in facet_results:
        if fr.facet_name != "learning":
            continue
        md = fr.metadata if isinstance(fr.metadata, dict) else {}
        lm = md.get("learning_result")
        if isinstance(lm, dict):
            app = lm.get("applied")
            meta = lm.get("metadata") if isinstance(lm.get("metadata"), dict) else {}
            if isinstance(app, list) and len(app) > 0:
                applied = True
            cfr = meta.get("cross_facet_routes")
            if isinstance(cfr, list) and len(cfr) > 0:
                routes = True
        raw_applied = md.get("applied")
        if isinstance(raw_applied, list) and len(raw_applied) > 0:
            applied = True
    return applied, routes


def _behavior_policy_block(trace: dict[str, Any] | None, sig: dict[str, Any]) -> bool:
    if sig.get("policy_blocks_act") is True:
        return True
    if isinstance(trace, dict):
        pr = str(trace.get("policy_result") or "").lower()
        if pr == "denied":
            return True
    return False


def _attention_conflict(state: NexusState | None) -> tuple[bool, float]:
    if state is None:
        return False, 0.0
    bucket = state.facet_state.get("attention")
    if not isinstance(bucket, dict):
        return False, 0.0
    conflict = bucket.get("last_attention_conflict") or bucket.get("attention_conflict")
    if isinstance(conflict, dict) and conflict:
        delta = conflict.get("score_delta")
        try:
            d = clamp01(delta) if delta is not None else 1.0
        except Exception:  # noqa: BLE001
            d = 1.0
        return True, clamp01(0.65 + d * 0.35)
    broadcast = bucket.get("last_attention_broadcast")
    contrib = 0.0
    if isinstance(broadcast, dict):
        raw = broadcast.get("repeated_pressure_contribution") or broadcast.get(
            "pressure_contribution",
        )
        contrib = clamp01(raw) if raw is not None else 0.0
    bc_score = clamp01((broadcast or {}).get("score", 0.0)) if isinstance(broadcast, dict) else 0.0
    motion_hint = clamp01(max(contrib, bc_score))
    return False, motion_hint


def _affect_confidence_novelty(
    facet_results: list[FacetResult],
    state: NexusState | None,
) -> tuple[float, float]:
    conf = 0.5
    nov = 0.5

    # Prefer persisted affect facet state snapshot
    bucket = (
        state.facet_state.get("affect") if isinstance(state, NexusState) else None,
    )
    if isinstance(bucket, dict):
        st = bucket.get("last_affect_state") or bucket.get("affect_state")
        if isinstance(st, dict):
            conf = clamp01(st.get("dominance", st.get("arousal", conf)))
            nov = clamp01(st.get("novelty", nov))
    # Facet results may expose affect metadata
    for fr in facet_results:
        if fr.facet_name != "affect":
            continue
        md = fr.metadata if isinstance(fr.metadata, dict) else {}
        mv = md.get("affect_state") or md.get("state_vector")
        if isinstance(mv, dict):
            conf = clamp01(mv.get("dominance", mv.get("arousal", conf)))
            nov = clamp01(mv.get("novelty", nov))
    return conf, nov


def _gather(
    record: NexusRecord,
    state: NexusState | None,
) -> dict[str, Any]:
    md_raw = record.metadata if isinstance(record.metadata, dict) else {}
    md = dict(md_raw)
    sig = md.get("signal_map") if isinstance(md.get("signal_map"), dict) else {}

    evt = record.event
    evt_type_str = evt.event_type.value if isinstance(evt, Event) else str(
        evt.get("event_type", "unknown"),
    )
    evt_meta = evt.metadata if isinstance(evt, Event) else (
        evt.get("metadata") if isinstance(evt, dict) else {},
    )
    if not isinstance(evt_meta, dict):
        evt_meta = {}

    latent_total = clamp01(md.get("latent_pressure") or sig.get("latent_pressure") or 0.0)
    lpr = md.get("latent_pressure_result")
    if isinstance(lpr, dict) and latent_total <= 0.0:
        latent_total = clamp01(lpr.get("latent_pressure_total", 0.0))

    lpb_ignite = isinstance(lpr, dict) and bool(lpr.get("ignition_recommended"))

    facet_results = list(record.facet_results)
    verifier_critical, verifier_verify, verifier_severity = _verifier_scan(facet_results)
    ip_score, interrupt_hot = _interrupt_priority_score(md=md)
    interrupt_list = md.get("interrupt_candidates")
    has_any_interrupt = isinstance(interrupt_list, list) and len(interrupt_list) > 0
    attn_conflict, attn_motion = _attention_conflict(state)

    expr = md.get("expression_recommendation")
    if not isinstance(expr, dict):
        expr = {}
    expr_mode = str(
        md.get("expression_mode") or expr.get("mode") or "silent",
    ).strip().lower()
    expr_suppressed = bool(md.get("expression_suppressed") or expr.get("suppressed"))
    expr_score = clamp01(md.get("expression_score") or expr.get("expression_score") or 0.0)
    expr_intent = str(expr.get("suggested_intent") or "none").strip().lower()

    trace = None
    for fr in facet_results:
        if fr.facet_name == "behavior":
            tr = fr.metadata.get("decision_trace") if isinstance(fr.metadata, dict) else None
            if isinstance(tr, dict):
                trace = tr
                break

    ps = str(sig.get("policy_status") or "unknown").lower()
    approval_required = bool(sig.get("policy_requires_approval")) or ps == "approval_required"
    denied_explicit = ps == "denied"
    behavior_policy_blocked = _behavior_policy_block(trace, sig)

    learning_applied, learning_routes = _learning_active(facet_results)
    confidence, novelty = _affect_confidence_novelty(facet_results, state)

    system_pressure = clamp01(sig.get("system_pressure") or md.get("system_pressure") or 0.0)
    overloaded = bool(sig.get("context_overloaded")) or system_pressure >= 0.85

    sleeping_hints = (
        evt_meta.get("sleeping"),
        evt_meta.get("sleep"),
        evt_meta.get("presentation_sleep"),
    )
    sleeping = any(bool(x) for x in sleeping_hints)
    sleeping = sleeping or evt_type_str.lower() == "sleep_signal"

    ctx = {
        "md": md,
        "signal_map": sig,
        "facet_results": facet_results,
        "event_type": evt_type_str,
        "is_system_tick": evt_type_str.lower() == "system_tick",
        "manual_tick": evt_meta.get("manual_tick"),
        "expr": expr,
        "expr_mode": expr_mode,
        "expr_suppressed": expr_suppressed,
        "expr_score": expr_score,
        "expr_intent": expr_intent,
        "verifier_critical": verifier_critical,
        "verifier_verifying": verifier_verify,
        "verifier_severity": verifier_severity,
        "interrupt_priority": ip_score,
        "interrupt_hot": interrupt_hot,
        "has_any_interrupt": has_any_interrupt,
        "attention_conflict": attn_conflict,
        "attention_motion": attn_motion,
        "latent_pressure": latent_total,
        "lpb_ignition": lpb_ignite,
        "system_pressure": system_pressure,
        "pressure_for_display": system_pressure,
        "context_overloaded": bool(sig.get("context_overloaded")),
        "overload_signal": overloaded,
        "approval_required": approval_required,
        "behavior_policy_blocked": behavior_policy_blocked,
        "denied_explicit": denied_explicit,
        "learning_applied": learning_applied,
        "learning_routes": learning_routes,
        "confidence": confidence,
        "novelty": novelty,
        "sleeping_hint": sleeping,
        "behavior_trace": trace or {},
        "intent_ask_approval": expr_intent == "ask_approval",
        "surface_warning": expr_intent == "surface_warning",
        "sleeping_explicit": evt_meta.get("presentation_hint") == "sleeping",
    }
    ctx["sleeping_explicit"] = bool(ctx["sleeping_explicit"] or evt_meta.get("sleep"))
    ctx["sleeping_hint"] = bool(ctx["sleeping_hint"] or ctx["sleeping_explicit"])
    return ctx


def _channel_for(mode: PresentationMode, ctx: dict[str, Any]) -> PresentationChannel:
    expr_mode = ctx["expr_mode"]
    suppressed = ctx["expr_suppressed"]
    sev_warning = ctx["verifier_critical"] or ctx["surface_warning"]

    if mode == PresentationMode.warning or sev_warning:
        return PresentationChannel.warning

    if mode == PresentationMode.blocked and expr_mode == "ask_user" and not suppressed:
        return PresentationChannel.ask_user

    mapping = {
        "silent": PresentationChannel.internal,
        "log_only": PresentationChannel.internal,
        "status_only": PresentationChannel.status,
        "short_utterance": PresentationChannel.user_expression,
        "ask_user": PresentationChannel.ask_user,
    }
    ch = mapping.get(expr_mode, PresentationChannel.internal)
    if suppressed:
        # Suppressed: downgrade user-facing channels
        if expr_mode == "status_only":
            return PresentationChannel.status if ctx["verifier_severity"] >= 0.65 else PresentationChannel.internal
        if ch in (PresentationChannel.user_expression, PresentationChannel.ask_user):
            return (
                PresentationChannel.status
                if ctx["surface_warning"]
                else PresentationChannel.internal
            )
    return ch


def _select_mode(ctx: dict[str, Any]) -> tuple[PresentationMode, list[str]]:
    reasons: list[str] = []

    sleeping = ctx["sleeping_hint"] or (
        ctx["sleeping_explicit"] and ctx["system_pressure"] < 0.1
    )
    if sleeping and ctx["expr_mode"] in {"silent", "log_only"} and not ctx["verifier_critical"]:
        reasons.append("priority:sleeping")
        return PresentationMode.sleeping, reasons

    expr_user_facing_ok = ctx["expr_mode"] in {"short_utterance", "ask_user"} and not ctx["expr_suppressed"]
    asks_approval = ctx["intent_ask_approval"] and ctx["expr_mode"] == "ask_user"

    speaking_ok = expr_user_facing_ok and (
        ctx["expr_mode"] == "short_utterance" or (
            ctx["expr_mode"] == "ask_user" and not asks_approval
        )
    )

    # Priority stack (inspectable reasons)
    # 2 warning — verifier critical or surface_warning intent beats most work modes
    if ctx["verifier_critical"] or ctx["surface_warning"]:
        tag = (
            "verifier_critical_escalation"
            if ctx["verifier_critical"]
            else "expression_surface_warning"
        )
        reasons.append(f"priority:warning:{tag}")
        return PresentationMode.warning, reasons

    # 3 blocked — policy gates and explicit approval intents
    if asks_approval:
        reasons.append("priority:blocked:expression_intent_ask_approval")
        return PresentationMode.blocked, reasons

    if ctx["approval_required"] or ctx["denied_explicit"]:
        reasons.append("priority:blocked:policy_denied_or_approval_required")
        return PresentationMode.blocked, reasons

    if ctx["behavior_policy_blocked"]:
        reasons.append("priority:blocked:behavior_policy_signal")
        return PresentationMode.blocked, reasons

    # 1 speaking (after safety gates above)
    if speaking_ok:
        reasons.append("priority:speaking:expression_gate_user_facing")
        return PresentationMode.speaking, reasons

    if ctx["overload_signal"]:
        reasons.append("priority:overloaded:context_or_pressure")
        return PresentationMode.overloaded, reasons

    verifier_noncritical_signals = ctx["verifier_verifying"] and not ctx["verifier_critical"]
    if verifier_noncritical_signals:
        reasons.append("priority:verifying:retry_or_escalation_recommendations")
        return PresentationMode.verifying, reasons

    if ctx["learning_applied"] or ctx["learning_routes"]:
        lr = []
        if ctx["learning_applied"]:
            lr.append("applied_adjustments")
        if ctx["learning_routes"]:
            lr.append("cross_facet_route")
        reasons.append(f"priority:learning:{','.join(lr)}")
        return PresentationMode.learning, reasons

    think_trigger = ctx["lpb_ignition"] or (
        ctx["latent_pressure"] >= 0.35
        or ctx["system_pressure"] >= 0.35
        or (ctx["has_any_interrupt"] and ctx["interrupt_hot"])
        or ctx["attention_conflict"]
        or ctx["is_system_tick"]
    )
    if think_trigger:
        reasons.append(
            "priority:thinking:"
            + (
                ",".join(
                    p
                    for p, ok in (
                        ("lpb_ignite", ctx["lpb_ignition"]),
                        ("pressure_latent_lp", ctx["latent_pressure"] >= 0.35),
                        ("pressure_system", ctx["system_pressure"] >= 0.35),
                        ("interrupt", ctx["has_any_interrupt"] and ctx["interrupt_hot"]),
                        ("attention_conflict", ctx["attention_conflict"]),
                        ("system_tick", ctx["is_system_tick"]),
                    )
                    if ok
                )
                or "signal"
            ),
        )
        return PresentationMode.thinking, reasons

    if ctx["event_type"].lower() == "user_message":
        reasons.append("priority:listening:user_message_fallback")
        return PresentationMode.listening, reasons

    low_energy = ctx["latent_pressure"] < 0.35 and ctx["system_pressure"] < 0.35 and not expr_user_facing_ok
    idle_interrupt_quiet = (not ctx["has_any_interrupt"]) or (not ctx["interrupt_hot"])
    if low_energy and idle_interrupt_quiet and not ctx["context_overloaded"]:
        reasons.append("priority:idle:low_signals")
        return PresentationMode.idle, reasons

    reasons.append("priority:unknown:fallback")
    return PresentationMode.unknown, reasons


def _floor_intensity_for_mode(mode: PresentationMode, intensity: float) -> float:
    x = clamp01(intensity)
    caps = {
        PresentationMode.idle: 0.25,
    }
    floors = {
        PresentationMode.warning: 0.75,
        PresentationMode.overloaded: 0.70,
        PresentationMode.blocked: 0.55,
        PresentationMode.speaking: 0.60,
    }
    if mode == PresentationMode.idle:
        x = min(x, caps.get(PresentationMode.idle, 1.0))
    x = max(x, floors.get(mode, 0.0))
    return clamp01(x)


def _score_intensity(
    ctx: dict[str, Any],
    mode: PresentationMode,
    reasons: list[str],
) -> tuple[float, dict[str, Any]]:
    pressure = clamp01(ctx["system_pressure"])
    latent = clamp01(ctx["latent_pressure"])
    expr = clamp01(ctx["expr_score"])
    intra = clamp01(ctx["interrupt_priority"])
    verifier_s = clamp01(ctx["verifier_severity"])

    raw = (
        pressure * 0.35
        + latent * 0.25
        + expr * 0.20
        + intra * 0.10
        + verifier_s * 0.10
    )
    intens = clamp01(raw)
    intens = _floor_intensity_for_mode(mode, intens)

    components = {
        "pressure_component": clamp01(pressure * 0.35),
        "latent_pressure_component": clamp01(latent * 0.25),
        "expression_component": clamp01(expr * 0.20),
        "interrupt_priority_component": clamp01(intra * 0.10),
        "verifier_severity_component": clamp01(verifier_s * 0.10),
        "raw_linear_sum": clamp01(raw),
        "floors_caps_applied": True,
        "presentation_mode": mode.value,
    }
    return intens, components


def _user_attention_needed(mode: PresentationMode, ctx: dict[str, Any]) -> bool:
    expr_ok_user = ctx["expr_mode"] == "ask_user" and not ctx["expr_suppressed"]
    if mode == PresentationMode.warning:
        return True
    if expr_ok_user:
        return True
    if ctx["approval_required"]:
        return True
    if ctx["verifier_critical"]:
        return True
    return False


def _expression_active(ctx: dict[str, Any]) -> bool:
    if ctx["expr_suppressed"]:
        return False
    return ctx["expr_mode"] in {"short_utterance", "ask_user"}


def derive_presentation_vector(
    record: NexusRecord | dict[str, Any],
    state: NexusState | dict[str, Any] | None = None,
) -> PresentationVector:
    """Build a deterministic Presentation Vector from ``record`` (+ optional facet_state via ``state``)."""
    nexus_record = _coerce_record(record)
    nx_state = None
    if isinstance(state, NexusState):
        nx_state = state
    elif isinstance(state, dict):
        nx_state = NexusState.from_dict(state)

    ctx = _gather(nexus_record, nx_state)

    evt = nexus_record.event
    if isinstance(evt, Event):
        eid = evt.event_id
        etype = evt.event_type.value
        emeta = evt.metadata if isinstance(evt.metadata, dict) else {}
    else:
        eid = str((evt if isinstance(evt, dict) else {}).get("event_id", ""))
        etype = str((evt if isinstance(evt, dict) else {}).get("event_type", "unknown"))
        raw_e = evt if isinstance(evt, dict) else {}
        emeta = raw_e.get("metadata") if isinstance(raw_e, dict) else {}
        if not isinstance(emeta, dict):
            emeta = {}

    mode, rsn = _select_mode(ctx)
    reasons = list(rsn)
    intensity, comp_meta = _score_intensity(ctx, mode, reasons)

    is_tick = ctx["is_system_tick"]
    motion = motion_for_mode(mode, is_system_tick=is_tick, intensity=intensity)
    face = face_state_for(mode)
    eyes = eye_state_for(mode, motion)
    mouth = mouth_state_for(mode, motion)
    anim = animation_hint_for(
        mode,
        motion,
        is_system_tick=is_tick,
        intensity=intensity,
    )

    # attention_motion combine signal map + facet attention
    sig = ctx["signal_map"]
    am = clamp01(
        max(
            clamp01(sig.get("attention_pressure", 0.0)),
            ctx["attention_motion"],
        ),
    )

    chan = _channel_for(mode, ctx)

    primary_reason = reasons[-1] if reasons else mode.value

    meta_out: dict[str, Any] = {
        **comp_meta,
        "priority_reasons": list(reasons),
        "presentation_version": "v0",
        "suppress_expression_tick": (
            ctx["manual_tick"] is True and emeta.get("suppress_expression") is True
        ),
        "expression_suppressed": bool(ctx["expr_suppressed"]),
    }

    pv = PresentationVector(
        mode=mode,
        intensity=intensity,
        motion=motion,
        channel=chan,
        user_attention_needed=_user_attention_needed(mode, ctx),
        expression_active=_expression_active(ctx),
        expression_mode=ctx["expr_mode"],
        pressure=ctx["pressure_for_display"],
        latent_pressure=ctx["latent_pressure"],
        confidence=ctx["confidence"],
        novelty=ctx["novelty"],
        attention_motion=am,
        blocked=mode == PresentationMode.blocked,
        overloaded=mode == PresentationMode.overloaded,
        warning=mode == PresentationMode.warning,
        speaking=mode == PresentationMode.speaking,
        thinking=mode == PresentationMode.thinking,
        idle=mode == PresentationMode.idle,
        face_state=face,
        eye_state=eyes,
        mouth_state=mouth,
        animation_hint=anim,
        reason=primary_reason,
        reasons=reasons,
        source_event_id=eid,
        source_event_type=str(etype),
        source_record_id=nexus_record.record_id,
        metadata=meta_out,
    )
    return pv


def derive_presentation_vector_from_summary(summary: dict[str, Any]) -> PresentationVector:
    """
    Recover a Presentation Vector when a compact tick summary already carries ``presentation_vector``.
    Otherwise build a deterministic vector from summarized scalars embedded in a minimal ``NexusRecord``.
    """
    raw_pv = summary.get("presentation_vector")
    if isinstance(raw_pv, dict) and raw_pv.get("mode"):
        return PresentationVector.from_dict(raw_pv)

    pseudo_md: dict[str, Any] = {
        "signal_map": {
            "system_pressure": clamp01(summary.get("system_pressure")),
            "latent_pressure": clamp01(summary.get("latent_pressure")),
            "context_overloaded": False,
        },
        "latent_pressure": clamp01(summary.get("latent_pressure")),
        "expression_mode": summary.get("expression_mode"),
        "expression_suppressed": summary.get("expression_suppressed"),
    }
    pseudo_evt = Event(
        event_type=EventType.SYSTEM_TICK,
        content="",
        metadata={
            "tick_index": summary.get("tick_index"),
            "manual_tick": True,
            "suppress_expression": summary.get("expression_suppressed", True),
        },
    )
    rec = NexusRecord(
        event=pseudo_evt,
        facet_results=[],
        decision=NexusDecision(
            action=DecisionAction.RECORD,
            reason="presentation_tick_summary_stub",
        ),
        metadata=pseudo_md,
    )
    return derive_presentation_vector(rec, state=None)