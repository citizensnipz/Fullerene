"""Expression Gate v0 — recommends mode/intent/payload only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from fullerene.expression.models import (
    ExpressionBudgetState,
    ExpressionMode,
    ExpressionRecommendation,
    SuggestedIntent,
    _bounded_dict,
    _clamp01,
    max_expression_mode,
)
from fullerene.expression.scoring import ExpressionScoreComponents, compute_expression_score
from fullerene.nexus.models import DecisionAction, Event, EventType, FacetResult, NexusDecision


def _parse_ts(iso: str | None) -> datetime | None:
    if not iso or not isinstance(iso, str):
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_delta(a: datetime | None, b: datetime | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((b - a).total_seconds())


def _stable_hash(parts: Mapping[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _latest_facet_metadata(facet_results: list[FacetResult], name: str) -> dict[str, Any]:
    for result in reversed(facet_results):
        if result.facet_name != name:
            continue
        if isinstance(result.metadata, dict):
            return result.metadata
    return {}


def _interrupt_suppressed_candidate_ids(
    suppression_decisions: list[dict[str, Any]],
) -> set[str]:
    out: set[str] = set()
    for row in suppression_decisions:
        if not isinstance(row, dict):
            continue
        if row.get("suppressed") is True and row.get("candidate_id"):
            out.add(str(row["candidate_id"]))
    return out


def _verifier_escalation_retry(
    facet_results: list[FacetResult],
) -> tuple[float, bool, bool]:
    """Returns verifier component in [0,1], retry_only, verifier_critical_hint."""
    meta = _latest_facet_metadata(facet_results, "verifier")
    escalate = bool(meta.get("escalation_recommended"))
    retry = bool(meta.get("retry_recommended"))
    critical = meta.get("verification_status") == "failed"
    results = meta.get("results") if isinstance(meta.get("results"), list) else []
    artifact_critical = False
    for row in results:
        if not isinstance(row, dict):
            continue
        sev = str(row.get("severity") or "").lower()
        if sev == "critical":
            artifact_critical = True
            break
    achecks = meta.get("artifact_checks")
    if isinstance(achecks, list):
        for row in achecks:
            if not isinstance(row, dict):
                continue
            sev = str(row.get("severity") or "").lower()
            status = str(row.get("status") or "").lower()
            if status == "failed" and sev in {"critical", "error"}:
                artifact_critical = True
            if row.get("escalation_recommended"):
                escalate = True
    critical_escalation = escalate and (artifact_critical or critical)
    if critical_escalation:
        return 1.0, False, True
    if escalate:
        return 1.0, False, False
    if retry:
        return 0.5, True, False
    return 0.0, False, False


def _policy_attention(
    policy_meta: dict[str, Any],
    decision: NexusDecision,
) -> tuple[float, str]:
    st = str(policy_meta.get("policy_status") or "").lower()
    if st == "approval_required":
        return 1.0, "approval_required"
    if st == "denied" and decision.action in (DecisionAction.ACT, DecisionAction.ASK):
        return 0.8, "denied_user_relevant"
    return 0.0, ""


def _effective_novelty(
    event_meta: Mapping[str, Any],
    interrupt_cand: dict[str, Any] | None,
    attention_md: dict[str, Any],
) -> float:
    for key in ("novelty", "expression_novelty"):
        if key in event_meta:
            return _clamp01(event_meta.get(key))
    if interrupt_cand:
        return _clamp01(interrupt_cand.get("novelty"))
    peak = attention_md.get("scores")
    if isinstance(peak, dict) and peak:
        vals = [
            float(x)
            for x in peak.values()
            if isinstance(x, (int, float)) and not isinstance(x, bool)
        ]
        if vals:
            return _clamp01(max(vals))
    return 0.0


def _learning_route_novelty(learning_md: dict[str, Any]) -> float:
    routes = learning_md.get("cross_facet_routes")
    if not isinstance(routes, list) or not routes:
        return 0.0
    return _clamp01(min(1.0, 0.1 * float(len(routes))))


def _mode_from_threshold(score: float) -> ExpressionMode:
    if score < 0.45:
        return ExpressionMode.silent
    if score < 0.60:
        return ExpressionMode.log_only
    if score < 0.75:
        return ExpressionMode.status_only
    if score < 0.90:
        return ExpressionMode.short_utterance
    return ExpressionMode.ask_user


def evaluate_expression_gate(
    *,
    event: Event,
    decision: NexusDecision,
    facet_results: list[FacetResult],
    signal_map: Mapping[str, Any],
    latent_pressure_total: float,
    lpb_ignition_recommended: bool,
    interrupt_candidates: list[dict[str, Any]],
    allowed_interrupt_candidate: dict[str, Any] | None,
    suppression_decisions: list[dict[str, Any]],
    budget: ExpressionBudgetState,
    cycle_wall_time: datetime,
    cycle_seq: int,
) -> tuple[ExpressionRecommendation, ExpressionBudgetState]:
    meta = dict(event.metadata or {})
    policy_md = _latest_facet_metadata(facet_results, "policy")
    behavior_md = _latest_facet_metadata(facet_results, "behavior")
    attention_md = _latest_facet_metadata(facet_results, "attention")
    learning_md = _latest_facet_metadata(facet_results, "learning")

    budget_work = ExpressionBudgetState.from_dict(budget.to_dict())
    uf_gap0 = _seconds_delta(
        _parse_ts(budget_work.last_user_facing_at),
        cycle_wall_time,
    )
    if uf_gap0 is None or uf_gap0 >= float(budget_work.window_seconds):
        budget_work.expression_count_window = 0

    verifier_score, verifier_retry_only, verifier_critical_hint = (
        _verifier_escalation_retry(facet_results)
    )

    system_pressure = _clamp01(signal_map.get("system_pressure", 0.0))
    latent_pressure = _clamp01(latent_pressure_total)
    interrupt_priority = _clamp01((allowed_interrupt_candidate or {}).get("priority", 0.0))
    if allowed_interrupt_candidate is None and interrupt_candidates:
        top = max(
            interrupt_candidates,
            key=lambda c: float(c.get("priority") or 0.0),
        )
        interrupt_priority = max(interrupt_priority, _clamp01(top.get("priority", 0.0)))

    policy_need, policy_reason = _policy_attention(policy_md, decision)
    novelty = max(
        _effective_novelty(meta, allowed_interrupt_candidate, attention_md),
        _learning_route_novelty(learning_md),
    )
    confidence = _clamp01(behavior_md.get("confidence", 0.0) or 0.0)

    suppressed_ids = _interrupt_suppressed_candidate_ids(suppression_decisions)
    ignition_ids = {
        str(c.get("id"))
        for c in interrupt_candidates
        if isinstance(c, dict)
        and "ignition" in str(c.get("interrupt_type") or "")
        and c.get("id")
    }
    ignition_suppressed = bool(ignition_ids) and ignition_ids.issubset(suppressed_ids)

    repetition_penalty = 0.0
    intent_seed = {
        "etype": (
            str(allowed_interrupt_candidate.get("interrupt_type") or "")
            if allowed_interrupt_candidate
            else ""
        ),
        "policy": policy_reason,
        "lpb_ignition": lpb_ignition_recommended,
    }
    h_seed = _stable_hash(intent_seed)
    if budget_work.last_expression_hash and budget_work.last_expression_hash == h_seed:
        repetition_penalty = 0.12

    recent_expression_penalty = 0.0
    last_ts = _parse_ts(budget_work.last_expression_at)
    delta = _seconds_delta(last_ts, cycle_wall_time)
    if delta is not None and delta < float(budget_work.window_seconds):
        recent_expression_penalty = _clamp01(
            0.05 + (1.0 - delta / float(budget_work.window_seconds)) * 0.12,
        )

    denied_policy = policy_md.get("policy_status") == "denied"
    safety_escalation = bool(
        verifier_critical_hint
        or denied_policy
        or behavior_md.get("belief_contradiction"),
    )

    verifier_critical_escape = verifier_critical_hint and verifier_score >= 1.0
    critical_path = safety_escalation or verifier_critical_hint or policy_need >= 1.0

    context_overload_penalty = 0.0
    if (
        bool(signal_map.get("context_overloaded"))
        and not verifier_critical_escape
        and not critical_path
    ):
        context_overload_penalty = 0.10

    components = ExpressionScoreComponents(
        system_pressure=system_pressure,
        latent_pressure=latent_pressure,
        interrupt_priority=interrupt_priority,
        verifier_escalation=verifier_score,
        policy_attention_need=policy_need,
        novelty=novelty,
        confidence=confidence,
        repetition_penalty=repetition_penalty,
        recent_expression_penalty=recent_expression_penalty,
        context_overload_penalty=context_overload_penalty,
    )
    # Round to avoid float edge cases sitting just below mode thresholds.
    score = round(float(compute_expression_score(components)), 4)
    mode = _mode_from_threshold(score)

    if policy_need >= 1.0:
        mode = max_expression_mode(mode, ExpressionMode.ask_user)
    elif verifier_score >= 1.0 and not verifier_retry_only:
        mode = max_expression_mode(mode, ExpressionMode.short_utterance)
        if verifier_critical_hint or policy_need >= 0.8:
            mode = max_expression_mode(mode, ExpressionMode.ask_user)
    elif verifier_retry_only and verifier_score > 0:
        mode = max_expression_mode(mode, ExpressionMode.status_only)

    intent: SuggestedIntent = SuggestedIntent.none
    if policy_need >= 1.0:
        intent = SuggestedIntent.ask_approval
    elif verifier_score >= 1.0 and not verifier_retry_only:
        intent = SuggestedIntent.surface_warning
    elif verifier_retry_only:
        intent = (
            SuggestedIntent.ask_clarification
            if confidence < 0.45
            else SuggestedIntent.status_update
        )
    elif lpb_ignition_recommended and not ignition_suppressed:
        intent = SuggestedIntent.surface_unresolved_pressure
    elif confidence < 0.45 and float(behavior_md.get("ambiguity_score") or 0.0) > 0.35:
        intent = SuggestedIntent.ask_clarification
    elif allowed_interrupt_candidate:
        intent = SuggestedIntent.status_update
    elif mode != ExpressionMode.silent:
        intent = SuggestedIntent.status_update

    max_words = 0
    if mode == ExpressionMode.short_utterance:
        max_words = 40
    elif mode == ExpressionMode.ask_user:
        max_words = 120
    elif mode in (ExpressionMode.status_only, ExpressionMode.log_only):
        max_words = 80

    allowed_user_facing = mode in (
        ExpressionMode.short_utterance,
        ExpressionMode.ask_user,
    )
    requires_attention = intent in (
        SuggestedIntent.ask_approval,
        SuggestedIntent.ask_clarification,
        SuggestedIntent.surface_warning,
    )

    suppressed = False
    suppression_reason = ""
    rules: list[str] = []

    candidate_id = (
        str(allowed_interrupt_candidate["id"])
        if allowed_interrupt_candidate
        and isinstance(allowed_interrupt_candidate.get("id"), str)
        else None
    )

    def note_hard_suppress(rule: str, reason: str) -> None:
        nonlocal suppressed, suppression_reason, allowed_user_facing, mode, max_words, intent
        suppressed = True
        rules.append(rule)
        suppression_reason = reason
        allowed_user_facing = False
        mode = ExpressionMode.silent
        max_words = 0
        intent = SuggestedIntent.none

    def strip_user_facing(rule: str) -> None:
        nonlocal allowed_user_facing, mode, max_words, intent, suppressed, suppression_reason
        allowed_user_facing = False
        rules.append(rule)
        suppressed = False
        suppression_reason = ""
        if mode in (ExpressionMode.short_utterance, ExpressionMode.ask_user):
            mode = ExpressionMode.status_only
            max_words = 80

    if meta.get("suppress_expression") is True:
        note_hard_suppress("E_metadata_suppress", "event.metadata suppress_expression")
    elif meta.get("expression_mode"):
        try:
            forced = ExpressionMode(str(meta.get("expression_mode")))
            mode = forced
            allowed_user_facing = mode in (
                ExpressionMode.short_utterance,
                ExpressionMode.ask_user,
            )
            rules.append("E_metadata_expression_mode_override")
        except ValueError:
            pass

    if (
        event.event_type == EventType.INTERNAL
        and meta.get("allow_expression") is not True
        and not critical_path
        and allowed_user_facing
    ):
        strip_user_facing("E_internal_default")

    policy_outward_denied = bool(meta.get("policy_denied_outward_expression")) or bool(
        meta.get("suppress_outward_expression"),
    )
    if policy_outward_denied and allowed_user_facing:
        strip_user_facing("E_policy_outward")

    ignition_path_blocked = ignition_suppressed and (
        allowed_interrupt_candidate is None
        or "ignition" in str(allowed_interrupt_candidate.get("interrupt_type") or "")
    )
    if ignition_path_blocked and lpb_ignition_recommended:
        note_hard_suppress(
            "E_nexus_interrupt_suppressed",
            "latent pressure ignition interrupt suppressed by Nexus v2",
        )

    if (
        bool(signal_map.get("context_overloaded"))
        and not verifier_critical_escape
        and not critical_path
        and mode in (ExpressionMode.short_utterance, ExpressionMode.ask_user)
    ):
        mode = ExpressionMode.status_only
        allowed_user_facing = False
        max_words = 80
        rules.append("E_context_overload_non_critical")
        intent = SuggestedIntent.status_update

    clarification_or_approval = intent in (
        SuggestedIntent.ask_approval,
        SuggestedIntent.ask_clarification,
        SuggestedIntent.surface_warning,
    )
    if (
        confidence < 0.30
        and not clarification_or_approval
        and mode in (ExpressionMode.short_utterance, ExpressionMode.ask_user)
    ):
        mode = ExpressionMode.log_only
        allowed_user_facing = False
        max_words = 80
        rules.append("E_low_confidence_drop_user_facing")
        intent = SuggestedIntent.none

    cooldown_applied = False
    budget_applied = False

    # Candidate cooldown applies to repeated user-facing use of same source_candidate_id.
    if allowed_user_facing and candidate_id:
        ck = candidate_id
        prior_cd = budget_work.cooldowns.get(ck)
        prev_at = prior_cd.get("last_user_facing_at") if isinstance(prior_cd, dict) else None
        pt = _parse_ts(prev_at if isinstance(prev_at, str) else None)
        gap = _seconds_delta(pt, cycle_wall_time)
        cd_seconds = float(budget_work.window_seconds) + int(
            budget_work.ignored_expression_count or 0,
        ) * 30
        if gap is not None and gap < cd_seconds:
            cooldown_applied = True
            strip_user_facing("E_candidate_cooldown")

    if allowed_user_facing:
        uf_prev = _parse_ts(budget_work.last_user_facing_at)
        uf_gap = _seconds_delta(uf_prev, cycle_wall_time)
        if (
            uf_gap is not None
            and uf_gap < float(budget_work.window_seconds)
            and int(budget_work.expression_count_window or 0) >= 1
        ):
            budget_applied = True
            strip_user_facing("E_user_facing_budget")

    reco_hash = _stable_hash(
        {
            "intent": intent.value,
            "mode": mode.value,
            "candidate": candidate_id or "",
            "evt": event.event_id,
        },
    )

    rh_gap = _seconds_delta(_parse_ts(budget_work.last_expression_at), cycle_wall_time)
    if (
        allowed_user_facing
        and budget_work.last_expression_hash == reco_hash
        and rh_gap is not None
        and rh_gap < float(budget_work.window_seconds)
    ):
        strip_user_facing("E_repeated_expression_hash")

    cd = dict(components.to_dict())
    cd["expression_score"] = float(score)
    new_budget = ExpressionBudgetState.from_dict(budget_work.to_dict())
    new_budget.last_expression_hash = reco_hash
    new_budget.last_expression_at = cycle_wall_time.isoformat()

    payload = _bounded_dict(
        {
            "decision_action": decision.action.value,
            "interrupt_type": (allowed_interrupt_candidate or {}).get("interrupt_type"),
            "lpb_ignition_recommended": lpb_ignition_recommended,
            "latent_pressure_total": latent_pressure,
            "verifier_retry_only": verifier_retry_only,
            "cycle_seq": cycle_seq,
        },
    )

    reasons = [
        f"score={round(score, 4)}",
        f"mode={mode.value}",
        f"intent={intent.value}",
    ]
    if policy_reason:
        reasons.append(f"policy_signal={policy_reason}")

    reco = ExpressionRecommendation(
        mode=mode,
        expression_score=_clamp01(score),
        reasons=reasons,
        source_event_id=event.event_id,
        source_cycle_id=str(cycle_seq),
        source_candidate_id=candidate_id,
        allowed_user_facing=allowed_user_facing and not suppressed,
        requires_user_attention=requires_attention,
        cooldown_applied=cooldown_applied,
        budget_applied=budget_applied,
        suppressed=suppressed,
        suppression_reason=suppression_reason,
        max_words=max_words,
        suggested_intent=intent,
        payload=payload,
        metadata={
            "evaluated_at": cycle_wall_time.isoformat(),
            "score_components": cd,
            "ignition_suppressed": ignition_suppressed,
            "verifier_critical_hint": verifier_critical_hint,
        },
        suppression_rules_triggered=rules,
    )

    if meta.get("expression_ignored") is True:
        new_budget.ignored_expression_count = int(
            budget_work.ignored_expression_count or 0,
        ) + 1

    if allowed_user_facing and not suppressed:
        new_budget.expression_count_window = int(budget_work.expression_count_window or 0) + 1
        new_budget.last_user_facing_at = cycle_wall_time.isoformat()
        if candidate_id:
            entry = dict(new_budget.cooldowns.get(candidate_id) or {})
            entry["last_user_facing_at"] = cycle_wall_time.isoformat()
            new_budget.cooldowns[candidate_id] = entry

    hist = list(new_budget.history)
    hist.append(
        {
            "at": cycle_wall_time.isoformat(),
            "mode": mode.value,
            "score": round(float(cd.get("expression_score", score)), 4),
            "suppressed": suppressed,
            "intent": intent.value,
        },
    )
    new_budget.history = hist[-20:]

    return reco, new_budget
