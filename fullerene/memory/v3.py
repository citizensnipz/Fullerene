"""Memory v3 scoring: activation, pressure, retrieval bonuses, salience v3."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fullerene.memory.models import MemoryRecord


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def compute_salience_v3(
    base_salience: float,
    *,
    novelty: float = 0.0,
    arousal: float = 0.0,
    valence: float = 0.0,
    urgency_or_pressure: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Affect-influenced salience; modest deltas, does not demolish low base."""
    base = clamp01(base_salience)
    nv = clamp01(novelty)
    ar = clamp01(arousal)
    ve = clamp01(abs(float(valence)))
    urg = clamp01(urgency_or_pressure)
    delta = (
        nv * 0.10
        + ar * 0.10
        + ve * 0.05
        + urg * 0.10
    )
    delta_capped = min(delta, 0.20)
    out = clamp01(base + delta_capped)
    breakdown = {
        "base_salience": base,
        "novelty_term": round(nv * 0.10, 5),
        "arousal_term": round(ar * 0.10, 5),
        "valence_extreme_term": round(ve * 0.05, 5),
        "urgency_pressure_term": round(urg * 0.10, 5),
        "delta_total": round(delta, 5),
        "salience_version": "v3",
        "affect_salience_contribution": round(clamp01(out - base), 5),
    }
    return clamp01(out), breakdown


def compute_activation_score(
    *,
    retrieval_density: float,
    max_member_salience: float,
    average_member_relevance: float,
    attention_focus: float,
    unresolved_factor: float,
    recency_factor: float,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = (
        clamp01(retrieval_density) * 0.35
        + clamp01(max_member_salience) * 0.20
        + clamp01(average_member_relevance) * 0.20
        + clamp01(attention_focus) * 0.10
        + clamp01(unresolved_factor) * 0.10
        + clamp01(recency_factor) * 0.05
    )
    if retrieval_density >= 0.5:
        reasons.append("retrieval_density")
    if max_member_salience >= 0.6:
        reasons.append("high_salience_member")
    if attention_focus >= 0.5:
        reasons.append("attention_focus")
    if unresolved_factor >= 0.3:
        reasons.append("unresolved_signals")
    if recency_factor >= 0.3:
        reasons.append("recency")
    return clamp01(score), reasons


def compute_pressure_score(
    *,
    activation_score: float,
    unresolved_score: float,
    contradiction_factor: float,
    priority_goal_factor: float,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = (
        clamp01(activation_score) * 0.45
        + clamp01(unresolved_score) * 0.25
        + clamp01(contradiction_factor) * 0.20
        + clamp01(priority_goal_factor) * 0.10
    )
    if activation_score >= 0.35:
        reasons.append("activation")
    if unresolved_score >= 0.2:
        reasons.append("unresolved")
    if contradiction_factor >= 0.15:
        reasons.append("contradiction")
    if priority_goal_factor >= 0.15:
        reasons.append("priority_goal")
    return clamp01(score), reasons


def retrieval_density(
    retrieved_members: int,
    member_count: int,
    context_retrieval_limit: int,
) -> float:
    cap = max(1, min(int(member_count), int(context_retrieval_limit)))
    return clamp01(retrieved_members / cap)


def recency_activation_factor(
    *,
    last_activated_at: datetime | None,
    newest_member_created_at: datetime | None,
    now: datetime | None = None,
) -> float:
    current = now or datetime.now(timezone.utc)
    best = 0.0
    for dt in (last_activated_at, newest_member_created_at):
        if dt is None:
            continue
        age_days = max((current - dt).total_seconds() / 86400.0, 0.0)
        best = max(best, 1.0 / (1.0 + age_days))
    return clamp01(best)


def v3_retrieval_bonuses(
    *,
    community_activation: float,
    community_pressure: float,
    member_overlap_ratio: float,
    best_neighbor_weight_norm: float,
) -> dict[str, Any]:
    """Bounded bonuses added to hybrid v2 total (spec caps)."""
    act_b = min(0.10, clamp01(community_activation) * 0.10)
    press_b = min(0.10, clamp01(community_pressure) * 0.10)
    overlap_b = min(0.08, clamp01(member_overlap_ratio) * 0.08)
    neigh_b = min(0.05, clamp01(best_neighbor_weight_norm) * 0.05)
    total_bonus = act_b + press_b + overlap_b + neigh_b
    return {
        "community_activation_bonus": round(act_b, 6),
        "community_pressure_bonus": round(press_b, 6),
        "community_member_overlap_bonus": round(overlap_b, 6),
        "direct_neighbor_bonus": round(neigh_b, 6),
        "memory_v3_bonus_total": round(total_bonus, 6),
    }


def merge_v3_into_hybrid_breakdown(
    breakdown: dict[str, Any],
    v3_extras: dict[str, Any],
) -> dict[str, Any]:
    base_total = clamp01(float(breakdown.get("total", 0.0)))
    bonus = float(v3_extras.get("memory_v3_bonus_total", 0.0))
    merged = dict(breakdown)
    merged.update(v3_extras)
    merged["total_before_v3"] = round(base_total, 6)
    merged["total"] = round(clamp01(base_total + bonus), 6)
    merged["memory_v3"] = True
    return merged


def is_long_term_record(record: MemoryRecord) -> bool:
    from fullerene.memory.models import MemoryLayer

    return record.memory_layer == MemoryLayer.LONG_TERM
