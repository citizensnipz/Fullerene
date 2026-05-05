"""Learning v1 — deterministic cross-facet feedback routing (no TD, no meta, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullerene.goals import GoalStore
from fullerene.learning.models import (
    AdjustmentRecord,
    AdjustmentStatus,
    AdjustmentTarget,
    LearningSignal,
    SignalSource,
    SignalType,
)
from fullerene.memory import MemoryStore
from fullerene.memory.edges import MemoryEdgeType
from fullerene.nexus.models import CycleSignalMap, Event, NexusState
from fullerene.world_model.models import Belief, BeliefStatus
from fullerene.world_model.store import WorldModelStore

DEFAULT_HEBBIAN_LEARNING_RATE = 0.05
DEFAULT_CONFIDENCE_DELTA = 0.05
STRONG_CONTRADICTION_DELTA = 0.10
SALIENCE_NUDGE = 0.05
CO_EDGE_TYPE = MemoryEdgeType.KEYWORD_SIMILARITY


def _clamp_unit(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


def _nexus_bucket(state: NexusState) -> dict[str, Any]:
    raw = state.facet_state.get("nexus")
    return raw if isinstance(raw, dict) else {}


def read_cycle_signal_map(state: NexusState) -> CycleSignalMap:
    bucket = _nexus_bucket(state)
    raw = bucket.get("current_cycle_signal_map") or bucket.get("last_cycle_signal_map")
    if not isinstance(raw, dict):
        return CycleSignalMap()
    return CycleSignalMap.from_dict(raw)


def read_cycle_learning_events(state: NexusState) -> list[dict[str, Any]]:
    bucket = _nexus_bucket(state)
    raw = bucket.get("current_cycle_learning_events")
    if raw is None:
        raw = bucket.get("last_learning_events")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def read_behavior_trace(state: NexusState) -> dict[str, Any] | None:
    behavior = state.facet_state.get("behavior")
    if not isinstance(behavior, dict):
        return None
    trace = behavior.get("last_decision_trace")
    return trace if isinstance(trace, dict) else None


def _activation_for_memory(
    memory_id: str,
    *,
    score_map: dict[str, float],
    salience_map: dict[str, float],
) -> float:
    if memory_id in score_map:
        return _clamp_unit(score_map[memory_id])
    if memory_id in salience_map:
        return _clamp_unit(salience_map[memory_id])
    return 0.5


def _co_retrieval_memory_ids(state: NexusState) -> list[str]:
    ctx = state.facet_state.get("context")
    if isinstance(ctx, dict):
        ids = ctx.get("last_included_memory_ids")
        if isinstance(ids, list):
            return [str(x) for x in ids if isinstance(x, str) and x.strip()]
    mem = state.facet_state.get("memory")
    if isinstance(mem, dict):
        rel = mem.get("last_relevant_memory_ids")
        work = mem.get("last_working_memory_ids")
        out: list[str] = []
        if isinstance(rel, list):
            out.extend(str(x) for x in rel if isinstance(x, str) and x.strip())
        if isinstance(work, list):
            out.extend(str(x) for x in work if isinstance(x, str) and x.strip())
        # de-dupe preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for mid in out:
            if mid not in seen:
                seen.add(mid)
                unique.append(mid)
        return unique
    return []


def _memory_score_maps(state: NexusState) -> tuple[dict[str, float], dict[str, float]]:
    score_map: dict[str, float] = {}
    salience_map: dict[str, float] = {}
    mem = state.facet_state.get("memory")
    if not isinstance(mem, dict):
        return score_map, salience_map
    br = mem.get("memory_score_breakdowns")
    if isinstance(br, list):
        for item in br:
            if not isinstance(item, dict):
                continue
            mid = item.get("memory_id")
            if not isinstance(mid, str) or not mid.strip():
                continue
            total = item.get("total")
            if isinstance(total, (int, float)):
                score_map[mid] = float(total)
    for key in ("last_hybrid_scores", "last_memory_scores"):
        raw = mem.get(key)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    score_map.setdefault(k, float(v))
    return score_map, salience_map


def _fill_salience_from_store(
    memory_ids: list[str],
    memory_store: MemoryStore | None,
    salience_map: dict[str, float],
) -> None:
    if memory_store is None:
        return
    for mid in memory_ids:
        if mid in salience_map:
            continue
        if hasattr(memory_store, "get_memory"):
            rec = memory_store.get_memory(mid)
            if rec is not None:
                salience_map[mid] = float(rec.salience)


@dataclass(slots=True)
class LearningV1Bundle:
    signals: list[LearningSignal]
    adjustments: list[AdjustmentRecord]
    cross_facet_routes: list[dict[str, Any]]
    reasons: list[str]
    signal_sources: list[str]


def compute_learning_v1(
    event: Event,
    state: NexusState,
    *,
    memory_store: MemoryStore | None,
    goal_store: GoalStore | None,
    world_model_store: WorldModelStore | None,
    hebbian_rate: float = DEFAULT_HEBBIAN_LEARNING_RATE,
) -> LearningV1Bundle:
    """Produce v1-only adjustments, supplemental signals, and routing records."""
    del goal_store  # reserved for cross-facet goal routes
    ex = state.facet_state.get("executor")
    signals: list[LearningSignal] = []
    adjustments: list[AdjustmentRecord] = []
    routes: list[dict[str, Any]] = []
    reasons: list[str] = []
    sources: list[str] = []

    cmap = read_cycle_signal_map(state)
    learning_events = read_cycle_learning_events(state)
    if learning_events:
        sources.append("nexus_cycle_learning_events")
        reasons.append(f"consumed_{len(learning_events)}_learning_events")

    trace = read_behavior_trace(state)
    if trace is not None:
        sources.append("behavior_decision_trace_v2")

    # --- Behavior trace analysis → routes (proposal-only) ---
    if trace:
        raw_scores = trace.get("raw_candidate_scores")
        adj_scores = trace.get("adjusted_candidate_scores")
        policy_result = trace.get("policy_result")
        final_decision = trace.get("final_decision")
        if (
            isinstance(raw_scores, dict)
            and isinstance(adj_scores, dict)
            and raw_scores.get("act") is not None
            and adj_scores.get("act") is not None
            and float(adj_scores["act"]) < float(raw_scores["act"])
            and str(policy_result).lower()
            in {"denied", "approval_required", "no_match"}
        ):
            routes.append(
                {
                    "source": "behavior",
                    "signal": "policy_downgrade",
                    "target": "behavior",
                    "suggested_adjustment": "reinforce_policy_aware_act_suppression",
                    "strength": 0.05,
                    "applied": False,
                    "reason": "ACT score lowered from raw to adjusted under policy constraints",
                }
            )
            reasons.append("route_policy_downgrade")
        if trace.get("contradiction_flag") and final_decision == "ask":
            routes.append(
                {
                    "source": "behavior",
                    "signal": "contradiction_ask",
                    "target": "world_model",
                    "suggested_adjustment": "review_contradicted_beliefs",
                    "strength": 0.05,
                    "applied": False,
                    "reason": "contradiction bias led to ASK",
                }
            )
            reasons.append("route_contradiction_to_world_model")
        ctx_ratio = trace.get("context_load_ratio")
        overload = isinstance(ctx_ratio, (int, float)) and float(ctx_ratio) >= 0.85
        if overload and final_decision in {"record", "wait", "ask"}:
            routes.append(
                {
                    "source": "behavior",
                    "signal": "context_overload",
                    "target": "context",
                    "suggested_adjustment": "context_consolidation",
                    "strength": 0.05,
                    "applied": False,
                    "reason": "high context load ratio with non-ACT decision",
                }
            )
            reasons.append("route_context_consolidation")
        if trace.get("interrupt_recommended"):
            routes.append(
                {
                    "source": "behavior",
                    "signal": "latent_pressure_interrupt",
                    "target": "nexus",
                    "suggested_adjustment": "track_unresolved_latent_pressure",
                    "strength": 0.05,
                    "applied": False,
                    "reason": "interrupt recommended; flag for future LPB work",
                }
            )
            reasons.append("route_latent_pressure_signal")

    # --- World model: contradiction / corroboration ---
    behavior_state = state.facet_state.get("behavior")
    aligned_ids: list[str] = []
    if isinstance(behavior_state, dict):
        raw_ids = behavior_state.get("last_aligned_belief_ids")
        if isinstance(raw_ids, list):
            aligned_ids = [str(x) for x in raw_ids if isinstance(x, str) and x.strip()]

    meta = event.metadata if isinstance(event.metadata, dict) else {}
    corroborate_ids: list[str] = []
    raw_corrob = meta.get("belief_corroboration")
    if isinstance(raw_corrob, list):
        corroborate_ids = [str(x) for x in raw_corrob if isinstance(x, str) and x.strip()]

    contradiction_trace = bool(trace.get("contradiction_flag")) if trace else False
    contradiction_cycle = bool(cmap.belief_contradiction)
    strong_contradiction = contradiction_trace and contradiction_cycle

    if world_model_store is not None and aligned_ids and (
        contradiction_trace or contradiction_cycle
    ):
        sources.append("world_model_contradiction")
        delta = (
            STRONG_CONTRADICTION_DELTA if strong_contradiction else DEFAULT_CONFIDENCE_DELTA
        )
        for belief_id in aligned_ids:
            belief = world_model_store.get_belief(belief_id)
            if belief is None:
                continue
            old_c = float(belief.confidence)
            new_c = _clamp_unit(old_c - delta)
            if new_c == old_c:
                continue
            prov = {
                "source_event_id": event.event_id,
                "source_signal": "strong_belief_contradiction"
                if strong_contradiction
                else "belief_contradiction",
                "old_confidence": old_c,
                "new_confidence": new_c,
                "reason": "contradiction signal from cycle map and/or behavior trace",
            }
            merged_meta = dict(belief.metadata)
            merged_meta["learning_v1_provenance"] = prov
            merged_meta["contradiction_flagged_at"] = event.event_id
            belief.confidence = Belief._validate_confidence(new_c)
            belief.metadata = merged_meta
            if belief.status == BeliefStatus.ACTIVE:
                belief.status = BeliefStatus.CONTRADICTED
            world_model_store.update_belief(belief)
            adjustments.append(
                AdjustmentRecord(
                    target=AdjustmentTarget.BELIEF_CONFIDENCE,
                    target_id=belief_id,
                    target_facet="world_model",
                    field="confidence",
                    old_value=old_c,
                    new_value=new_c,
                    delta=round(new_c - old_c, 3),
                    status=AdjustmentStatus.APPLIED,
                    source_signal_id=event.event_id,
                    reasons=["belief_contradiction_adjustment"],
                    metadata=prov,
                )
            )
        reasons.append("belief_contradiction_updates")

    if world_model_store is not None and corroborate_ids:
        sources.append("explicit_belief_corroboration")
        for belief_id in corroborate_ids:
            belief = world_model_store.get_belief(belief_id)
            if belief is None:
                continue
            old_c = float(belief.confidence)
            new_c = _clamp_unit(old_c + DEFAULT_CONFIDENCE_DELTA)
            if new_c == old_c:
                continue
            prov = {
                "source_event_id": event.event_id,
                "source_signal": "belief_corroboration",
                "old_confidence": old_c,
                "new_confidence": new_c,
                "reason": "explicit corroboration metadata",
            }
            world_model_store.update_belief_confidence(
                belief_id,
                new_c,
                metadata_update={"learning_v1_provenance": prov},
            )
            adjustments.append(
                AdjustmentRecord(
                    target=AdjustmentTarget.BELIEF_CONFIDENCE,
                    target_id=belief_id,
                    target_facet="world_model",
                    field="confidence",
                    old_value=old_c,
                    new_value=new_c,
                    delta=round(new_c - old_c, 3),
                    status=AdjustmentStatus.APPLIED,
                    source_signal_id=event.event_id,
                    reasons=["belief_corroboration_adjustment"],
                    metadata=prov,
                )
            )
        reasons.append("belief_corroboration_updates")

    if cmap.belief_contradiction and not aligned_ids:
        routes.append(
            {
                "source": "world_model",
                "signal": "belief_contradiction",
                "target": "behavior",
                "suggested_adjustment": "increase_ask_bias_for_low_confidence_act",
                "strength": 0.05,
                "applied": False,
                "reason": "cycle signal map reported belief contradiction without aligned ids",
            }
        )
        reasons.append("route_wm_contradiction_pressure")

    # --- Memory: Hebbian co-retrieval edge strengthening ---
    co_ids = _co_retrieval_memory_ids(state)
    score_map, salience_map = _memory_score_maps(state)
    _fill_salience_from_store(co_ids, memory_store, salience_map)

    if (
        memory_store is not None
        and hasattr(memory_store, "strengthen_memory_edge")
        and len(co_ids) >= 2
    ):
        sources.append("memory_co_retrieval")
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(co_ids):
            for b in co_ids[i + 1 :]:
                pairs.append((a, b))
        for src_m, tgt_m in pairs:
            act_a = _activation_for_memory(src_m, score_map=score_map, salience_map=salience_map)
            act_b = _activation_for_memory(tgt_m, score_map=score_map, salience_map=salience_map)
            delta_w = _clamp_unit(hebbian_rate * act_a * act_b)
            if delta_w <= 0.0:
                continue
            edge = memory_store.strengthen_memory_edge(
                src_m,
                tgt_m,
                CO_EDGE_TYPE,
                delta_w,
                reason="co_retrieval_hebbian",
                provenance={
                    "source_event_id": event.event_id,
                    "activation_a": act_a,
                    "activation_b": act_b,
                    "hebbian_rate": hebbian_rate,
                },
            )
            old_w = _clamp_unit(float(edge.weight) - delta_w)
            adjustments.append(
                AdjustmentRecord(
                    target=AdjustmentTarget.MEMORY_EDGE,
                    target_id=edge.id,
                    target_facet="memory",
                    field="weight",
                    old_value=old_w,
                    new_value=float(edge.weight),
                    delta=delta_w,
                    status=AdjustmentStatus.APPLIED,
                    source_signal_id=event.event_id,
                    reasons=["co_retrieval_edge_strengthen"],
                    metadata={
                        "source_memory_id": edge.source_memory_id,
                        "target_memory_id": edge.target_memory_id,
                        "edge_type": CO_EDGE_TYPE.value,
                        "activation_a": act_a,
                        "activation_b": act_b,
                    },
                )
            )
        if pairs:
            reasons.append("memory_edge_hebbian")

    # --- Salience validation (bounded) ---
    if memory_store is not None and hasattr(memory_store, "list_high_salience"):
        included_set = set(co_ids)
        # High-salience downweight proposal: salient but not in current co-retrieval
        list_hs = memory_store.list_high_salience(20)  # type: ignore[union-attr]
        for rec in list_hs:
            if float(rec.salience) < 0.75:
                continue
            if rec.id in included_set:
                continue
            meta_m = rec.metadata if isinstance(rec.metadata, dict) else {}
            has_link = bool(
                meta_m.get("goal_id")
                or meta_m.get("belief_id")
                or meta_m.get("policy_rule_id")
            )
            if has_link:
                continue
            adjustments.append(
                AdjustmentRecord(
                    target=AdjustmentTarget.MEMORY_SALIENCE,
                    target_id=rec.id,
                    target_facet="memory",
                    field="salience",
                    old_value=float(rec.salience),
                    new_value=_clamp_unit(float(rec.salience) - SALIENCE_NUDGE),
                    delta=-SALIENCE_NUDGE,
                    status=AdjustmentStatus.PROPOSED,
                    source_signal_id=event.event_id,
                    reasons=["high_salience_low_recurrence_no_linkage"],
                    metadata={"strategy": "salience_decay_validation"},
                )
            )
            reasons.append("salience_downweight_proposal")
            break  # one per cycle for bounded churn

    # --- Salience upweight: success signals + memory in co-retrieval set ---
    positive_execution = False
    if isinstance(ex, dict):
        er = ex.get("last_execution_result")
        if isinstance(er, dict) and str(er.get("overall_status", "")).lower() == "success":
            positive_execution = True
    feedback_positive = meta.get("feedback") == "positive"
    if (
        memory_store is not None
        and hasattr(memory_store, "update_memory_salience")
        and co_ids
        and (positive_execution or feedback_positive)
    ):
        sources.append("salience_recurrence_boost")
        tgt_mem = meta.get("target_memory_id")
        for mid in co_ids[:5]:
            if (
                feedback_positive
                and isinstance(tgt_mem, str)
                and tgt_mem.strip()
                and mid == tgt_mem.strip()
            ):
                continue
            rec = memory_store.get_memory(mid)
            if rec is None:
                continue
            old_s = float(rec.salience)
            new_s = _clamp_unit(old_s + SALIENCE_NUDGE)
            if new_s == old_s:
                continue
            memory_store.update_memory_salience(mid, new_s)
            adjustments.append(
                AdjustmentRecord(
                    target=AdjustmentTarget.MEMORY_SALIENCE,
                    target_id=mid,
                    target_facet="memory",
                    field="salience",
                    old_value=old_s,
                    new_value=new_s,
                    delta=SALIENCE_NUDGE,
                    status=AdjustmentStatus.APPLIED,
                    source_signal_id=event.event_id,
                    reasons=["recurrence_or_success_salience_upweight"],
                    metadata={"strategy": "salience_recurrence"},
                )
            )
        reasons.append("salience_upweight_applied")

    # --- Goal resolution route → memory salience ( proposal only ) ---
    if isinstance(meta.get("goal_status"), str) and meta.get("goal_status") == "completed":
        routes.append(
            {
                "source": "goals",
                "signal": "goal_resolution",
                "target": "memory",
                "suggested_adjustment": "reinforce_related_memory_salience",
                "strength": 0.05,
                "applied": False,
                "reason": "goal completed; reinforce linked episodic context in future work",
            }
        )
        reasons.append("route_goal_to_memory")

    if positive_execution:
        routes.append(
            {
                "source": "executor",
                "signal": "execution_success",
                "target": "planner",
                "suggested_adjustment": "note_success_pattern",
                "strength": 0.05,
                "applied": False,
                "reason": "successful execution observed (planner pattern library is v2)",
            }
        )
    ex_fail = isinstance(ex, dict) and isinstance(ex.get("last_execution_result"), dict)
    if ex_fail:
        er = ex.get("last_execution_result")
        assert isinstance(er, dict)
        if str(er.get("overall_status", "")).lower() == "failed":
            routes.append(
                {
                    "source": "executor",
                    "signal": "execution_failure",
                    "target": "planner",
                    "suggested_adjustment": "flag_planner_pattern",
                    "strength": 0.05,
                    "applied": False,
                    "reason": "execution failure surfaced for future routing",
                }
            )

    # --- Synthetic signals for consumed learning-event types (inspection only) ---
    for ev in learning_events:
        et = ev.get("event_type") or ev.get("type")
        if et == "behavior_decision_trace_v2":
            signals.append(
                LearningSignal(
                    signal_type=SignalType.NEUTRAL,
                    source=SignalSource.BEHAVIOR_TRACE,
                    magnitude=0.0,
                    source_event_id=event.event_id,
                    metadata={"consumed_learning_event": ev},
                    reasons=["ingested_behavior_decision_trace_v2"],
                )
            )

    return LearningV1Bundle(
        signals=signals,
        adjustments=adjustments,
        cross_facet_routes=routes,
        reasons=reasons,
        signal_sources=sources,
    )


def merge_signal_sources(
    v0_reasons: list[str],
    v1: LearningV1Bundle,
) -> list[str]:
    out: list[str] = ["explicit_feedback", "execution", "goal_lifecycle"]
    out.extend(v1.signal_sources)
    for token in (*v1.reasons, *v0_reasons):
        if isinstance(token, str) and token.strip():
            cleaned = token.strip()
            if cleaned not in out:
                out.append(cleaned)
    return out
