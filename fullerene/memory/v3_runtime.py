"""Memory v3 runtime helpers: community activation, LPB signals, learning events."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fullerene.memory.communities import MemoryCommunity
from fullerene.memory.models import MemoryLayer, MemoryRecord
from fullerene.memory.store import SQLiteMemoryStore
from fullerene.memory import v3 as mem_v3
from fullerene.nexus.models import Event, EventType, NexusState


def _prior_lp_memory_refs(state: NexusState) -> set[str]:
    out: set[str] = set()
    lp = state.facet_state.get("signals", {}).get("latent_pressure", {})
    if not isinstance(lp, dict):
        return out
    for row in lp.get("entries", []):
        if not isinstance(row, dict) or row.get("status") != "active":
            continue
        md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for k in ("memory_ids", "related_memory_ids", "memory_id"):
            v = md.get(k)
            if isinstance(v, list):
                for item in v:
                    out.add(str(item))
            elif v:
                out.add(str(v))
    return out


def _world_model_memory_refs(state: NexusState) -> set[str]:
    out: set[str] = set()
    wm = state.facet_state.get("world_model")
    if not isinstance(wm, dict):
        return out
    for key in ("last_contradiction_signals", "contradiction_signals"):
        rows = wm.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            mids = md.get("memory_ids") or md.get("source_memory_ids") or md.get("related_memory_ids")
            if isinstance(mids, list):
                for m in mids:
                    out.add(str(m))
    return out


def _attention_focus_retrieved(state: NexusState, retrieved_ids: set[str]) -> float:
    att = state.facet_state.get("attention")
    if not isinstance(att, dict):
        return 0.0
    iid = str(att.get("last_attention_broadcast_item_id") or "")
    if iid and iid in retrieved_ids:
        return 1.0
    raw = att.get("last_attention_broadcast")
    if isinstance(raw, dict) and str(raw.get("source") or "") == "memory":
        mid = str(raw.get("item_id") or "")
        if mid and mid in retrieved_ids:
            return 1.0
    return 0.0


def run_community_activation_cycle(
    store: SQLiteMemoryStore,
    event: Event,
    state: NexusState,
    relevant_memories: list[MemoryRecord],
    breakdowns: list[dict[str, Any]],
    *,
    retrieve_limit: int,
) -> dict[str, Any]:
    """Update community rows, decay inactive, emit LPB-shaped signals + learning hooks."""
    retrieved_lt: list[tuple[MemoryRecord, dict[str, Any]]] = []
    for mem, br in zip(relevant_memories, breakdowns):
        if mem.memory_layer != MemoryLayer.WORKING:
            retrieved_lt.append((mem, br))

    by_cid: dict[str, list[tuple[MemoryRecord, dict[str, Any]]]] = defaultdict(list)
    retrieved_ids = {m.id for m, _ in retrieved_lt}
    for mem, br in retrieved_lt:
        cid = mem.community_id
        if not cid:
            comms = store.list_communities_for_memory(mem.id)
            if comms:
                cid = comms[0].community_id
        if cid:
            by_cid[cid].append((mem, br))

    lp_refs = _prior_lp_memory_refs(state)
    wm_refs = _world_model_memory_refs(state)
    att_focus = _attention_focus_retrieved(state, retrieved_ids)

    now = datetime.now(timezone.utc)
    activated_ids: set[str] = set()
    latent_signals: list[dict[str, Any]] = []
    learning_events: list[dict[str, Any]] = []
    context_communities: list[dict[str, Any]] = []

    priority_goal_hint = 0.0
    goals_fs = state.facet_state.get("goals")
    if isinstance(goals_fs, dict) and goals_fs.get("last_active_goal_ids"):
        agi = goals_fs.get("last_active_goal_ids") or []
        for mem, _b in retrieved_lt:
            md = mem.metadata if isinstance(mem.metadata, dict) else {}
            eg = md.get("event_metadata") if isinstance(md.get("event_metadata"), dict) else {}
            gid = md.get("goal_id") or eg.get("goal_id")
            if gid and gid in agi:
                priority_goal_hint = 0.25
                break

    for cid, group in sorted(by_cid.items(), key=lambda kv: kv[0]):
        mc = store.get_memory_community(cid)
        if mc is None:
            continue
        mems = [p[0] for p in group]
        brs = [p[1] for p in group]
        max_sal = max((float(m.salience) for m in mems), default=0.0)
        rel_vals = [min(float(b.get("total", 0.0)), 1.5) for b in brs]
        avg_rel = sum(rel_vals) / max(len(rel_vals), 1)
        density = mem_v3.retrieval_density(
            len(mems),
            max(mc.member_count, 1),
            max(retrieve_limit, 1),
        )
        unresolved = 0.0
        for m in mems:
            if m.id in lp_refs or m.id in wm_refs:
                unresolved = max(unresolved, 0.6)
        if mc.unresolved_score:
            unresolved = max(unresolved, float(mc.unresolved_score) * 0.5)

        rec_fn = mem_v3.recency_activation_factor(
            last_activated_at=mc.last_activated_at,
            newest_member_created_at=max((m.created_at for m in mems), default=None)
            if mems
            else None,
            now=now,
        )

        act, act_reasons = mem_v3.compute_activation_score(
            retrieval_density=density,
            max_member_salience=max_sal,
            average_member_relevance=min(avg_rel, 1.0),
            attention_focus=att_focus,
            unresolved_factor=unresolved,
            recency_factor=rec_fn,
        )

        contrad_factor = min(1.0, float(mc.contradiction_count or 0) * 0.15)
        press, press_reasons = mem_v3.compute_pressure_score(
            activation_score=act,
            unresolved_score=max(float(mc.unresolved_score or 0.0), unresolved * 0.5),
            contradiction_factor=contrad_factor,
            priority_goal_factor=priority_goal_hint,
        )

        streak = (int(mc.activation_streak or 0) + 1) if act >= 0.2 else max(
            int(mc.activation_streak or 0) - 1, 0
        )
        inact = (int(mc.inactive_streak or 0) + 1) if act < 0.15 else 0

        unres2 = min(
            1.0,
            float(mc.unresolved_score or 0.0) * 0.9 + (0.03 if act >= 0.35 else 0.0),
        )

        mc2 = MemoryCommunity(
            community_id=mc.community_id,
            label=mc.label,
            member_memory_ids=mc.member_memory_ids,
            member_count=mc.member_count,
            top_tags=mc.top_tags,
            top_domains=mc.top_domains,
            top_roles=mc.top_roles,
            representative_memory_ids=mc.representative_memory_ids,
            centroid_embedding_id=mc.centroid_embedding_id,
            centroid_vector_hash=mc.centroid_vector_hash,
            activation_score=act,
            pressure_score=press,
            unresolved_score=unres2,
            contradiction_count=mc.contradiction_count,
            refinement_count=mc.refinement_count,
            activation_streak=streak,
            inactive_streak=inact,
            last_activated_at=now if act >= 0.12 else mc.last_activated_at,
            last_activated_event_id=event.event_id if act >= 0.12 else mc.last_activated_event_id,
            last_pressure_update_at=now,
            last_resolution_event_id=mc.last_resolution_event_id,
            resolved_recently=mc.resolved_recently,
            activation_reasons=act_reasons[:8],
            pressure_reasons=press_reasons[:8],
            community_detection_strategy=mc.community_detection_strategy,
            created_at=mc.created_at,
            updated_at=now,
            metadata=dict(mc.metadata),
        )
        store.update_memory_community_row(mc2)
        activated_ids.add(cid)

        learning_events.append(
            {
                "kind": "memory_community_activated",
                "community_id": cid,
                "activation_score": act,
                "pressure_score": press,
            }
        )

        if act >= 0.28 or press >= 0.22:
            context_communities.append(
                {
                    "community_id": cid,
                    "label": mc.label,
                    "activation_score": act,
                    "pressure_score": press,
                    "unresolved_score": unres2,
                    "contradiction_count": mc.contradiction_count,
                    "representative_memory_ids": list(mc.representative_memory_ids)[:5],
                    "top_tags": mc.top_tags[:6],
                    "top_domains": mc.top_domains[:4],
                }
            )

        should_emit_lp = event.event_type != EventType.SYSTEM_TICK and act >= 0.18
        if should_emit_lp:
            latent_signals.append(
                {
                    "source": "memory",
                    "entry_type": "memory_cluster_activation",
                    "source_id": event.event_id,
                    "description": f"Memory cluster {cid[:8]} activation {act:.2f}.",
                    "metadata": {
                        "community_id": cid,
                        "activation_score": act,
                        "pressure_score": press,
                        "unresolved_score": unres2,
                        "representative_memory_ids": list(mc.representative_memory_ids)[:8],
                        "reason": ",".join(act_reasons[:4]) or "community_cycle",
                        "intensity_hint": press,
                    },
                }
            )

    store.apply_memory_community_inactivity_decay(
        activated_ids=activated_ids,
        context_ran=True,
    )

    return {
        "latent_pressure_signals": latent_signals,
        "learning_events": learning_events,
        "last_context_memory_communities": context_communities[:6],
        "memory_v3_activated_communities": sorted(activated_ids),
    }
