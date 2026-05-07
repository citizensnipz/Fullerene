"""Deterministic world model facet for Fullerene World Model v1/v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, EventType, FacetResult, NexusState
from fullerene.world_model import Belief, BeliefSource, BeliefStatus, SQLiteWorldModelStore, WorldModelStore
from fullerene.world_model.models import BeliefType, normalize_statement, stable_belief_id, utcnow
from fullerene.world_model.rules import seed_builtin_belief_rules


@dataclass(slots=True)
class _BeliefMatch:
    belief: Belief
    score: float
    tag_overlap: float
    keyword_overlap: float
    shared_tags: list[str]
    shared_keywords: list[str]
    edge_neighbor_bonus: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.belief.id,
            "claim": self.belief.claim,
            "confidence": self.belief.confidence,
            "status": self.belief.status.value,
            "score": round(self.score, 3),
            "tag_overlap": round(self.tag_overlap, 3),
            "keyword_overlap": round(self.keyword_overlap, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "source": self.belief.source.value,
            "edge_neighbor_bonus": round(self.edge_neighbor_bonus, 3),
        }


class WorldModelFacet:
    """Expose deterministic belief lifecycle, graph v2, clusters, and relevance signals."""

    name = "world_model"

    def __init__(
        self,
        store: WorldModelStore,
        *,
        active_limit: int = 20,
        relevant_limit: int = 3,
        support_weight: float = 0.2,
        contradiction_weight: float = 0.3,
        contradiction_threshold: int = 2,
        low_confidence_threshold: float = 0.35,
    ) -> None:
        self.store = store
        self.active_limit = max(int(active_limit), 1)
        self.relevant_limit = max(int(relevant_limit), 1)
        self.support_weight = max(0.0, min(float(support_weight), 1.0))
        self.contradiction_weight = max(0.0, min(float(contradiction_weight), 1.0))
        self.contradiction_threshold = max(int(contradiction_threshold), 1)
        self.low_confidence_threshold = max(0.0, min(float(low_confidence_threshold), 1.0))

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        active_limit: int = 20,
        relevant_limit: int = 3,
    ) -> "WorldModelFacet":
        return cls(
            SQLiteWorldModelStore(path),
            active_limit=active_limit,
            relevant_limit=relevant_limit,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        world_updates = self._ingest_event_as_belief(event, state)
        active_beliefs = self.store.list_beliefs(limit=self.active_limit)
        if not active_beliefs:
            return FacetResult(
                facet_name=self.name,
                summary="World model facet found no active beliefs.",
                state_updates={
                    "last_active_belief_ids": [],
                    "last_relevant_beliefs": [],
                    "last_relevance_score": 0.0,
                },
                metadata={
                    "active_belief_count": 0,
                    "relevant_beliefs": [],
                    "relevance_score": 0.0,
                    "score_formula": "tag_overlap + keyword_overlap + confidence + bounded_edge_neighbor_bonus",
                    "world_model_updates": world_updates,
                    "learning_events": [],
                },
            )

        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)

        wm_learning: list[dict[str, object]] = []
        if isinstance(self.store, SQLiteWorldModelStore):
            seed_builtin_belief_rules(self.store)

        neighbor_bonus: dict[str, float] = {}
        sqlite_store = self.store if isinstance(self.store, SQLiteWorldModelStore) else None
        if sqlite_store is not None:
            neighbor_bonus = self._neighbor_bonus_map(active_beliefs, event_keywords, sqlite_store)

        relevant_matches = [
            match
            for belief in active_beliefs
            if (
                match := self._score_belief(
                    belief,
                    event_tags=event_tags,
                    event_keywords=event_keywords,
                    neighbor_bonus=neighbor_bonus.get(belief.id, 0.0),
                    memory_community_bonus=self._memory_community_bonus(state, belief),
                )
            )
            is not None
        ]

        wm_v2_pressure_extra: list[dict[str, object]] = []

        sqlite_extras: dict[str, object] | None = None
        if sqlite_store is not None:
            sqlite_extras = self._wm_v2_post_cycle(
                event,
                state,
                sqlite_store,
                relevant_matches=relevant_matches,
                world_updates=world_updates,
                wm_learning=wm_learning,
            )
            if isinstance(sqlite_extras.get("belief_cluster_pressure_signals"), list):
                wm_v2_pressure_extra.extend(
                    list(sqlite_extras["belief_cluster_pressure_signals"])
                )

        merged_pressure = world_updates.setdefault("pressure_signals", [])
        if isinstance(merged_pressure, list) and wm_v2_pressure_extra:
            merged_pressure.extend(wm_v2_pressure_extra)

        relevant_matches.sort(
            key=lambda match: (
                match.score,
                match.belief.confidence,
                match.belief.updated_at.timestamp(),
                match.belief.id,
            ),
            reverse=True,
        )
        relevant_matches = relevant_matches[: self.relevant_limit]
        relevance_score = (
            round(relevant_matches[0].score, 3) if relevant_matches else 0.0
        )

        if relevant_matches:
            summary = (
                f"World model facet matched {len(relevant_matches)} beliefs; "
                f"top relevance score {relevance_score:.3f}."
            )
        else:
            summary = (
                f"World model facet checked {len(active_beliefs)} beliefs "
                "and found no relevant matches."
            )

        relevant_belief_payload = [match.to_dict() for match in relevant_matches]

        wm_v2_requires_approval = False
        wm_v2_low_suppress = False
        wm_v2_graph_confidence = relevance_score / max(
            len(relevant_matches) + 3, 1
        )  # benign default
        top_contrad = 0.0
        top_cluster_pressure = 0.0
        if sqlite_extras:
            wm_v2_requires_approval = bool(sqlite_extras.get("requires_approval_due_to_contradiction"))
            wm_v2_low_suppress = bool(sqlite_extras.get("suppress_act_due_to_low_confidence"))
            wm_v2_graph_confidence = float(sqlite_extras.get("belief_graph_confidence") or 0.0)
            top_contrad = float(sqlite_extras.get("top_contradiction_score") or 0.0)
            top_cluster_pressure = float(sqlite_extras.get("top_belief_cluster_pressure") or 0.0)

        if wm_v2_requires_approval and not any(
            isinstance(s, dict) and s.get("entry_type") == "contradiction" for s in merged_pressure if isinstance(s, dict)
        ):
            merged_pressure.append(
                {
                    "source": "world_model",
                    "entry_type": "contradiction",
                    "source_id": event.event_id,
                    "description": "High contradiction cluster warrants caution.",
                    "metadata": {"world_model_graph_v2": True},
                }
            )

        edges_sample: list[dict[str, object]] = []
        if sqlite_store and relevant_matches:
            for rm in relevant_matches[:2]:
                edges_sample.extend(sqlite_store.get_belief_edges_for_belief(rm.belief.id, limit=6, min_weight=0.2))

        clusters_out: list[dict[str, object]] = []
        if sqlite_store:
            clusters_out = sqlite_store.list_belief_communities(
                cluster_type="contradiction_cluster", limit=4
            )

        cluster_sample = [dict(c) for c in (clusters_out[:2] if clusters_out else [])]
        state_updates_wm: dict[str, object] = {
            "last_active_belief_ids": [belief.id for belief in active_beliefs],
            "last_relevant_beliefs": relevant_belief_payload,
            "last_relevance_score": relevance_score,
            "wm_v2_requires_approval_due_to_contradiction": wm_v2_requires_approval,
            "wm_v2_suppress_act_due_to_low_confidence": wm_v2_low_suppress,
            "wm_v2_belief_graph_confidence": round(wm_v2_graph_confidence, 4),
            "wm_v2_top_contradiction_score": round(top_contrad, 4),
            "wm_v2_top_belief_cluster_pressure": round(top_cluster_pressure, 4),
            "wm_v2_contradiction_cluster_sample": cluster_sample,
            "belief_graph_strategy": sqlite_extras.get("belief_graph_strategy", "v1_compatible")
            if sqlite_extras
            else "v1_compatible",
        }

        meta_out: dict[str, object] = {
            "world_model_graph_version": "v2",
            "active_belief_count": len(active_beliefs),
            "relevant_beliefs": relevant_belief_payload,
            "relevance_score": relevance_score,
            "event_tags": sorted(event_tags),
            "event_keywords": sorted(event_keywords),
            "score_formula": "tag_overlap + keyword_overlap + confidence + bounded_edge_neighbor_bonus",
            "world_model_updates": world_updates,
            "contradiction_signals": merged_pressure if isinstance(merged_pressure, list) else [],
            "learning_events": wm_learning,
            "active_contradiction_clusters": clusters_out[:4],
            "relevant_belief_edges_sample": edges_sample[:12],
            "requires_approval_due_to_contradiction": wm_v2_requires_approval,
            "suppress_act_due_to_low_confidence": wm_v2_low_suppress,
            "belief_graph_confidence": round(wm_v2_graph_confidence, 4),
            "top_contradiction_score": round(top_contrad, 4),
            "top_belief_cluster_pressure": round(top_cluster_pressure, 4),
        }

        return FacetResult(
            facet_name=self.name,
            summary=summary,
            state_updates=dict(state_updates_wm),
            metadata=meta_out,
        )

    # --- World Model v2 helpers -----------------------------------------

    def _neighbor_bonus_map(
        self,
        active_beliefs: list[Belief],
        event_keywords: set[str],
        store: SQLiteWorldModelStore,
        *,
        min_edge_weight: float = 0.22,
        cap: float = 0.12,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        seeded: list[str] = []
        for b in active_beliefs[:60]:
            for tok in tokenize(b.claim):
                if tok in event_keywords:
                    seeded.append(b.id)
                    break
        for bid in seeded[:24]:
            for edge in store.get_belief_edges_for_belief(
                bid, limit=14, min_weight=min_edge_weight
            ):
                et = str(edge.get("edge_type") or "")
                if et not in {"supporting", "related"}:
                    continue
                nb = edge["target_belief_id"] if edge["source_belief_id"] == bid else edge[
                    "source_belief_id"
                ]
                w = float(edge.get("weight") or 0.0)
                bonus = min(cap, w * 0.06)
                out[nb] = max(out.get(nb, 0.0), bonus)
        return out

    def _memory_community_bonus(self, state: NexusState, belief: Belief) -> float:
        mem = state.facet_state.get("memory")
        if not isinstance(mem, dict):
            return 0.0
        active = False
        for row in mem.get("last_context_memory_communities") or []:
            if not isinstance(row, dict):
                continue
            ac = row.get("activation_score") or row.get("activation")
            try:
                if float(ac or 0.0) >= 0.18:
                    active = True
            except (TypeError, ValueError):
                continue
            if active:
                break
        mids = belief.metadata.get("memory_community_ids") if belief.metadata else None
        has_mem = isinstance(mids, list) and bool(mids)
        if active and has_mem:
            return 0.05
        if has_mem:
            return 0.025
        return 0.0

    def _wm_v2_post_cycle(
        self,
        event: Event,
        state: NexusState,
        store: SQLiteWorldModelStore,
        *,
        relevant_matches: list[_BeliefMatch],
        world_updates: dict[str, object],
        wm_learning: list[dict[str, object]],
    ) -> dict[str, object]:
        learning_out: dict[str, object] = {
            "belief_graph_strategy": "deterministic_contradiction_components_v0",
        }
        updated_ids_raw = world_updates.get("updated_belief_ids")
        updated_ids: set[str] = set()
        if isinstance(updated_ids_raw, list):
            updated_ids = {str(x) for x in updated_ids_raw if str(x)}

        rel_ids = {m.belief.id for m in relevant_matches}
        activated_ids = sorted(rel_ids | updated_ids)

        if len(activated_ids) >= 2:
            amt = 0.06 if event.event_type != EventType.SYSTEM_TICK else 0.02
            for i, a_id in enumerate(activated_ids[:20]):
                for b_id in activated_ids[i + 1 : 20]:
                    store.strengthen_belief_edge_pair(
                        a_id,
                        b_id,
                        "supporting",
                        amount=amt,
                        provenance={
                            "event_ids": [str(event.event_id)],
                        },
                    )

        mem_activation = self._aggregate_memory_activation(state)
        mc_by_id = self._active_memory_cluster_ids(state)
        for belief_id in sorted(rel_ids | updated_ids)[:28]:
            for mid in mc_by_id:
                store.attach_memory_communities_to_belief(
                    belief_id, [mid], event_id=str(event.event_id)
                )
            mids = mc_by_id
            for oid in sorted(rel_ids | updated_ids)[:12]:
                if oid == belief_id:
                    continue
                if mids:
                    store.strengthen_belief_edge_pair(
                        belief_id,
                        oid,
                        "related",
                        amount=0.03 + min(0.04, float(mem_activation) * 0.06),
                        provenance={
                            "memory_ids": [],
                            "event_ids": [str(event.event_id)],
                        },
                    )

        contradiction = bool(world_updates.get("contradiction_detected"))
        cluster_ids: list[str] = []
        should_rebuild = contradiction or bool(updated_ids) or bool(
            world_updates.get("created_belief_id")
        )
        if should_rebuild:
            cluster_ids = store.rebuild_belief_communities(max_beliefs=120)

        max_contra_weight = store.list_belief_edges_global(limit=120, min_weight=0.2)
        mw = max(
            (float(e.get("weight") or 0.0) for e in max_contra_weight if str(e.get("edge_type")) == "contradicting"),
            default=0.0,
        )

        touched = store.update_belief_community_activation(
            activated_member_ids=activated_ids,
            memory_cluster_activation=mem_activation,
            max_edge_strength=mw,
        )

        rule_out = store.evaluate_belief_rules(max_rule_evaluations=50)
        inferred = rule_out.get("inferred_belief_ids") or []
        applied_rules = rule_out.get("applied_rule_ids") or []
        if inferred:
            wm_learning.append({"kind": "inferred_belief_created", "belief_ids": list(inferred)})
        for rid in applied_rules:
            wm_learning.append({"kind": "belief_rule_applied", "rule_id": rid})
        edge_meta = rule_out.get("negative_hits") or []
        for hit in edge_meta:
            if isinstance(hit, dict):
                wm_learning.append(
                    {"kind": "belief_negative_pattern_trigger", "detail": dict(hit)},
                )

        contradiction_cluster_signals: list[dict[str, object]] = []
        high_clusters = store.list_belief_communities(
            cluster_type="contradiction_cluster", limit=5, min_pressure=0.28
        )
        top_contra = max((float(c["contradiction_score"]) for c in high_clusters), default=0.0)
        top_pressure = max((float(c["pressure_score"]) for c in high_clusters), default=0.0)
        for c in high_clusters:
            cid = str(c["cluster_id"])
            if float(c["pressure_score"] or 0.0) < 0.28:
                continue
            contradiction_cluster_signals.append(
                {
                    "source": "world_model",
                    "entry_type": "belief_contradiction_cluster",
                    "source_id": cid,
                    "description": f"Contradiction cluster {cid[:8]} pressure "
                    f"{float(c['pressure_score']):.2f}",
                    "metadata": {
                        "cluster_id": cid,
                        "intensity": float(c["pressure_score"]),
                        "pressure_score": float(c["pressure_score"]),
                        "contradiction_score": float(c["contradiction_score"]),
                        "unresolved_score": float(c["unresolved_score"]),
                        "representative_belief_ids": list(c.get("representative_belief_ids") or [])[
                            :5
                        ],
                        "reason": "contradiction_cluster_active",
                        "belief_graph_strategy": learning_out["belief_graph_strategy"],
                        "intensity_hint": float(c["pressure_score"]),
                    },
                }
            )
            wm_learning.append(
                {
                    "kind": "contradiction_cluster_pressure",
                    "cluster_id": cid,
                    "pressure_score": float(c["pressure_score"]),
                }
            )

        requires_approval = top_contra >= 0.55 or (
            contradiction and mw >= 0.45 and len(cluster_ids or []) >= 1
        )
        low_suppress = any(
            m.belief.confidence <= self.low_confidence_threshold for m in relevant_matches
        )
        if requires_approval:
            wm_learning.append({"kind": "belief_requires_approval", "top_contradiction": top_contra})

        learning_out.update(
            {
                "belief_cluster_pressure_signals": contradiction_cluster_signals,
                "requires_approval_due_to_contradiction": requires_approval,
                "suppress_act_due_to_low_confidence": low_suppress,
                "belief_graph_confidence": round(
                    min(1.0, sum(m.belief.confidence for m in relevant_matches) / max(len(relevant_matches), 1))
                    if relevant_matches
                    else 0.35,
                    4,
                ),
                "top_contradiction_score": top_contra,
                "top_belief_cluster_pressure": top_pressure,
            }
        )

        wm_learning.extend(
            [
                {"kind": "belief_cluster_touch", "touched_clusters": list(touched.keys())[:16]},
            ]
        )
        learning_out.setdefault("clusters_rebuilt_ids", cluster_ids)
        memory_community_bonus_applied = bool(mc_by_id and rel_ids)

        wm_learning.extend(
            [
                {
                    "kind": "belief_graph_coactivation_hebbian",
                    "belief_ids": activated_ids[:32],
                    "tick": bool(event.event_type == EventType.SYSTEM_TICK),
                },
                {"kind": "memory_community_belief_link", "active": memory_community_bonus_applied},
            ]
        )

        return learning_out

    @staticmethod
    def _aggregate_memory_activation(state: NexusState) -> float:
        mem = state.facet_state.get("memory")
        if not isinstance(mem, dict):
            return 0.0
        best = 0.0
        for row in mem.get("last_context_memory_communities") or []:
            if not isinstance(row, dict):
                continue
            try:
                act = float(row.get("activation_score") or row.get("activation") or 0.0)
            except (TypeError, ValueError):
                act = 0.0
            best = max(best, min(1.0, act))
        return round(best, 4)

    @staticmethod
    def _active_memory_cluster_ids(state: NexusState) -> list[str]:
        mem = state.facet_state.get("memory")
        if not isinstance(mem, dict):
            return []
        out: list[str] = []
        for row in mem.get("last_context_memory_communities") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("community_id") or "")
            if cid and cid not in out:
                out.append(cid)
        return out[:8]

    # --- Ingest / edges / scoring --------------------------------------

    def _canonical_belief_for_contradiction_cycle(self, belief: Belief) -> Belief:
        """Map partner rows back to the primary belief so v1 contradiction counts hold."""
        md = belief.metadata if isinstance(belief.metadata, dict) else {}
        if md.get("origin") != "contradiction_partner":
            return belief
        pid = md.get("contradicts_belief_id") or md.get("paired_belief_id")
        if not pid:
            return belief
        primary = self.store.get_belief(str(pid))
        return primary if primary is not None else belief

    def _ingest_event_as_belief(self, event: Event, state: NexusState) -> dict[str, object]:
        content = event.content.strip()
        if not content:
            return {"updated_belief_ids": [], "pressure_signals": []}
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        sqlite_ex = isinstance(self.store, SQLiteWorldModelStore)
        explicit_belief_create = bool(metadata.get("create_belief"))
        is_question_like = content.endswith("?")
        if not explicit_belief_create and is_question_like:
            return {"updated_belief_ids": [], "pressure_signals": []}
        normalized_key = normalize_statement(content)
        if not normalized_key:
            return {"updated_belief_ids": [], "pressure_signals": []}

        contradiction_partner_id: str | None = None

        belief = self.store.get_belief_by_normalized_key(normalized_key)
        if belief is not None:
            belief = self._canonical_belief_for_contradiction_cycle(belief)
        if belief is None:
            belief = self._find_contradicted_match(content, normalized_key)
        if (
            belief is not None
            and explicit_belief_create
            and belief.source_event_id == event.event_id
        ):
            return {"updated_belief_ids": [belief.id], "pressure_signals": []}
        if belief is None:
            belief = Belief(
                id=stable_belief_id(normalized_key),
                claim=content,
                confidence=0.6,
                status=BeliefStatus.VALID,
                source=BeliefSource.MEMORY,
                source_event_id=event.event_id,
                normalized_key=normalized_key,
                support_count=1,
                contradiction_count=0,
                last_support_event_id=event.event_id,
                last_updated_event_id=event.event_id,
                belief_type=self._classify_belief_type(content),
                sources=[event.event_id],
                metadata={
                    "origin": "memory",
                    "domain": metadata.get("event_domain"),
                    "tags": sorted(extract_event_tags(event)),
                },
            )
            self.store.add_belief(belief)
            self._upsert_edges(belief)
            return {
                "updated_belief_ids": [belief.id],
                "pressure_signals": [],
                "created_belief_id": belief.id,
            }

        contradiction, contradiction_kind = self._is_contradiction(
            belief.claim,
            content,
            belief.normalized_key,
            normalized_key,
        )

        partner: Belief | None = None
        if contradiction and sqlite_ex:
            partner = self.store.get_belief_by_normalized_key(normalized_key)
            if partner is None:
                partner = Belief(
                    id=stable_belief_id(normalized_key + "|ctx"),
                    claim=content,
                    confidence=0.42,
                    status=BeliefStatus.CONTRADICTED,
                    source=BeliefSource.MEMORY,
                    source_event_id=event.event_id,
                    normalized_key=normalized_key,
                    support_count=0,
                    contradiction_count=1,
                    last_support_event_id=event.event_id,
                    last_updated_event_id=event.event_id,
                    last_contradiction_event_id=event.event_id,
                    belief_type=self._classify_belief_type(content),
                    sources=sorted(set([*(belief.sources), event.event_id])),
                    metadata={
                        "origin": "contradiction_partner",
                        "paired_belief_id": belief.id,
                        "contradicts_belief_id": belief.id,
                    },
                )
                self.store.add_belief(partner)
                contradiction_partner_id = partner.id
            else:
                plist = partner.metadata.setdefault("paired_belief_ids", [])
                if belief.id not in plist:
                    plist.append(belief.id)
                contradiction_partner_id = partner.id
                self.store.update_belief(partner)

        if contradiction:
            belief.contradiction_count += 1
            belief.confidence = self._clamp01(
                belief.confidence - (belief.confidence * self.contradiction_weight)
            )
            belief.last_contradiction_event_id = event.event_id
            if belief.contradiction_count >= self.contradiction_threshold:
                belief.status = BeliefStatus.CONTRADICTED
            belief.metadata["last_contradiction_kind"] = contradiction_kind
        else:
            belief.support_count += 1
            belief.confidence = self._clamp01(
                belief.confidence + ((1.0 - belief.confidence) * self.support_weight)
            )
            belief.last_support_event_id = event.event_id
            if normalized_key == belief.normalized_key:
                belief.status = BeliefStatus.REDUNDANT
            elif belief.status == BeliefStatus.CONTRADICTED and belief.confidence >= 0.5:
                belief.status = BeliefStatus.VALID

        belief.last_updated_event_id = event.event_id
        belief.updated_at = utcnow()
        belief.sources = sorted(set([*belief.sources, event.event_id]))
        belief.metadata["last_updated_timestamp"] = belief.updated_at.isoformat()

        mids = metadata.get("source_memory_community_ids")
        if sqlite_ex and isinstance(mids, list) and mids:
            belief.metadata.setdefault("memory_community_ids", [])
            merged = sorted(
                set(belief.metadata.get("memory_community_ids") or []).union(str(x) for x in mids)
            )[:20]
            belief.metadata["memory_community_ids"] = merged

        self.store.update_belief(belief)
        self._upsert_edges(belief)

        if contradiction and sqlite_ex and partner is not None:
            sid = contradiction_partner_id or partner.id
            self.store.strengthen_belief_edge_pair(
                belief.id,
                sid,
                "contradicting",
                amount=0.22,
                provenance={"event_ids": [str(event.event_id)]},
                bump_support=False,
            )
            for weaker in ("related", "supporting"):
                self.store.weaken_belief_edge_pair(
                    belief.id,
                    sid,
                    weaker,
                    amount=0.12,
                    provenance={"event_ids": [str(event.event_id)]},
                )

        pressure_signals = self._pressure_signals_for_belief(belief)

        extras: dict[str, object] = {
            "updated_belief_ids": [belief.id]
            + ([contradiction_partner_id] if contradiction_partner_id else []),
            "pressure_signals": pressure_signals,
            "contradiction_detected": contradiction,
        }
        if contradiction_partner_id:
            extras["contradiction_partner_belief_id"] = contradiction_partner_id
            extras.setdefault("edges", []).append(
                {
                    "source_belief_id": belief.id,
                    "target_belief_id": contradiction_partner_id,
                    "edge_type": "contradicting",
                    "weight": 0.4,
                    "metadata": {"origin": "ingest"},
                }
            )
        return extras

    def _find_contradicted_match(self, incoming_claim: str, incoming_key: str) -> Belief | None:
        incoming_tokens = tokenize(incoming_claim)
        for candidate in self.store.list_beliefs(limit=100):
            contradiction, _ = self._is_contradiction(
                candidate.claim,
                incoming_claim,
                candidate.normalized_key,
                incoming_key,
            )
            if not contradiction:
                continue
            candidate_tokens = tokenize(candidate.claim)
            overlap = len(incoming_tokens & candidate_tokens)
            union = len(incoming_tokens | candidate_tokens) or 1
            if (overlap / union) >= 0.25:
                return candidate
        return None

    def _upsert_edges(self, anchor: Belief) -> None:
        if not hasattr(self.store, "add_belief_edge"):
            return
        candidates = self.store.list_beliefs(limit=40)
        anchor_tokens = set(tokenize(anchor.claim))
        anchor_low = anchor.claim.lower()
        causal_hints = ("because", "therefore", "due to", "cause")
        for belief in candidates:
            if belief.id == anchor.id:
                continue
            weight = 0.0
            edge_type = "related"
            shared = anchor_tokens & set(tokenize(belief.claim))
            if shared:
                weight = max(weight, len(shared) / max(len(anchor_tokens | set(tokenize(belief.claim))), 1))
            temporal = False
            if set(anchor.sources) & set(belief.sources):
                weight = max(weight, 0.7)
                edge_type = "temporal"
                temporal = True

            causal = False
            if shared and (
                any(h in anchor_low for h in causal_hints)
                or any(h in belief.claim.lower() for h in causal_hints)
            ):
                causal = True
                edge_type = "causal"
                weight = max(weight, min(len(shared) / max(len(shared) + 1, 1), 0.62))

            if weight >= 0.2:
                self.store.add_belief_edge(
                    source_belief_id=anchor.id,
                    target_belief_id=belief.id,
                    edge_type=edge_type,
                    weight=min(1.0, weight),
                    metadata={"shared_keywords": sorted(shared), "temporal": temporal, "causal": causal},
                )

    def _pressure_signals_for_belief(self, belief: Belief) -> list[dict[str, object]]:
        signals: list[dict[str, object]] = []
        if belief.contradiction_count > 0:
            signals.append(
                {
                    "source": "world_model",
                    "entry_type": "contradiction",
                    "source_id": belief.id,
                    "description": f"Belief contradiction: {belief.claim[:100]}",
                    "metadata": {
                        "belief_id": belief.id,
                        "confidence": belief.confidence,
                        "contradiction_count": belief.contradiction_count,
                        "support_count": belief.support_count,
                        "priority": belief.priority,
                    },
                }
            )
        if belief.contradiction_count >= self.contradiction_threshold:
            signals[-1]["metadata"]["severity"] = "high"
        if belief.confidence <= self.low_confidence_threshold:
            signals.append(
                {
                    "source": "world_model",
                    "entry_type": "uncertainty",
                    "source_id": belief.id,
                    "description": f"Belief uncertainty: {belief.claim[:100]}",
                    "metadata": {"belief_id": belief.id, "belief_confidence": belief.confidence},
                }
            )
        return signals

    @staticmethod
    def _classify_belief_type(content: str) -> BeliefType:
        text = content.lower()
        if any(token in text for token in ("i like", "i prefer", "favorite")):
            return BeliefType.PREFERENCE
        if any(token in text for token in ("can ", "able to", "capable")):
            return BeliefType.CAPABILITY
        if text:
            return BeliefType.FACT
        return BeliefType.UNKNOWN

    @staticmethod
    def _is_contradiction(
        existing_claim: str,
        incoming_claim: str,
        existing_key: str,
        incoming_key: str,
    ) -> tuple[bool, str | None]:
        if existing_key == incoming_key:
            return False, None
        existing = existing_claim.lower()
        incoming = incoming_claim.lower()
        neg_pairs = [(" is ", " is not "), (" has ", " does not have "), (" can ", " cannot ")]
        for pos, neg in neg_pairs:
            if (pos in existing and neg in incoming) or (neg in existing and pos in incoming):
                return True, "direct_negation"
        num_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        existing_nums = num_pattern.findall(existing)
        incoming_nums = num_pattern.findall(incoming)
        if existing_nums and incoming_nums and existing_nums != incoming_nums:
            base_existing = re.sub(num_pattern, "<num>", existing_key)
            base_incoming = re.sub(num_pattern, "<num>", incoming_key)
            if base_existing == base_incoming:
                return True, "numeric_conflict"
        if any(word in incoming.split() for word in ("not", "never", "no")) and existing_key in incoming_key:
            return True, "keyword_negation"
        return False, None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _score_belief(
        belief: Belief,
        *,
        event_tags: set[str],
        event_keywords: set[str],
        neighbor_bonus: float = 0.0,
        memory_community_bonus: float = 0.0,
    ) -> _BeliefMatch | None:
        belief_tags = set(belief.tags)
        belief_keywords = tokenize(belief.claim)
        shared_tags = sorted(event_tags & belief_tags)
        shared_keywords = sorted(event_keywords & belief_keywords)

        if not shared_tags and not shared_keywords and neighbor_bonus <= 0.0:
            return None

        tag_overlap = len(shared_tags) / len(belief_tags) if belief_tags else 0.0
        keyword_overlap = (
            len(shared_keywords) / len(belief_keywords) if belief_keywords else 0.0
        )
        bonus = neighbor_bonus + memory_community_bonus
        score = tag_overlap + keyword_overlap + belief.confidence + bonus

        return _BeliefMatch(
            belief=belief,
            score=score,
            tag_overlap=tag_overlap,
            keyword_overlap=keyword_overlap,
            shared_tags=shared_tags,
            shared_keywords=shared_keywords,
            edge_neighbor_bonus=neighbor_bonus + memory_community_bonus,
        )
