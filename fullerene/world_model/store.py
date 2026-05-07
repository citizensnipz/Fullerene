"""SQLite-backed belief storage for Fullerene World Model v1."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, Sequence

from fullerene.world_model.community_detection import (
    DeterministicContradictionComponentDetector,
    stable_cluster_id,
)

from fullerene.world_model.hebbian import strengthen, weaken
from fullerene.world_model.models import (
    Belief,
    BeliefStatus,
    belief_edge_to_dict,
    normalize_statement,
    stable_belief_edge_id,
    utcnow,
)


class WorldModelStore(Protocol):
    def add_belief(self, belief: Belief) -> None:
        """Persist a belief."""

    def get_belief(self, belief_id: str) -> Belief | None:
        """Fetch a belief by id."""

    def list_beliefs(
        self,
        limit: int,
        status: BeliefStatus | None = None,
    ) -> list[Belief]:
        """Return beliefs, optionally filtered by status."""

    def list_active_beliefs(self, limit: int) -> list[Belief]:
        """Return active beliefs."""

    def update_belief(self, belief: Belief) -> None:
        """Persist edits to an existing belief."""

    def update_belief_confidence(
        self,
        belief_id: str,
        confidence: float,
        *,
        metadata_update: dict[str, Any] | None = None,
    ) -> Belief:
        """Update confidence only with optional merged metadata (Learning v1 helper)."""

    def get_belief_edges_for_belief(
        self, belief_id: str, *, limit: int = 50, min_weight: float = 0.0
    ) -> list[dict[str, Any]]:
        ...

    def strengthen_belief_edge(
        self,
        edge_id: str,
        amount: float,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def weaken_belief_edge(
        self,
        edge_id: str,
        amount: float,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def list_belief_edges(self, belief_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return lightweight belief edges for one belief."""


class SQLiteWorldModelStore:
    """SQLite-backed source of truth for Fullerene world-model beliefs."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add_belief(self, belief: Belief) -> None:
        payload = belief.to_dict()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO beliefs (
                    id,
                    claim,
                    confidence,
                    status,
                    tags_json,
                    source,
                    source_event_id,
                    source_memory_id,
                    created_at,
                    updated_at,
                    metadata_json,
                    sources_json,
                    normalized_key,
                    belief_type,
                    support_count,
                    contradiction_count,
                    last_support_event_id,
                    last_contradiction_event_id,
                    last_updated_event_id,
                    priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["claim"],
                    payload["confidence"],
                    payload["status"],
                    json.dumps(payload["tags"], sort_keys=True),
                    payload["source"],
                    payload["source_event_id"],
                    payload["source_memory_id"],
                    payload["created_at"],
                    payload["updated_at"],
                    json.dumps(payload["metadata"], sort_keys=True),
                    json.dumps(payload.get("sources", []), sort_keys=True),
                    payload.get("normalized_key", normalize_statement(payload["claim"])),
                    payload.get("belief_type", "unknown"),
                    payload.get("support_count", 0),
                    payload.get("contradiction_count", 0),
                    payload.get("last_support_event_id"),
                    payload.get("last_contradiction_event_id"),
                    payload.get("last_updated_event_id"),
                    payload.get("priority", 1.0),
                ),
            )
            connection.commit()

    def get_belief(self, belief_id: str) -> Belief | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    claim,
                    confidence,
                    status,
                    tags_json,
                    source,
                    source_event_id,
                    source_memory_id,
                    created_at,
                    updated_at,
                    metadata_json
                    , sources_json
                    , normalized_key
                    , belief_type
                    , support_count
                    , contradiction_count
                    , last_support_event_id
                    , last_contradiction_event_id
                    , last_updated_event_id
                    , priority
                FROM beliefs
                WHERE id = ?
                """,
                (belief_id,),
            ).fetchone()
        return self._row_to_belief(row) if row else None

    def list_beliefs(
        self,
        limit: int,
        status: BeliefStatus | None = None,
    ) -> list[Belief]:
        bounded_limit = self._normalize_limit(limit)
        query = """
            SELECT
                id,
                claim,
                confidence,
                status,
                tags_json,
                source,
                source_event_id,
                source_memory_id,
                created_at,
                updated_at,
                metadata_json
                    , sources_json
                    , normalized_key
                    , belief_type
                    , support_count
                    , contradiction_count
                    , last_support_event_id
                    , last_contradiction_event_id
                    , last_updated_event_id
                    , priority
            FROM beliefs
        """
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY confidence DESC, updated_at DESC, id DESC LIMIT ?"
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_belief(row) for row in rows]

    def list_active_beliefs(self, limit: int) -> list[Belief]:
        return self.list_beliefs(limit=limit, status=BeliefStatus.VALID)

    def update_belief(self, belief: Belief) -> None:
        belief.updated_at = utcnow()
        payload = belief.to_dict()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE beliefs
                SET
                    claim = ?,
                    confidence = ?,
                    status = ?,
                    tags_json = ?,
                    source = ?,
                    source_event_id = ?,
                    source_memory_id = ?,
                    updated_at = ?,
                    metadata_json = ?
                    , sources_json = ?
                    , normalized_key = ?
                    , belief_type = ?
                    , support_count = ?
                    , contradiction_count = ?
                    , last_support_event_id = ?
                    , last_contradiction_event_id = ?
                    , last_updated_event_id = ?
                    , priority = ?
                WHERE id = ?
                """,
                (
                    payload["claim"],
                    payload["confidence"],
                    payload["status"],
                    json.dumps(payload["tags"], sort_keys=True),
                    payload["source"],
                    payload["source_event_id"],
                    payload["source_memory_id"],
                    payload["updated_at"],
                    json.dumps(payload["metadata"], sort_keys=True),
                    json.dumps(payload.get("sources", []), sort_keys=True),
                    payload.get("normalized_key", normalize_statement(payload["claim"])),
                    payload.get("belief_type", "unknown"),
                    payload.get("support_count", 0),
                    payload.get("contradiction_count", 0),
                    payload.get("last_support_event_id"),
                    payload.get("last_contradiction_event_id"),
                    payload.get("last_updated_event_id"),
                    payload.get("priority", 1.0),
                    payload["id"],
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Belief {belief.id!r} does not exist")
            connection.commit()

    def update_belief_confidence(
        self,
        belief_id: str,
        confidence: float,
        *,
        metadata_update: dict[str, Any] | None = None,
    ) -> Belief:
        belief = self.get_belief(belief_id)
        if belief is None:
            raise KeyError(f"Belief {belief_id!r} does not exist")
        belief.confidence = Belief._validate_confidence(confidence)
        if metadata_update:
            merged = dict(belief.metadata)
            merged.update(metadata_update)
            belief.metadata = merged
        self.update_belief(belief)
        updated = self.get_belief(belief_id)
        assert updated is not None
        return updated

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _merge_provenance(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        out = dict(existing)
        for key in ("memory_ids", "event_ids", "rule_ids"):
            a = out.get(key) if isinstance(out.get(key), list) else []
            b = incoming.get(key) if isinstance(incoming.get(key), list) else []
            seen = {str(x) for x in a}
            merged = list(a)
            for x in b:
                sx = str(x)
                if sx not in seen:
                    merged.append(sx)
                    seen.add(sx)
            out[key] = merged[:120]
        for k, v in incoming.items():
            if k in {"memory_ids", "event_ids", "rule_ids"}:
                continue
            out.setdefault(k, v)
        return out

    def _serialize_edge_row(self, row: sqlite3.Row) -> dict[str, Any]:
        meta = json.loads(row["metadata_json"] or "{}")
        prov_raw = row["provenance_json"] if "provenance_json" in row.keys() else "{}"
        prov = json.loads(prov_raw or "{}")
        eid = row["edge_id"] if "edge_id" in row.keys() else stable_belief_edge_id(
            row["source_belief_id"], row["target_belief_id"], row["edge_type"]
        )
        sc = int(row["support_count"]) if "support_count" in row.keys() else 0
        cc = int(row["contradiction_count"]) if "contradiction_count" in row.keys() else 0
        la = row["last_activated_at"] if "last_activated_at" in row.keys() else None
        base = {
            "edge_id": eid,
            "source_belief_id": row["source_belief_id"],
            "target_belief_id": row["target_belief_id"],
            "edge_type": row["edge_type"],
            "weight": float(row["weight"]),
            "support_count": sc,
            "contradiction_count": cc,
            "last_activated_at": la,
            "last_updated_at": row["updated_at"],
            "provenance": prov,
            "metadata": meta,
        }
        return belief_edge_to_dict(base)

    def add_belief_edge(
        self,
        *,
        source_belief_id: str,
        target_belief_id: str,
        edge_type: str,
        weight: float,
        metadata: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        allow_self_link: bool = False,
    ) -> str | None:
        src, tgt = sorted((str(source_belief_id), str(target_belief_id)))
        if src == tgt and not allow_self_link:
            return None
        et = str(edge_type).strip().lower() or "related"
        wgt = self._clamp01(weight)
        edge_id = stable_belief_edge_id(src, tgt, et)
        now_iso = utcnow().isoformat()
        meta = dict(metadata or {})
        prov_raw: dict[str, Any] = {}
        if provenance:
            prov_raw = self._merge_provenance(prov_raw, provenance)

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT support_count, contradiction_count, provenance_json FROM belief_edges
                WHERE source_belief_id = ? AND target_belief_id = ? AND edge_type = ?
                """,
                (src, tgt, et),
            ).fetchone()
            sup = int(row["support_count"]) if row is not None and row["support_count"] is not None else 0
            ccd = (
                int(row["contradiction_count"])
                if row is not None and row["contradiction_count"] is not None
                else 0
            )
            if row is not None and row["provenance_json"]:
                prev = json.loads(row["provenance_json"] or "{}")
                prov_raw = self._merge_provenance(prev, prov_raw)

            connection.execute(
                """
                INSERT INTO belief_edges (
                    edge_id,
                    source_belief_id,
                    target_belief_id,
                    edge_type,
                    weight,
                    updated_at,
                    metadata_json,
                    support_count,
                    contradiction_count,
                    last_activated_at,
                    provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_belief_id, target_belief_id, edge_type) DO UPDATE SET
                    edge_id = COALESCE(excluded.edge_id, belief_edges.edge_id),
                    weight = excluded.weight,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json,
                    provenance_json = excluded.provenance_json,
                    support_count = belief_edges.support_count,
                    contradiction_count = belief_edges.contradiction_count
                """,
                (
                    edge_id,
                    src,
                    tgt,
                    et,
                    wgt,
                    now_iso,
                    json.dumps(meta, sort_keys=True),
                    sup,
                    ccd,
                    now_iso,
                    json.dumps(prov_raw, sort_keys=True),
                ),
            )
            connection.commit()
        return edge_id

    def get_belief_edge_by_edge_id(self, edge_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM belief_edges WHERE edge_id = ?",
                (edge_id,),
            ).fetchone()
        return self._serialize_edge_row(row) if row else None

    def strengthen_belief_edge(
        self,
        edge_id: str,
        amount: float,
        *,
        provenance: dict[str, Any] | None = None,
        bump_support: bool = True,
    ) -> dict[str, Any] | None:
        edge = (
            self.get_belief_edge_by_edge_id(edge_id)
            if edge_id
            else None  # pragma: no cover - guarded
        )
        if edge is None:
            return None
        return self._apply_edge_adjustment(edge, amount, weaken_mode=False, provenance=provenance, bump_support=bump_support)

    def weaken_belief_edge(
        self,
        edge_id: str,
        amount: float,
        *,
        provenance: dict[str, Any] | None = None,
        bump_contradiction: bool = False,
    ) -> dict[str, Any] | None:
        edge = self.get_belief_edge_by_edge_id(edge_id)
        if edge is None:
            return None
        return self._apply_edge_adjustment(
            edge, amount, weaken_mode=True, provenance=provenance, bump_contradiction=bump_contradiction
        )

    def strengthen_belief_edge_pair(
        self,
        source_belief_id: str,
        target_belief_id: str,
        edge_type: str,
        *,
        amount: float,
        provenance: dict[str, Any] | None = None,
        bump_support: bool = True,
    ) -> dict[str, Any] | None:
        a, b = sorted((str(source_belief_id), str(target_belief_id)))
        et = str(edge_type).strip().lower() or "related"
        eid = stable_belief_edge_id(a, b, et)
        if self.get_belief_edge_by_edge_id(eid) is None:
            self.add_belief_edge(source_belief_id=a, target_belief_id=b, edge_type=et, weight=0.05, provenance=provenance)
        return self.strengthen_belief_edge(eid, amount, provenance=provenance, bump_support=bump_support)

    def weaken_belief_edge_pair(
        self,
        source_belief_id: str,
        target_belief_id: str,
        edge_type: str,
        *,
        amount: float,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        a, b = sorted((str(source_belief_id), str(target_belief_id)))
        et = str(edge_type).strip().lower() or "related"
        eid = stable_belief_edge_id(a, b, et)
        if self.get_belief_edge_by_edge_id(eid) is None:
            return None
        return self.weaken_belief_edge(eid, amount, provenance=provenance)

    def _apply_edge_adjustment(
        self,
        edge_payload: dict[str, Any],
        amount: float,
        *,
        weaken_mode: bool,
        provenance: dict[str, Any] | None,
        bump_support: bool = False,
        bump_contradiction: bool = False,
    ) -> dict[str, Any] | None:
        src = edge_payload["source_belief_id"]
        tgt = edge_payload["target_belief_id"]
        et = edge_payload["edge_type"]
        cur_w = float(edge_payload["weight"])
        new_w = weaken(cur_w, amount) if weaken_mode else strengthen(cur_w, amount)
        now_iso = utcnow().isoformat()
        edge_id = str(edge_payload.get("edge_id") or stable_belief_edge_id(src, tgt, et))

        prov = dict(edge_payload.get("provenance") or {})
        if provenance:
            prov = self._merge_provenance(prov, provenance)

        sup = int(edge_payload.get("support_count") or 0)
        cco = int(edge_payload.get("contradiction_count") or 0)
        if weaken_mode:
            if bump_contradiction:
                cco += 1
        elif bump_support:
            sup += 1

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE belief_edges SET
                    weight = ?,
                    updated_at = ?,
                    last_activated_at = ?,
                    provenance_json = ?,
                    support_count = ?,
                    contradiction_count = ?
                WHERE edge_id = ?
                """,
                (new_w, now_iso, now_iso, json.dumps(prov, sort_keys=True), sup, cco, edge_id),
            )
            connection.commit()
        refreshed = self.get_belief_edge_by_edge_id(edge_id)
        return refreshed

    def list_belief_edges(
        self,
        belief_id: str,
        limit: int = 20,
        *,
        min_weight: float = 0.0,
    ) -> list[dict[str, Any]]:
        return self.get_belief_edges_for_belief(
            belief_id, limit=self._normalize_limit(limit), min_weight=min_weight
        )

    def get_belief_edges_for_belief(
        self, belief_id: str, *, limit: int = 50, min_weight: float = 0.0
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM belief_edges
                WHERE source_belief_id = ? OR target_belief_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (belief_id, belief_id, self._normalize_limit(limit)),
            ).fetchall()
        edges = [self._serialize_edge_row(row) for row in rows]
        if min_weight <= 0.0:
            return edges
        return [e for e in edges if float(e.get("weight") or 0.0) >= min_weight]

    def list_belief_edges_global(self, *, limit: int = 50, min_weight: float = 0.0) -> list[dict[str, Any]]:
        bounded = min(max(int(limit), 1), 500)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM belief_edges ORDER BY updated_at DESC LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        out = [self._serialize_edge_row(row) for row in rows]
        if min_weight > 0:
            out = [e for e in out if float(e.get("weight") or 0.0) >= min_weight]
        return out

    def add_belief_rule(
        self,
        *,
        rule_id: str,
        rule_type: str,
        antecedent_patterns: list[dict[str, Any]],
        consequent_pattern: str,
        confidence_weight: float = 0.5,
        historical_validity: float = 1.0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        rid = str(rule_id).strip()
        now_iso = utcnow().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO belief_rules (
                    rule_id, rule_type, antecedent_patterns_json, consequent_pattern,
                    confidence_weight, historical_validity,
                    support_count, failure_count, enabled, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                (
                    rid,
                    str(rule_type).strip().lower(),
                    json.dumps(antecedent_patterns, sort_keys=True),
                    consequent_pattern,
                    self._clamp01(confidence_weight),
                    self._clamp01(historical_validity),
                    1 if enabled else 0,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    now_iso,
                    now_iso,
                ),
            )
            connection.commit()

    def list_enabled_belief_rules(self, limit: int = 50) -> list[dict[str, Any]]:
        bounded = min(max(int(limit), 1), 200)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM belief_rules WHERE enabled = 1 ORDER BY rule_id ASC LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        return [self._row_to_rule(row) for row in rows]

    def count_belief_rules(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS c FROM belief_rules").fetchone()
        return int(row["c"]) if row else 0

    def increment_belief_rule_support(self, rule_id: str) -> None:
        self._bump_rule_counts(rule_id, support_delta=1)

    def increment_belief_rule_failure(self, rule_id: str) -> None:
        self._bump_rule_counts(rule_id, failure_delta=1)

    def _bump_rule_counts(
        self, rule_id: str, *, support_delta: int = 0, failure_delta: int = 0
    ) -> None:
        rid = str(rule_id).strip()
        if not rid:
            return
        now_iso = utcnow().isoformat()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT support_count, failure_count FROM belief_rules WHERE rule_id = ?",
                (rid,),
            ).fetchone()
            if row is None:
                return
            sc = int(row["support_count"] or 0) + support_delta
            fc = int(row["failure_count"] or 0) + failure_delta
            sc = max(0, sc)
            fc = max(0, fc)
            denom = max(sc + fc, 1)
            hv = self._clamp01(sc / denom)
            connection.execute(
                """
                UPDATE belief_rules SET
                    support_count = ?,
                    failure_count = ?,
                    historical_validity = ?,
                    updated_at = ?
                WHERE rule_id = ?
                """,
                (sc, fc, hv, now_iso, rid),
            )
            connection.commit()

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "rule_id": row["rule_id"],
            "rule_type": row["rule_type"],
            "antecedent_patterns": json.loads(row["antecedent_patterns_json"] or "[]"),
            "consequent_pattern": row["consequent_pattern"],
            "confidence_weight": float(row["confidence_weight"]),
            "historical_validity": float(row["historical_validity"]),
            "support_count": int(row["support_count"] or 0),
            "failure_count": int(row["failure_count"] or 0),
            "enabled": bool(int(row["enabled"])),
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def list_belief_communities(
        self,
        *,
        cluster_type: str | None = None,
        limit: int = 20,
        min_pressure: float = 0.0,
    ) -> list[dict[str, Any]]:
        bounded = min(max(int(limit), 1), 100)
        q = "SELECT * FROM belief_communities"
        params: list[Any] = []
        clauses: list[str] = []
        if cluster_type:
            clauses.append("cluster_type = ?")
            params.append(str(cluster_type))
        if min_pressure > 0:
            clauses.append("pressure_score >= ?")
            params.append(float(min_pressure))
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY pressure_score DESC, updated_at DESC LIMIT ?"
        params.append(bounded)
        with closing(self._connect()) as connection:
            rows = connection.execute(q, params).fetchall()
        return [self._row_to_community(r) for r in rows]

    def get_belief_community(self, cluster_id: str) -> dict[str, Any] | None:
        cid = str(cluster_id).strip()
        if not cid:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM belief_communities WHERE cluster_id = ?",
                (cid,),
            ).fetchone()
        return self._row_to_community(row) if row else None

    def list_communities_for_belief(self, belief_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        bounded = min(max(int(limit), 1), 50)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT bc.* FROM belief_communities bc
                INNER JOIN belief_community_members m ON bc.cluster_id = m.cluster_id
                WHERE m.belief_id = ?
                ORDER BY bc.pressure_score DESC
                LIMIT ?
                """,
                (belief_id, bounded),
            ).fetchall()
        return [self._row_to_community(r) for r in rows]

    def iter_belief_community_member_ids(self, cluster_id: str) -> list[str]:
        cid = str(cluster_id).strip()
        if not cid:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT belief_id FROM belief_community_members WHERE cluster_id = ? ORDER BY belief_id",
                (cid,),
            ).fetchall()
        return [str(row["belief_id"]) for row in rows]

    @staticmethod
    def _row_to_community(row: sqlite3.Row) -> dict[str, Any]:
        meta = json.loads(row["metadata_json"] or "{}")
        return {
            "cluster_id": row["cluster_id"],
            "cluster_type": row["cluster_type"],
            "member_count": int(row["member_count"] or 0),
            "dominant_status": row["dominant_status"],
            "contradiction_score": float(row["contradiction_score"] or 0.0),
            "activation_score": float(row["activation_score"] or 0.0),
            "pressure_score": float(row["pressure_score"] or 0.0),
            "unresolved_score": float(row["unresolved_score"] or 0.0),
            "representative_belief_ids": json.loads(
                row["representative_belief_ids_json"] or "[]"
            ),
            "top_terms": json.loads(row["top_terms_json"] or "[]"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_activated_at": row["last_activated_at"],
            "metadata": meta,
            "community_detection_strategy": row["community_detection_strategy"] or "",
        }

    def rebuild_belief_communities(
        self,
        *,
        max_beliefs: int = 120,
        min_contradiction_edge_weight: float = 0.25,
        strategy: DeterministicContradictionComponentDetector | None = None,
    ) -> list[str]:
        det = strategy or DeterministicContradictionComponentDetector(
            min_edge_weight=min_contradiction_edge_weight
        )
        beliefs = self.list_beliefs(limit=min(max(int(max_beliefs), 1), 300))
        by_id = {b.id: b for b in beliefs}
        ids = sorted(by_id.keys())
        bounded_edges = min(400, max(len(ids) * 8, 16))
        edge_rows = self.list_belief_edges_global(
            limit=bounded_edges, min_weight=min_contradiction_edge_weight
        )
        contra: list[tuple[str, str, float]] = []
        for e in edge_rows:
            if str(e.get("edge_type") or "") != "contradicting":
                continue
            s, t = e["source_belief_id"], e["target_belief_id"]
            if s not in by_id or t not in by_id:
                continue
            contra.append((s, t, float(e.get("weight") or 0.0)))

        clusters = det.detect_contradiction_clusters(belief_ids=ids, contradicting_edges=contra)
        now_iso = utcnow().isoformat()

        with closing(self._connect()) as connection:
            connection.execute(
                """
                DELETE FROM belief_community_members WHERE cluster_id IN (
                    SELECT cluster_id FROM belief_communities WHERE cluster_type = 'contradiction_cluster'
                )
                """
            )
            connection.execute(
                "DELETE FROM belief_communities WHERE cluster_type = 'contradiction_cluster'"
            )
            connection.commit()

        pair_weights: dict[tuple[str, str], float] = {}
        for s, t, w in contra:
            pair = tuple(sorted((s, t)))
            pair_weights[pair] = max(pair_weights.get(pair, 0.0), float(w))

        created_ids: list[str] = []
        for comp in clusters:
            members = sorted(comp)
            if len(members) < 2:
                continue
            label_terms: list[str] = []
            for mid in members[:4]:
                b = by_id.get(mid)
                if b:
                    for tok in (b.normalized_key or "").split()[:3]:
                        if tok and tok not in label_terms:
                            label_terms.append(tok)
            strat_name = det.strategy_name
            cid = stable_cluster_id(
                members, cluster_type="contradiction_cluster", strategy=strat_name
            )
            internal_weights: list[float] = []
            ms = sorted(members)
            for idx_i, m1 in enumerate(ms):
                for m2 in ms[idx_i + 1 :]:
                    pk = tuple(sorted((m1, m2)))
                    if pk in pair_weights:
                        internal_weights.append(pair_weights[pk])
            avg_w = sum(internal_weights) / len(internal_weights) if internal_weights else 0.0
            low_conf = sum(1 for m in members if by_id[m].confidence < 0.35) / len(members)
            contra_status = (
                sum(1 for m in members if by_id[m].status == BeliefStatus.CONTRADICTED)
                / len(members)
            )
            unresolved = 0.12
            if avg_w >= 0.5 and contra_status > 0.3:
                unresolved = 0.28
            c_score = self._clamp01(
                avg_w * 0.45 + low_conf * 0.20 + contra_status * 0.25 + unresolved * 0.10
            )
            mx = max(internal_weights) if internal_weights else avg_w
            activation = self._clamp01(0.04 + mx * 0.15)
            p_score = self._clamp01(activation * 0.35 + c_score * 0.45 + unresolved * 0.20)
            dom = "contradicted" if contra_status >= 0.5 else ("uncertain" if low_conf >= 0.5 else "mixed")

            rep_ids = sorted(members)[:3]
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO belief_communities (
                        cluster_id, cluster_type, member_count, dominant_status,
                        contradiction_score, activation_score, pressure_score, unresolved_score,
                        representative_belief_ids_json, top_terms_json,
                        created_at, updated_at, last_activated_at, metadata_json, community_detection_strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        "contradiction_cluster",
                        len(members),
                        dom,
                        c_score,
                        activation,
                        p_score,
                        unresolved,
                        json.dumps(rep_ids, sort_keys=True),
                        json.dumps(label_terms[:6], sort_keys=True),
                        now_iso,
                        now_iso,
                        now_iso,
                        json.dumps({}, sort_keys=True),
                        strat_name,
                    ),
                )
                for mb in members:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO belief_community_members (cluster_id, belief_id)
                        VALUES (?, ?)
                        """,
                        (cid, mb),
                    )
                connection.commit()
            created_ids.append(cid)

        return created_ids

    def update_belief_community_activation(
        self,
        *,
        activated_member_ids: Sequence[str],
        memory_cluster_activation: float = 0.0,
        max_edge_strength: float = 0.0,
    ) -> dict[str, float]:
        mids = sorted({str(x) for x in activated_member_ids if str(x)})
        now_iso = utcnow().isoformat()
        decay = 0.03
        memo_act = self._clamp01(float(memory_cluster_activation))

        if not mids and memo_act <= 0.0001:
            self._decay_all_contradiction_clusters(decay_amount=decay + 0.02, now_iso=now_iso)
            return {}

        touched: dict[str, float] = {}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM belief_communities WHERE cluster_type = 'contradiction_cluster'
                ORDER BY cluster_id ASC LIMIT ?
                """,
                (60,),
            ).fetchall()
            for raw in rows:
                row_dict = self._row_to_community(raw)
                cid = row_dict["cluster_id"]
                memb = set(self.iter_belief_community_member_ids(cid))
                active_member_ratio = (len(memb & set(mids)) / len(memb)) if memb else 0.0
                max_conflict = float(row_dict.get("contradiction_score") or 0.0)
                edge_component = self._clamp01(max_edge_strength * 0.25)
                recency_component = min(1.0, active_member_ratio * 2.5) * 0.05
                activation_score = self._clamp01(
                    active_member_ratio * 0.35
                    + max_conflict * 0.25
                    + edge_component
                    + memo_act * 0.10
                    + recency_component
                )
                unresolved_prev = float(row_dict.get("unresolved_score") or 0.0)
                c_score = float(row_dict.get("contradiction_score") or 0.0)
                if active_member_ratio > 0:
                    unresolved = self._clamp01(unresolved_prev + 0.04 * activation_score)
                else:
                    unresolved = self._clamp01(max(unresolved_prev - decay, 0.0))
                    activation_score = self._clamp01(max(activation_score - decay * 0.75, 0.0))

                pressure_score = self._clamp01(
                    activation_score * 0.35 + c_score * 0.45 + unresolved * 0.20
                )
                touched[cid] = pressure_score

                meta = dict(row_dict.get("metadata") or {})
                meta["activation_member_ratio_snapshot"] = round(active_member_ratio, 4)

                connection.execute(
                    """
                    UPDATE belief_communities SET
                        activation_score = ?,
                        unresolved_score = ?,
                        pressure_score = ?,
                        updated_at = ?,
                        last_activated_at = ?,
                        metadata_json = ?
                    WHERE cluster_id = ?
                    """,
                    (
                        activation_score,
                        unresolved,
                        pressure_score,
                        now_iso,
                        now_iso,
                        json.dumps(meta, sort_keys=True),
                        cid,
                    ),
                )
            connection.commit()
        return touched

    def _decay_all_contradiction_clusters(self, *, decay_amount: float, now_iso: str) -> None:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT cluster_id, activation_score, unresolved_score, contradiction_score "
                "FROM belief_communities WHERE cluster_type='contradiction_cluster' LIMIT ?",
                (40,),
            ).fetchall()
            for row in rows:
                us = max(float(row["unresolved_score"] or 0.0) - decay_amount, 0.0)
                activation = max(float(row["activation_score"] or 0.0) - decay_amount * 0.85, 0.0)
                c_score = float(row["contradiction_score"] or 0.0)
                ps = self._clamp01(
                    activation * 0.35 + c_score * 0.45 + us * 0.20
                )
                connection.execute(
                    """
                    UPDATE belief_communities SET
                        unresolved_score = ?,
                        activation_score = ?,
                        pressure_score = ?,
                        updated_at = ?
                    WHERE cluster_id = ?
                    """,
                    (us, activation, ps, now_iso, row["cluster_id"]),
                )
            connection.commit()

    def attach_memory_communities_to_belief(
        self,
        belief_id: str,
        memory_community_ids: Sequence[str],
        event_id: str | None = None,
    ) -> None:
        belief = self.get_belief(belief_id)
        if belief is None:
            return
        mid_list = sorted({str(x) for x in memory_community_ids if str(x)})[:40]
        if not mid_list:
            return
        merged = dict(belief.metadata)
        existing = merged.get("memory_community_ids")
        cur = set(existing if isinstance(existing, list) else [])
        cur.update(mid_list)
        merged["memory_community_ids"] = sorted(cur)[:40]
        if event_id:
            merged["memory_community_last_event_id"] = str(event_id)
        belief.metadata = merged
        self.update_belief(belief)

    def evaluate_belief_rules(
        self,
        *,
        beliefs: Sequence[Belief] | None = None,
        max_rule_evaluations: int = 50,
        max_inferred_per_cycle: int = 5,
        min_confidence_for_rule: float = 0.6,
    ) -> dict[str, Any]:
        from fullerene.world_model.rules import RuleEvalConfig, evaluate_enabled_rules_bounded

        raw_cand = list(beliefs) if beliefs is not None else self.list_beliefs(limit=120)
        cand = sorted(raw_cand, key=lambda x: (-x.updated_at.timestamp(), x.id))[:120]
        cfg = RuleEvalConfig(
            max_rule_evaluations=max_rule_evaluations,
            max_inferred_beliefs_per_cycle=max_inferred_per_cycle,
            min_confidence_for_rule=min_confidence_for_rule,
        )
        return evaluate_enabled_rules_bounded(self, cand, config=cfg)

    def merge_belief_metadata(self, belief_id: str, updates: dict[str, Any]) -> Belief | None:
        belief = self.get_belief(belief_id)
        if belief is None:
            return None
        m = dict(belief.metadata)
        m.update(updates or {})
        belief.metadata = m
        self.update_belief(belief)
        return belief

    def get_belief_by_normalized_key(self, normalized_key: str) -> Belief | None:
        key = normalize_statement(normalized_key)
        if not key:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id, claim, confidence, status, tags_json, source, source_event_id, source_memory_id,
                    created_at, updated_at, metadata_json, sources_json, normalized_key, belief_type,
                    support_count, contradiction_count, last_support_event_id, last_contradiction_event_id,
                    last_updated_event_id, priority
                FROM beliefs
                WHERE normalized_key = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        return self._row_to_belief(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS beliefs (
                    id TEXT PRIMARY KEY,
                    claim TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_event_id TEXT,
                    source_memory_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    normalized_key TEXT NOT NULL DEFAULT '',
                    belief_type TEXT NOT NULL DEFAULT 'unknown',
                    support_count INTEGER NOT NULL DEFAULT 0,
                    contradiction_count INTEGER NOT NULL DEFAULT 0,
                    last_support_event_id TEXT,
                    last_contradiction_event_id TEXT,
                    last_updated_event_id TEXT,
                    priority REAL NOT NULL DEFAULT 1.0,
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    CHECK (
                        status IN ('valid', 'contradicted', 'redundant', 'active', 'stale', 'retired')
                    ),
                    CHECK (source IN ('user', 'system', 'memory', 'goal', 'context', 'runtime'))
                )
                """
            )
            self._migrate_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beliefs_status_confidence_updated
                ON beliefs (status, confidence DESC, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beliefs_normalized_key
                ON beliefs (normalized_key)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS belief_edges (
                    source_belief_id TEXT NOT NULL,
                    target_belief_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (source_belief_id, target_belief_id, edge_type),
                    FOREIGN KEY (source_belief_id) REFERENCES beliefs(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_belief_id) REFERENCES beliefs(id) ON DELETE CASCADE
                )
                """
            )
            self._migrate_belief_edges_columns(connection)
            self._create_world_model_v2_tables(connection)
            self._backfill_belief_edge_ids(connection)
            connection.commit()

    @staticmethod
    def _migrate_columns(connection: sqlite3.Connection) -> None:
        existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(beliefs)").fetchall()}
        migrations = {
            "sources_json": "ALTER TABLE beliefs ADD COLUMN sources_json TEXT NOT NULL DEFAULT '[]'",
            "normalized_key": "ALTER TABLE beliefs ADD COLUMN normalized_key TEXT NOT NULL DEFAULT ''",
            "belief_type": "ALTER TABLE beliefs ADD COLUMN belief_type TEXT NOT NULL DEFAULT 'unknown'",
            "support_count": "ALTER TABLE beliefs ADD COLUMN support_count INTEGER NOT NULL DEFAULT 0",
            "contradiction_count": "ALTER TABLE beliefs ADD COLUMN contradiction_count INTEGER NOT NULL DEFAULT 0",
            "last_support_event_id": "ALTER TABLE beliefs ADD COLUMN last_support_event_id TEXT",
            "last_contradiction_event_id": "ALTER TABLE beliefs ADD COLUMN last_contradiction_event_id TEXT",
            "last_updated_event_id": "ALTER TABLE beliefs ADD COLUMN last_updated_event_id TEXT",
            "priority": "ALTER TABLE beliefs ADD COLUMN priority REAL NOT NULL DEFAULT 1.0",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                connection.execute(statement)

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    @staticmethod
    def _migrate_belief_edges_columns(connection: sqlite3.Connection) -> None:
        cols = SQLiteWorldModelStore._table_columns(connection, "belief_edges")
        alters: list[tuple[str, str]] = [
            ("edge_id", "TEXT"),
            ("support_count", "INTEGER NOT NULL DEFAULT 0"),
            ("contradiction_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_activated_at", "TEXT"),
            ("provenance_json", "TEXT NOT NULL DEFAULT '{}'"),
        ]
        for name, decl in alters:
            if name not in cols:
                connection.execute(f"ALTER TABLE belief_edges ADD COLUMN {name} {decl}")

    @staticmethod
    def _create_world_model_v2_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS belief_rules (
                rule_id TEXT PRIMARY KEY,
                rule_type TEXT NOT NULL,
                antecedent_patterns_json TEXT NOT NULL,
                consequent_pattern TEXT NOT NULL,
                confidence_weight REAL NOT NULL DEFAULT 0.5,
                historical_validity REAL NOT NULL DEFAULT 0.5,
                support_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS belief_communities (
                cluster_id TEXT PRIMARY KEY,
                cluster_type TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                dominant_status TEXT NOT NULL DEFAULT 'mixed',
                contradiction_score REAL NOT NULL DEFAULT 0.0,
                activation_score REAL NOT NULL DEFAULT 0.0,
                pressure_score REAL NOT NULL DEFAULT 0.0,
                unresolved_score REAL NOT NULL DEFAULT 0.0,
                representative_belief_ids_json TEXT NOT NULL DEFAULT '[]',
                top_terms_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_activated_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                community_detection_strategy TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS belief_community_members (
                cluster_id TEXT NOT NULL,
                belief_id TEXT NOT NULL,
                PRIMARY KEY (cluster_id, belief_id),
                FOREIGN KEY (cluster_id) REFERENCES belief_communities(cluster_id) ON DELETE CASCADE,
                FOREIGN KEY (belief_id) REFERENCES beliefs(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_edges_edge_id ON belief_edges(edge_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_community_members_belief ON belief_community_members(belief_id)"
        )

    @staticmethod
    def _backfill_belief_edge_ids(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT source_belief_id, target_belief_id, edge_type, edge_id FROM belief_edges
            """
        ).fetchall()
        for row in rows:
            eid = row["edge_id"]
            if eid:
                continue
            nid = stable_belief_edge_id(
                row["source_belief_id"], row["target_belief_id"], row["edge_type"]
            )
            connection.execute(
                "UPDATE belief_edges SET edge_id = ? WHERE source_belief_id = ? AND target_belief_id = ? AND edge_type = ?",
                (nid, row["source_belief_id"], row["target_belief_id"], row["edge_type"]),
            )

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(int(limit), 1)

    @staticmethod
    def _row_to_belief(row: sqlite3.Row) -> Belief:
        return Belief.from_dict(
            {
                "id": row["id"],
                "claim": row["claim"],
                "confidence": row["confidence"],
                "status": row["status"],
                "tags": json.loads(row["tags_json"]),
                "source": row["source"],
                "source_event_id": row["source_event_id"],
                "source_memory_id": row["source_memory_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata_json"]),
                "sources": json.loads(row["sources_json"] or "[]"),
                "normalized_key": row["normalized_key"],
                "belief_type": row["belief_type"],
                "support_count": row["support_count"],
                "contradiction_count": row["contradiction_count"],
                "last_support_event_id": row["last_support_event_id"],
                "last_contradiction_event_id": row["last_contradiction_event_id"],
                "last_updated_event_id": row["last_updated_event_id"],
                "priority": row["priority"],
            }
        )
