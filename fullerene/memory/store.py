"""SQLite-backed memory store for Fullerene memory records.

Memory v1 stored canonical episodic records and supported deterministic
keyword/tag/salience/recency retrieval. Memory v2 keeps every Memory v1
guarantee and adds:

- Optional ``role`` and ``domain`` columns on ``memories`` so role-aware
  retrieval can sort without re-classifying on every read.
- A ``memory_embeddings`` table that stores optional embedding vectors as
  an *index* into the canonical ``memories`` rows. SQLite remains the
  source of truth; missing embedding rows must always fall back to
  deterministic v1 retrieval.
- A ``memory_edges`` table for write-time edges. Memory v2 only writes
  edges; retrieval does not yet traverse the graph.
- A ``hybrid_retrieve_relevant`` helper that scores a bounded candidate set
  using :mod:`fullerene.memory.hybrid` and returns ranked results with a
  per-record score breakdown.

Schema migrations are additive: when the database was created by Memory v1
or earlier, the new columns are added with defaults so existing records
remain readable without rewrite.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from fullerene.memory.communities import MemoryCommunity
from fullerene.memory.community_detection import (
    DeterministicConnectedComponentsDetector,
    aggregate_top_tags_domains_roles,
    label_from_tags_domains,
    stable_community_id,
)
from fullerene.memory.edges import MemoryEdge, MemoryEdgeType
from fullerene.memory.embeddings import deserialize_vector, serialize_vector
from fullerene.memory.hybrid import explain_hybrid_score, hybrid_sort_key
from fullerene.memory import v3 as memory_v3_formula
from fullerene.memory.models import MemoryLayer, MemoryRecord, MemoryType
from fullerene.memory.scoring import score_sort_key, tokenize
from fullerene.memory.roles import QueryIntent, classify_query_intent
from fullerene.nexus.models import Event


class MemoryStore(Protocol):
    def add_memory(self, record: MemoryRecord) -> None:
        """Persist a memory record."""

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a memory record by id."""

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """Return the newest records, optionally filtered by memory type."""

    def search_keyword(self, query: str, limit: int) -> list[MemoryRecord]:
        """Return recent records whose content matches the query tokens."""

    def retrieve_relevant(self, event: Event, limit: int) -> list[MemoryRecord]:
        """Return a bounded, deterministically ranked set of related memories."""

    def update_memory_salience(self, memory_id: str, salience: float) -> None:
        """Persist a salience-only edit to an existing memory record."""

    def strengthen_memory_edge(
        self,
        source_memory_id: str,
        target_memory_id: str,
        edge_type: MemoryEdgeType,
        delta: float,
        *,
        reason: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Increase (or create) a bounded write-time edge weight by ``delta``."""

    def add_working_turn(
        self,
        *,
        content: str,
        session_id: str,
        turn_index: int,
        dialogue_role: str,
        source_event_id: str | None = None,
        created_from: str = "interactive",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Persist one exact working-memory dialogue turn."""

    def list_working_turns(self, session_id: str, limit: int = 8) -> list[MemoryRecord]:
        """Return recent working turns for one session in chronological order."""

    def prune_working_memory(self, session_id: str, keep_last: int = 20) -> int:
        """Delete older working turns for a session and return removed count."""


class SQLiteMemoryStore:
    """SQLite-backed source of truth for Fullerene memory.

    The store is additive across versions: v1 callers do not need to know
    about role/domain/embeddings/edges, and v2 callers can opt into the new
    surfaces without breaking older data files.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    # ---- Memory v1 surface (kept compatible) ---------------------------

    def add_memory(self, record: MemoryRecord) -> None:
        payload = record.to_dict()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    id,
                    created_at,
                    memory_type,
                    content,
                    source_event_id,
                    salience,
                    confidence,
                    tags_json,
                    metadata_json,
                    role,
                    domain,
                    memory_layer,
                    community_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["created_at"],
                    payload["memory_type"],
                    payload["content"],
                    payload["source_event_id"],
                    payload["salience"],
                    payload["confidence"],
                    json.dumps(payload["tags"], sort_keys=True),
                    json.dumps(payload["metadata"], sort_keys=True),
                    payload.get("role") or "unknown",
                    payload.get("domain"),
                    payload.get("memory_layer", MemoryLayer.LONG_TERM.value),
                    payload.get("community_id"),
                ),
            )
            connection.commit()

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT {self._SELECT_MEMORY_COLUMNS} FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        bounded_limit = self._normalize_limit(limit)
        query = f"SELECT {self._SELECT_MEMORY_COLUMNS} FROM memories"
        params: list[object] = []
        query += " WHERE memory_layer != ?"
        params.append(MemoryLayer.WORKING.value)
        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search_keyword(self, query: str, limit: int) -> list[MemoryRecord]:
        bounded_limit = self._normalize_limit(limit)
        tokens = sorted(tokenize(query))
        if not tokens:
            return []

        clauses = " OR ".join("lower(content) LIKE ?" for _ in tokens)
        params: list[object] = [f"%{token}%" for token in tokens]
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT {self._SELECT_MEMORY_COLUMNS}
                FROM memories
                WHERE memory_layer != ? AND ({clauses})
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                [MemoryLayer.WORKING.value, *params],
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def retrieve_relevant(self, event: Event, limit: int) -> list[MemoryRecord]:
        bounded_limit = self._normalize_limit(limit)
        candidate_limit = max(bounded_limit * 8, 32)

        candidates: dict[str, MemoryRecord] = {}
        for record in self.list_recent(limit=candidate_limit):
            candidates[record.id] = record
        for record in self.search_keyword(event.content, limit=candidate_limit):
            candidates[record.id] = record

        ranked = sorted(
            candidates.values(),
            key=lambda memory: score_sort_key(event, memory),
            reverse=True,
        )
        return ranked[:bounded_limit]

    def update_memory_salience(self, memory_id: str, salience: float) -> None:
        normalized_salience = MemoryRecord._validate_score("salience", salience)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE memories
                SET salience = ?
                WHERE id = ?
                """,
                (normalized_salience, memory_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Memory {memory_id!r} does not exist")
            connection.commit()

    def merge_memory_metadata(self, memory_id: str, patch: dict[str, Any]) -> None:
        """Merge JSON-safe keys into episodic metadata (Memory v3 contradiction seam)."""
        rec = self.get_memory(memory_id)
        if rec is None:
            raise KeyError(f"Memory {memory_id!r} does not exist")
        md = dict(rec.metadata or {})
        for k, v in (patch or {}).items():
            md[k] = v
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE memories SET metadata_json = ? WHERE id = ?
                """,
                (json.dumps(md, sort_keys=True), memory_id),
            )
            connection.commit()

    # ---- Memory v2 surface ---------------------------------------------

    def list_high_salience(
        self,
        limit: int,
        *,
        salience_threshold: float = 0.6,
    ) -> list[MemoryRecord]:
        """Return the newest high-salience records up to ``limit``.

        Used by Memory v2 to pick a bounded edge candidate set without
        traversing the full graph at retrieval time.
        """
        bounded_limit = self._normalize_limit(limit)
        threshold = max(0.0, min(float(salience_threshold), 1.0))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT {self._SELECT_MEMORY_COLUMNS}
                FROM memories
                WHERE memory_layer != ? AND salience >= ?
                ORDER BY salience DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (MemoryLayer.WORKING.value, threshold, bounded_limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_by_domain(
        self,
        domain: str,
        limit: int,
    ) -> list[MemoryRecord]:
        """Return the newest records sharing ``domain`` up to ``limit``."""
        bounded_limit = self._normalize_limit(limit)
        cleaned_domain = (domain or "").strip().lower()
        if not cleaned_domain:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT {self._SELECT_MEMORY_COLUMNS}
                FROM memories
                WHERE memory_layer != ? AND domain = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (MemoryLayer.WORKING.value, cleaned_domain, bounded_limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def add_memory_embedding(
        self,
        *,
        memory_id: str,
        model: str,
        vector: Sequence[float],
        created_at: str | None = None,
    ) -> None:
        """Persist a single embedding row keyed by ``(memory_id, model)``."""
        if not vector:
            return
        from fullerene.memory.models import utcnow

        timestamp = created_at or utcnow().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                    memory_id,
                    model,
                    vector_json,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    memory_id,
                    model,
                    serialize_vector(vector),
                    timestamp,
                ),
            )
            connection.commit()

    def get_memory_embedding(
        self,
        memory_id: str,
        *,
        model: str | None = None,
    ) -> tuple[list[float] | None, str | None]:
        """Return ``(vector, model)`` for the requested memory.

        When ``model`` is omitted the most recently stored embedding wins.
        """
        with closing(self._connect()) as connection:
            if model is not None:
                row = connection.execute(
                    """
                    SELECT model, vector_json
                    FROM memory_embeddings
                    WHERE memory_id = ? AND model = ?
                    """,
                    (memory_id, model),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT model, vector_json
                    FROM memory_embeddings
                    WHERE memory_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (memory_id,),
                ).fetchone()
        if row is None:
            return None, None
        return deserialize_vector(row["vector_json"]), row["model"]

    def list_memory_embeddings(
        self,
        memory_ids: Iterable[str],
        *,
        model: str | None = None,
    ) -> dict[str, list[float]]:
        """Bulk-load embeddings for the given memory ids."""
        ids = sorted({memory_id for memory_id in memory_ids if memory_id})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        params: list[object] = list(ids)
        sql = (
            "SELECT memory_id, model, vector_json FROM memory_embeddings "
            f"WHERE memory_id IN ({placeholders})"
        )
        if model is not None:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY created_at ASC"
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, params).fetchall()
        vectors: dict[str, list[float]] = {}
        for row in rows:
            decoded = deserialize_vector(row["vector_json"])
            if decoded is None:
                continue
            # Newest row wins because we sort ascending and overwrite.
            vectors[row["memory_id"]] = decoded
        return vectors

    def add_memory_edge(self, edge: MemoryEdge) -> None:
        payload = edge.to_dict()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_edges (
                    id,
                    source_memory_id,
                    target_memory_id,
                    edge_type,
                    weight,
                    created_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["source_memory_id"],
                    payload["target_memory_id"],
                    payload["edge_type"],
                    payload["weight"],
                    payload["created_at"],
                    json.dumps(payload["metadata"], sort_keys=True),
                ),
            )
            connection.commit()

    def strengthen_memory_edge(
        self,
        source_memory_id: str,
        target_memory_id: str,
        edge_type: MemoryEdgeType,
        delta: float,
        *,
        reason: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Apply a bounded Hebbian-style weight increment to an existing or new edge.

        Canonical orientation uses lexicographic ordering on memory ids so the
        same undirected pair always maps to one directed row.
        """
        a, b = source_memory_id, target_memory_id
        src, tgt = (a, b) if a < b else (b, a)
        if src == tgt:
            raise ValueError("source and target memory ids must differ")
        delta_clamped = max(-1.0, min(float(delta), 1.0))
        prov = dict(provenance or {})
        if reason:
            prov.setdefault("reason", reason)
        meta_merge: dict[str, Any] = {"learning_v1": prov}

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, weight, metadata_json
                FROM memory_edges
                WHERE source_memory_id = ? AND target_memory_id = ? AND edge_type = ?
                """,
                (src, tgt, edge_type.value),
            ).fetchone()
            if row is None:
                new_weight = MemoryRecord._validate_score(
                    "weight",
                    max(0.0, min(1.0, delta_clamped)),
                )
                meta = dict(meta_merge)
                edge_id = uuid4().hex
                timestamp = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT INTO memory_edges (
                        id,
                        source_memory_id,
                        target_memory_id,
                        edge_type,
                        weight,
                        created_at,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        src,
                        tgt,
                        edge_type.value,
                        new_weight,
                        timestamp,
                        json.dumps(meta, sort_keys=True),
                    ),
                )
                connection.commit()
                return MemoryEdge(
                    id=edge_id,
                    source_memory_id=src,
                    target_memory_id=tgt,
                    edge_type=edge_type,
                    weight=new_weight,
                    created_at=datetime.now(timezone.utc),
                    metadata=meta,
                )

            old_weight = float(row["weight"])
            old_meta: dict[str, Any] = {}
            if row["metadata_json"]:
                try:
                    old_meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    old_meta = {}
            merged_meta = {**old_meta, **meta_merge}
            new_weight = MemoryRecord._validate_score(
                "weight",
                max(0.0, min(1.0, old_weight + delta_clamped)),
            )
            connection.execute(
                """
                UPDATE memory_edges
                SET weight = ?, metadata_json = ?
                WHERE id = ?
                """,
                (new_weight, json.dumps(merged_meta, sort_keys=True), row["id"]),
            )
            connection.commit()
            refreshed = connection.execute(
                """
                SELECT id, source_memory_id, target_memory_id, edge_type, weight,
                       created_at, metadata_json
                FROM memory_edges WHERE id = ?
                """,
                (row["id"],),
            ).fetchone()
            assert refreshed is not None
            return MemoryEdge.from_dict(
                {
                    "id": refreshed["id"],
                    "source_memory_id": refreshed["source_memory_id"],
                    "target_memory_id": refreshed["target_memory_id"],
                    "edge_type": refreshed["edge_type"],
                    "weight": refreshed["weight"],
                    "created_at": refreshed["created_at"],
                    "metadata": json.loads(refreshed["metadata_json"] or "{}"),
                }
            )

    def list_memory_edges(
        self,
        *,
        memory_id: str | None = None,
        edge_types: Iterable[MemoryEdgeType] | None = None,
        limit: int = 50,
    ) -> list[MemoryEdge]:
        bounded_limit = self._normalize_limit(limit)
        clauses: list[str] = []
        params: list[object] = []
        if memory_id is not None:
            clauses.append("(source_memory_id = ? OR target_memory_id = ?)")
            params.extend([memory_id, memory_id])
        if edge_types is not None:
            edge_type_values = [edge_type.value for edge_type in edge_types]
            if edge_type_values:
                placeholders = ",".join("?" for _ in edge_type_values)
                clauses.append(f"edge_type IN ({placeholders})")
                params.extend(edge_type_values)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, source_memory_id, target_memory_id, edge_type, weight, "
            "created_at, metadata_json FROM memory_edges "
            f"{where_clause} ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            MemoryEdge.from_dict(
                {
                    "id": row["id"],
                    "source_memory_id": row["source_memory_id"],
                    "target_memory_id": row["target_memory_id"],
                    "edge_type": row["edge_type"],
                    "weight": row["weight"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
            for row in rows
        ]

    def hybrid_retrieve_relevant(
        self,
        event: Event,
        *,
        limit: int,
        embedding_provider_name: str | None = None,
        event_vector: Sequence[float] | None = None,
        candidate_limit: int | None = None,
        domain_hint: str | None = None,
    ) -> list[tuple[MemoryRecord, dict[str, Any]]]:
        """Return ranked ``(record, breakdown)`` tuples using hybrid scoring.

        SQLite is still the source of truth: we collect a bounded candidate
        set from recent + keyword + same-domain rows, attach any stored
        embedding vectors, score with :func:`explain_hybrid_score`, and
        return the top ``limit`` rows. When embeddings are missing the
        semantic component is ``0.0`` and retrieval degrades smoothly to
        deterministic v1 plus role-aware bonuses.
        """
        bounded_limit = self._normalize_limit(limit)
        candidate_size = candidate_limit or max(bounded_limit * 8, 32)

        candidates: dict[str, MemoryRecord] = {}
        for record in self.list_recent(limit=candidate_size):
            candidates[record.id] = record
        for record in self.search_keyword(event.content, limit=candidate_size):
            candidates[record.id] = record
        for record in self.list_high_salience(limit=candidate_size):
            candidates[record.id] = record
        if domain_hint:
            for record in self.list_by_domain(domain_hint, limit=candidate_size):
                candidates[record.id] = record

        memory_ids = list(candidates.keys())
        embedding_index = (
            self.list_memory_embeddings(memory_ids, model=embedding_provider_name)
            if embedding_provider_name
            else {}
        )

        intent: QueryIntent = classify_query_intent(event.content)

        scored: list[tuple[MemoryRecord, dict[str, Any]]] = []
        for record in candidates.values():
            memory_vector = embedding_index.get(record.id)
            breakdown = explain_hybrid_score(
                event,
                record,
                event_vector=event_vector,
                memory_vector=memory_vector,
                query_intent=intent,
                event_domain=domain_hint,
            )
            scored.append((record, breakdown))

        scored.sort(
            key=lambda pair: (
                float(pair[1]["total"]),
                pair[0].created_at.timestamp(),
                pair[0].id,
            ),
            reverse=True,
        )
        return scored[:bounded_limit]

    def add_working_turn(
        self,
        *,
        content: str,
        session_id: str,
        turn_index: int,
        dialogue_role: str,
        source_event_id: str | None = None,
        created_from: str = "interactive",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        cleaned_content = str(content or "").strip()
        if not cleaned_content:
            raise ValueError("Working turn content must not be empty")
        record = MemoryRecord(
            memory_type=MemoryType.WORKING,
            memory_layer=MemoryLayer.WORKING,
            content=cleaned_content,
            source_event_id=source_event_id,
            salience=0.5,
            confidence=1.0,
            metadata={
                "session_id": session_id,
                "turn_index": int(turn_index),
                "dialogue_role": str(dialogue_role),
                "created_from": str(created_from or "interactive"),
                **(metadata or {}),
            },
            role="dialogue_turn",
            domain="conversation",
        )
        self.add_memory(record)
        return record

    def list_working_turns(self, session_id: str, limit: int = 8) -> list[MemoryRecord]:
        bounded_limit = self._normalize_limit(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT {self._SELECT_MEMORY_COLUMNS}
                FROM memories
                WHERE memory_layer = ?
                  AND json_extract(metadata_json, '$.session_id') = ?
                ORDER BY json_extract(metadata_json, '$.turn_index') DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (MemoryLayer.WORKING.value, session_id, bounded_limit),
            ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        records.sort(key=lambda r: int((r.metadata or {}).get("turn_index", 0)))
        return records

    def prune_working_memory(self, session_id: str, keep_last: int = 20) -> int:
        keep = self._normalize_limit(keep_last)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM memories
                WHERE memory_layer = ?
                  AND json_extract(metadata_json, '$.session_id') = ?
                ORDER BY json_extract(metadata_json, '$.turn_index') DESC, created_at DESC, id DESC
                """,
                (MemoryLayer.WORKING.value, session_id),
            ).fetchall()
            ids = [row["id"] for row in rows]
            stale_ids = ids[keep:]
            if not stale_ids:
                return 0
            placeholders = ",".join("?" for _ in stale_ids)
            cursor = connection.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                stale_ids,
            )
            connection.commit()
            return int(cursor.rowcount or 0)

    # ---- Memory v3 communities ------------------------------------------

    @staticmethod
    def normalize_edge_weight(edge: MemoryEdge) -> float:
        """Normalize edge contribution to [0,1] for community detection."""
        w = float(edge.weight)
        boost = {
            MemoryEdgeType.SAME_GOAL.value: 1.0,
            MemoryEdgeType.SAME_DOMAIN.value: 0.95,
            MemoryEdgeType.SEMANTIC_SIMILARITY.value: 1.0,
            MemoryEdgeType.TAG_OVERLAP.value: 0.85,
            MemoryEdgeType.KEYWORD_SIMILARITY.value: 0.9,
            MemoryEdgeType.TEMPORAL_PROXIMITY.value: 0.75,
            MemoryEdgeType.ROLE_RELATED.value: 0.7,
        }.get(edge.edge_type.value, 0.8)
        return memory_v3_formula.clamp01(w * boost)

    def _list_edges_raw_bounded(self, limit: int) -> list[tuple[str, str, str, float]]:
        bounded = self._normalize_limit(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT source_memory_id, target_memory_id, edge_type, weight
                FROM memory_edges
                ORDER BY weight DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        out: list[tuple[str, str, str, float]] = []
        for row in rows:
            e = MemoryEdge(
                source_memory_id=row["source_memory_id"],
                target_memory_id=row["target_memory_id"],
                edge_type=MemoryEdgeType(row["edge_type"]),
                weight=float(row["weight"]),
            )
            nw = self.normalize_edge_weight(e)
            out.append(
                (row["source_memory_id"], row["target_memory_id"], row["edge_type"], nw)
            )
        return out

    def _long_term_ids_bounded(self, limit: int) -> list[str]:
        bounded = self._normalize_limit(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT id FROM memories
                WHERE memory_layer != ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (MemoryLayer.WORKING.value, bounded),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def rebuild_memory_communities(
        self,
        limit: int | None = None,
        *,
        detection_strategy: str = "deterministic_connected_components_v0",
    ) -> int:
        """Recompute communities from stored edges (bounded). Returns count written."""
        edge_cap = limit if limit is not None else 5000
        edges = self._list_edges_raw_bounded(edge_cap)
        mem_cap = limit if limit is not None else 20000
        memory_ids = self._long_term_ids_bounded(mem_cap)
        if not memory_ids:
            return 0

        tag_map: dict[str, list[str]] = {}
        domain_map: dict[str, str | None] = {}
        role_map: dict[str, str] = {}

        def tags_of(mid: str) -> list[str]:
            if mid not in tag_map:
                r = self.get_memory(mid)
                tag_map[mid] = list(r.tags) if r else []
            return tag_map[mid]

        def domain_of(mid: str) -> str | None:
            if mid not in domain_map:
                r = self.get_memory(mid)
                domain_map[mid] = r.domain if r else None
            return domain_map[mid]

        def role_of(mid: str) -> str:
            if mid not in role_map:
                r = self.get_memory(mid)
                role_map[mid] = (r.role or "unknown") if r else "unknown"
            return role_map[mid]

        detector = DeterministicConnectedComponentsDetector()
        components = detector.detect(
            memory_ids=memory_ids,
            edges=edges,
            memory_tags={m: tags_of(m) for m in memory_ids},
            memory_domains={m: domain_of(m) for m in memory_ids},
        )
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM memory_community_members")
            connection.execute("DELETE FROM memory_communities")
            connection.execute(
                "UPDATE memories SET community_id = NULL WHERE memory_layer != ?",
                (MemoryLayer.WORKING.value,),
            )
            count = 0
            for comp in components:
                members = sorted(comp)
                if not members:
                    continue
                top_tags, top_doms, top_roles = aggregate_top_tags_domains_roles(
                    members,
                    tags_of,
                    domain_of,
                    role_of,
                )
                cid = stable_community_id(
                    members,
                    top_tags,
                    strategy=detection_strategy,
                )
                label = label_from_tags_domains(top_tags, top_doms)
                rep_ids = members[:3]
                payload = MemoryCommunity(
                    community_id=cid,
                    label=label,
                    member_memory_ids=list(members),
                    member_count=len(members),
                    top_tags=top_tags,
                    top_domains=top_doms,
                    top_roles=top_roles,
                    representative_memory_ids=rep_ids,
                    community_detection_strategy=detection_strategy,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                self._insert_community_row(connection, payload, top_tags, top_doms, top_roles, rep_ids)
                for mid in members:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO memory_community_members (community_id, memory_id)
                        VALUES (?, ?)
                        """,
                        (cid, mid),
                    )
                    connection.execute(
                        "UPDATE memories SET community_id = ? WHERE id = ?",
                        (cid, mid),
                    )
                count += 1
            connection.commit()
        return count

    @staticmethod
    def _insert_community_row(
        connection: sqlite3.Connection,
        community: MemoryCommunity,
        top_tags: list[str],
        top_doms: list[str],
        top_roles: list[str],
        rep_ids: list[str],
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_communities (
                community_id, label, member_count,
                top_tags_json, top_domains_json, top_roles_json,
                representative_memory_ids_json,
                centroid_embedding_id, centroid_vector_hash,
                activation_score, pressure_score, unresolved_score,
                contradiction_count, refinement_count,
                activation_streak, inactive_streak,
                last_activated_at, last_activated_event_id,
                last_pressure_update_at, last_resolution_event_id,
                resolved_recently,
                activation_reasons_json, pressure_reasons_json,
                community_detection_strategy,
                created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                community.community_id,
                community.label,
                int(community.member_count),
                json.dumps(top_tags, sort_keys=True),
                json.dumps(top_doms, sort_keys=True),
                json.dumps(top_roles, sort_keys=True),
                json.dumps(rep_ids, sort_keys=True),
                community.centroid_embedding_id,
                community.centroid_vector_hash,
                float(community.activation_score),
                float(community.pressure_score),
                float(community.unresolved_score),
                int(community.contradiction_count),
                int(community.refinement_count),
                int(community.activation_streak),
                int(community.inactive_streak),
                community.last_activated_at.isoformat() if community.last_activated_at else None,
                community.last_activated_event_id,
                community.last_pressure_update_at.isoformat()
                if community.last_pressure_update_at
                else None,
                community.last_resolution_event_id,
                1 if community.resolved_recently else 0,
                json.dumps(community.activation_reasons, sort_keys=True),
                json.dumps(community.pressure_reasons, sort_keys=True),
                community.community_detection_strategy,
                community.created_at.isoformat(),
                community.updated_at.isoformat(),
                json.dumps(community.metadata, sort_keys=True),
            ),
        )

    def _row_to_memory_community(self, row: sqlite3.Row) -> MemoryCommunity:
        def _parse_dt(val: object) -> datetime | None:
            if not val:
                return None
            return datetime.fromisoformat(str(val))

        return MemoryCommunity(
            community_id=str(row["community_id"]),
            label=str(row["label"] or ""),
            member_count=int(row["member_count"] or 0),
            member_memory_ids=[],
            top_tags=json.loads(row["top_tags_json"] or "[]"),
            top_domains=json.loads(row["top_domains_json"] or "[]"),
            top_roles=json.loads(row["top_roles_json"] or "[]"),
            representative_memory_ids=json.loads(row["representative_memory_ids_json"] or "[]"),
            centroid_embedding_id=row["centroid_embedding_id"],
            centroid_vector_hash=row["centroid_vector_hash"],
            activation_score=float(row["activation_score"] or 0.0),
            pressure_score=float(row["pressure_score"] or 0.0),
            unresolved_score=float(row["unresolved_score"] or 0.0),
            contradiction_count=int(row["contradiction_count"] or 0),
            refinement_count=int(row["refinement_count"] or 0),
            activation_streak=int(row["activation_streak"] or 0),
            inactive_streak=int(row["inactive_streak"] or 0),
            last_activated_at=_parse_dt(row["last_activated_at"]),
            last_activated_event_id=row["last_activated_event_id"],
            last_pressure_update_at=_parse_dt(row["last_pressure_update_at"]),
            last_resolution_event_id=row["last_resolution_event_id"],
            resolved_recently=bool(row["resolved_recently"]),
            activation_reasons=json.loads(row["activation_reasons_json"] or "[]"),
            pressure_reasons=json.loads(row["pressure_reasons_json"] or "[]"),
            community_detection_strategy=str(
                row["community_detection_strategy"]
                or "deterministic_connected_components_v0"
            ),
            created_at=_parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get_memory_community(self, community_id: str) -> MemoryCommunity | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_communities WHERE community_id = ?",
                (community_id,),
            ).fetchone()
        if row is None:
            return None
        mc = self._row_to_memory_community(row)
        with closing(self._connect()) as connection:
            mids = connection.execute(
                """
                SELECT memory_id FROM memory_community_members
                WHERE community_id = ?
                ORDER BY memory_id ASC
                """,
                (community_id,),
            ).fetchall()
        mc.member_memory_ids = [str(r["memory_id"]) for r in mids]
        return mc

    def list_communities_for_memory(self, memory_id: str) -> list[MemoryCommunity]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.* FROM memory_communities c
                INNER JOIN memory_community_members m ON m.community_id = c.community_id
                WHERE m.memory_id = ?
                """,
                (memory_id,),
            ).fetchall()
        return [self._row_to_memory_community(r) for r in rows]

    def list_memory_communities(
        self,
        *,
        limit: int = 50,
        min_activation: float = 0.0,
        min_pressure: float = 0.0,
    ) -> list[MemoryCommunity]:
        bounded = self._normalize_limit(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_communities
                WHERE activation_score >= ? AND pressure_score >= ?
                ORDER BY activation_score DESC, pressure_score DESC, community_id ASC
                LIMIT ?
                """,
                (float(min_activation), float(min_pressure), bounded),
            ).fetchall()
        return [self._row_to_memory_community(r) for r in rows]

    def get_memory_neighbors(
        self,
        memory_id: str,
        *,
        depth: int = 1,
        limit: int = 40,
    ) -> list[str]:
        d = min(max(int(depth), 1), 2)
        cap = self._normalize_limit(limit)
        seen: set[str] = {memory_id}
        frontier = {memory_id}
        for _ in range(d):
            next_front: set[str] = set()
            for mid in sorted(frontier):
                for edge in self.list_memory_edges(memory_id=mid, limit=cap):
                    other = (
                        edge.target_memory_id
                        if edge.source_memory_id == mid
                        else edge.source_memory_id
                    )
                    if other not in seen:
                        seen.add(other)
                        next_front.add(other)
                    if len(seen) - 1 >= cap:
                        break
                if len(seen) - 1 >= cap:
                    break
            frontier = next_front
            if len(seen) - 1 >= cap:
                break
        out = [m for m in sorted(seen) if m != memory_id]
        return out[:cap]

    def update_memory_communities_for_new_memory(self, memory_id: str) -> str | None:
        """Assign a new long-term memory to a community using bounded neighborhood."""
        rec = self.get_memory(memory_id)
        if rec is None or rec.memory_layer == MemoryLayer.WORKING:
            return None
        edges = self.list_memory_edges(memory_id=memory_id, limit=60)
        best_neighbor: str | None = None
        best_score = 0.0
        for edge in edges:
            other = (
                edge.target_memory_id
                if edge.source_memory_id == memory_id
                else edge.source_memory_id
            )
            nw = self.normalize_edge_weight(edge)
            if nw > best_score:
                best_score = nw
                best_neighbor = other
        if best_neighbor is None or best_score < 0.12:
            cid = self._ensure_singleton_community(memory_id)
            return cid
        other_rec = self.get_memory(best_neighbor)
        if other_rec is None:
            cid = self._ensure_singleton_community(memory_id)
            return cid
        oc = other_rec.community_id
        if oc:
            self._add_member_to_community(oc, memory_id)
            return oc
        cid = self._ensure_singleton_community(memory_id)
        return cid

    def _ensure_singleton_community(self, memory_id: str) -> str:
        rec = self.get_memory(memory_id)
        tags = list(rec.tags) if rec else []
        dom = rec.domain if rec else None
        cid = stable_community_id([memory_id], tags, strategy="deterministic_connected_components_v0")
        label = label_from_tags_domains(tags, [dom] if dom else [])
        now = datetime.now(timezone.utc)
        mc = MemoryCommunity(
            community_id=cid,
            label=label or "concern_area",
            member_memory_ids=[memory_id],
            member_count=1,
            top_tags=tags[:6],
            top_domains=[dom] if dom else [],
            top_roles=[rec.role] if rec else [],
            representative_memory_ids=[memory_id],
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM memory_community_members WHERE community_id = ?", (cid,))
            connection.execute("DELETE FROM memory_communities WHERE community_id = ?", (cid,))
            self._insert_community_row(
                connection,
                mc,
                mc.top_tags,
                mc.top_domains,
                mc.top_roles,
                mc.representative_memory_ids,
            )
            connection.execute(
                "INSERT OR REPLACE INTO memory_community_members (community_id, memory_id) VALUES (?, ?)",
                (cid, memory_id),
            )
            connection.execute(
                "UPDATE memories SET community_id = ? WHERE id = ?",
                (cid, memory_id),
            )
            connection.commit()
        return cid

    def _add_member_to_community(self, community_id: str, memory_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_community_members (community_id, memory_id)
                VALUES (?, ?)
                """,
                (community_id, memory_id),
            )
            connection.execute(
                "UPDATE memories SET community_id = ? WHERE id = ?",
                (community_id, memory_id),
            )
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM memory_community_members WHERE community_id = ?",
                (community_id,),
            ).fetchone()
            count = int(row["c"] if row else 0)
            connection.execute(
                """
                UPDATE memory_communities
                SET member_count = ?, updated_at = ?
                WHERE community_id = ?
                """,
                (count, datetime.now(timezone.utc).isoformat(), community_id),
            )
            connection.commit()

    def update_memory_community_row(self, community: MemoryCommunity) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE memory_communities SET
                    label = ?,
                    member_count = ?,
                    top_tags_json = ?,
                    top_domains_json = ?,
                    top_roles_json = ?,
                    representative_memory_ids_json = ?,
                    activation_score = ?,
                    pressure_score = ?,
                    unresolved_score = ?,
                    contradiction_count = ?,
                    refinement_count = ?,
                    activation_streak = ?,
                    inactive_streak = ?,
                    last_activated_at = ?,
                    last_activated_event_id = ?,
                    last_pressure_update_at = ?,
                    last_resolution_event_id = ?,
                    resolved_recently = ?,
                    activation_reasons_json = ?,
                    pressure_reasons_json = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE community_id = ?
                """,
                (
                    community.label,
                    int(community.member_count),
                    json.dumps(community.top_tags, sort_keys=True),
                    json.dumps(community.top_domains, sort_keys=True),
                    json.dumps(community.top_roles, sort_keys=True),
                    json.dumps(community.representative_memory_ids, sort_keys=True),
                    float(community.activation_score),
                    float(community.pressure_score),
                    float(community.unresolved_score),
                    int(community.contradiction_count),
                    int(community.refinement_count),
                    int(community.activation_streak),
                    int(community.inactive_streak),
                    community.last_activated_at.isoformat() if community.last_activated_at else None,
                    community.last_activated_event_id,
                    community.last_pressure_update_at.isoformat()
                    if community.last_pressure_update_at
                    else None,
                    community.last_resolution_event_id,
                    1 if community.resolved_recently else 0,
                    json.dumps(community.activation_reasons, sort_keys=True),
                    json.dumps(community.pressure_reasons, sort_keys=True),
                    json.dumps(community.metadata, sort_keys=True),
                    now,
                    community.community_id,
                ),
            )
            connection.commit()

    def apply_memory_community_inactivity_decay(
        self,
        *,
        activated_ids: set[str],
        context_ran: bool = True,
    ) -> None:
        """Increment inactive streak / decay scores for communities not activated."""
        if not context_ran:
            return
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM memory_communities").fetchall()
            now = datetime.now(timezone.utc)
            for row in rows:
                cid = str(row["community_id"])
                if cid in activated_ids:
                    continue
                inact = int(row["inactive_streak"] or 0) + 1
                act = float(row["activation_score"] or 0.0) * 0.92
                press = float(row["pressure_score"] or 0.0) * 0.9
                unres = float(row["unresolved_score"] or 0.0) * 0.95
                connection.execute(
                    """
                    UPDATE memory_communities
                    SET inactive_streak = ?,
                        activation_score = ?,
                        pressure_score = ?,
                        unresolved_score = ?,
                        updated_at = ?
                    WHERE community_id = ?
                    """,
                    (inact, act, press, unres, now.isoformat(), cid),
                )
            connection.commit()

    # ---- Internals -----------------------------------------------------

    _SELECT_MEMORY_COLUMNS = (
        "id, created_at, memory_type, content, source_event_id, "
        "salience, confidence, tags_json, metadata_json, role, domain, memory_layer, community_id"
    )

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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_event_id TEXT,
                    salience REAL NOT NULL,
                    confidence REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'unknown',
                    domain TEXT,
                    memory_layer TEXT NOT NULL DEFAULT 'long_term',
                    CHECK (memory_type IN ('working', 'episodic', 'semantic')),
                    CHECK (memory_layer IN ('working', 'long_term')),
                    CHECK (salience >= 0.0 AND salience <= 1.0),
                    CHECK (confidence >= 0.0 AND confidence <= 1.0)
                )
                """
            )
            self._migrate_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories (created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_type_created_at
                ON memories (memory_type, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_role
                ON memories (role)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_domain
                ON memories (domain)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_layer_session_turn
                ON memories (
                    memory_layer,
                    json_extract(metadata_json, '$.session_id'),
                    json_extract(metadata_json, '$.turn_index') DESC,
                    created_at DESC
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, model),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_memory
                ON memory_embeddings (memory_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id TEXT PRIMARY KEY,
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE (source_memory_id, target_memory_id, edge_type),
                    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_edges_source
                ON memory_edges (source_memory_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_edges_target
                ON memory_edges (target_memory_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_communities (
                    community_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL DEFAULT '',
                    member_count INTEGER NOT NULL DEFAULT 0,
                    top_tags_json TEXT NOT NULL DEFAULT '[]',
                    top_domains_json TEXT NOT NULL DEFAULT '[]',
                    top_roles_json TEXT NOT NULL DEFAULT '[]',
                    representative_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    centroid_embedding_id TEXT,
                    centroid_vector_hash TEXT,
                    activation_score REAL NOT NULL DEFAULT 0.0,
                    pressure_score REAL NOT NULL DEFAULT 0.0,
                    unresolved_score REAL NOT NULL DEFAULT 0.0,
                    contradiction_count INTEGER NOT NULL DEFAULT 0,
                    refinement_count INTEGER NOT NULL DEFAULT 0,
                    activation_streak INTEGER NOT NULL DEFAULT 0,
                    inactive_streak INTEGER NOT NULL DEFAULT 0,
                    last_activated_at TEXT,
                    last_activated_event_id TEXT,
                    last_pressure_update_at TEXT,
                    last_resolution_event_id TEXT,
                    resolved_recently INTEGER NOT NULL DEFAULT 0,
                    activation_reasons_json TEXT NOT NULL DEFAULT '[]',
                    pressure_reasons_json TEXT NOT NULL DEFAULT '[]',
                    community_detection_strategy TEXT NOT NULL DEFAULT 'deterministic_connected_components_v0',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_community_members (
                    community_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (community_id, memory_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_community_members_memory
                ON memory_community_members (memory_id)
                """
            )
            connection.commit()

    def _migrate_columns(self, connection: sqlite3.Connection) -> None:
        """Add backward-compatible columns when older databases are opened."""
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "role" not in existing_columns:
            connection.execute(
                "ALTER TABLE memories ADD COLUMN role TEXT NOT NULL DEFAULT 'unknown'"
            )
        if "domain" not in existing_columns:
            connection.execute("ALTER TABLE memories ADD COLUMN domain TEXT")
        if "memory_layer" not in existing_columns:
            connection.execute(
                "ALTER TABLE memories ADD COLUMN memory_layer TEXT NOT NULL DEFAULT 'long_term'"
            )
        if "community_id" not in existing_columns:
            connection.execute("ALTER TABLE memories ADD COLUMN community_id TEXT")

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(int(limit), 1)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        keys = row.keys()
        role = row["role"] if "role" in keys and row["role"] is not None else "unknown"
        domain = row["domain"] if "domain" in keys else None
        return MemoryRecord.from_dict(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "source_event_id": row["source_event_id"],
                "salience": row["salience"],
                "confidence": row["confidence"],
                "tags": json.loads(row["tags_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "role": role,
                "domain": domain,
                "memory_layer": (
                    row["memory_layer"]
                    if "memory_layer" in keys and row["memory_layer"] is not None
                    else MemoryLayer.LONG_TERM.value
                ),
                "community_id": (
                    row["community_id"]
                    if "community_id" in keys and row["community_id"]
                    else None
                ),
            }
        )
