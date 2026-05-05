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
import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from fullerene.memory.edges import MemoryEdge, MemoryEdgeType
from fullerene.memory.embeddings import deserialize_vector, serialize_vector
from fullerene.memory.hybrid import explain_hybrid_score, hybrid_sort_key
from fullerene.memory.models import MemoryRecord, MemoryType
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
                    domain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        if memory_type is not None:
            query += " WHERE memory_type = ?"
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
                WHERE {clauses}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
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
                WHERE salience >= ?
                ORDER BY salience DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (threshold, bounded_limit),
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
                WHERE domain = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (cleaned_domain, bounded_limit),
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

    # ---- Internals -----------------------------------------------------

    _SELECT_MEMORY_COLUMNS = (
        "id, created_at, memory_type, content, source_event_id, "
        "salience, confidence, tags_json, metadata_json, role, domain"
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
                    CHECK (memory_type IN ('working', 'episodic', 'semantic')),
                    CHECK (salience >= 0.0 AND salience <= 1.0),
                    CHECK (confidence >= 0.0 AND confidence <= 1.0)
                )
                """
            )
            self._migrate_role_and_domain_columns(connection)
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
            connection.commit()

    def _migrate_role_and_domain_columns(self, connection: sqlite3.Connection) -> None:
        """Add v2 columns when an older v1 database is opened."""
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
            }
        )
