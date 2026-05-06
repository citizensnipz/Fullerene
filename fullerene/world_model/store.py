"""SQLite-backed belief storage for Fullerene World Model v1."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from fullerene.world_model.models import Belief, BeliefStatus, normalize_statement, utcnow


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

    def add_belief_edge(
        self,
        *,
        source_belief_id: str,
        target_belief_id: str,
        edge_type: str,
        weight: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        src, tgt = sorted((source_belief_id, target_belief_id))
        if src == tgt:
            return
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO belief_edges (
                    source_belief_id, target_belief_id, edge_type, weight, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    src,
                    tgt,
                    str(edge_type).strip().lower() or "related",
                    max(0.0, min(float(weight), 1.0)),
                    utcnow().isoformat(),
                    json.dumps(dict(metadata or {}), sort_keys=True),
                ),
            )
            connection.commit()

    def list_belief_edges(self, belief_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT source_belief_id, target_belief_id, edge_type, weight, updated_at, metadata_json
                FROM belief_edges
                WHERE source_belief_id = ? OR target_belief_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (belief_id, belief_id, self._normalize_limit(limit)),
            ).fetchall()
        return [
            {
                "source_belief_id": row["source_belief_id"],
                "target_belief_id": row["target_belief_id"],
                "edge_type": row["edge_type"],
                "weight": row["weight"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

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
