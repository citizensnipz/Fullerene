"""SQLite-backed belief storage for Fullerene World Model v0."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from fullerene.world_model.models import Belief, BeliefStatus, utcnow


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
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        return self.list_beliefs(limit=limit, status=BeliefStatus.ACTIVE)

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
                    payload["id"],
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Belief {belief.id!r} does not exist")
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
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
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    CHECK (
                        status IN ('active', 'stale', 'contradicted', 'retired')
                    ),
                    CHECK (source IN ('user', 'system', 'memory', 'goal'))
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beliefs_status_confidence_updated
                ON beliefs (status, confidence DESC, updated_at DESC)
                """
            )
            connection.commit()

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
            }
        )
