"""SQLite-backed goal storage for Fullerene Goals v0."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from fullerene.goals.models import Goal, GoalStatus, utcnow


class GoalStore(Protocol):
    def add_goal(self, goal: Goal) -> None:
        """Persist a goal."""

    def get_goal(self, goal_id: str) -> Goal | None:
        """Fetch a goal by id."""

    def list_goals(
        self,
        limit: int,
        status: GoalStatus | None = None,
    ) -> list[Goal]:
        """Return goals, optionally filtered by status."""

    def list_active_goals(self, limit: int) -> list[Goal]:
        """Return active goals."""

    def update_goal(self, goal: Goal) -> None:
        """Persist edits to an existing goal."""


class SQLiteGoalStore:
    """SQLite-backed source of truth for Fullerene goals."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add_goal(self, goal: Goal) -> None:
        payload = goal.to_dict()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO goals (
                    id,
                    description,
                    priority,
                    status,
                    tags_json,
                    created_at,
                    updated_at,
                    source,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["description"],
                    payload["priority"],
                    payload["status"],
                    json.dumps(payload["tags"], sort_keys=True),
                    payload["created_at"],
                    payload["updated_at"],
                    payload["source"],
                    json.dumps(payload["metadata"], sort_keys=True),
                ),
            )
            connection.commit()

    def get_goal(self, goal_id: str) -> Goal | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    description,
                    priority,
                    status,
                    tags_json,
                    created_at,
                    updated_at,
                    source,
                    metadata_json
                FROM goals
                WHERE id = ?
                """,
                (goal_id,),
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def list_goals(
        self,
        limit: int,
        status: GoalStatus | None = None,
    ) -> list[Goal]:
        bounded_limit = self._normalize_limit(limit)
        query = """
            SELECT
                id,
                description,
                priority,
                status,
                tags_json,
                created_at,
                updated_at,
                source,
                metadata_json
            FROM goals
        """
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY priority DESC, updated_at DESC, id DESC LIMIT ?"
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def list_active_goals(self, limit: int) -> list[Goal]:
        return self.list_goals(limit=limit, status=GoalStatus.ACTIVE)

    def update_goal(self, goal: Goal) -> None:
        goal.updated_at = utcnow()
        payload = goal.to_dict()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE goals
                SET
                    description = ?,
                    priority = ?,
                    status = ?,
                    tags_json = ?,
                    updated_at = ?,
                    source = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    payload["description"],
                    payload["priority"],
                    payload["status"],
                    json.dumps(payload["tags"], sort_keys=True),
                    payload["updated_at"],
                    payload["source"],
                    json.dumps(payload["metadata"], sort_keys=True),
                    payload["id"],
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Goal {goal.id!r} does not exist")
            connection.commit()

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
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority REAL NOT NULL,
                    status TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    CHECK (priority >= 0.0 AND priority <= 1.0),
                    CHECK (status IN ('active', 'paused', 'completed')),
                    CHECK (source IN ('user', 'system'))
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goals_status_priority_updated
                ON goals (status, priority DESC, updated_at DESC)
                """
            )
            connection.commit()

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(int(limit), 1)

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        return Goal.from_dict(
            {
                "id": row["id"],
                "description": row["description"],
                "priority": row["priority"],
                "status": row["status"],
                "tags": json.loads(row["tags_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "source": row["source"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
