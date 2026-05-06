"""SQLite-backed goal storage for Fullerene Goals."""

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

    def list_active_goals(
        self,
        limit: int,
        *,
        include_inactive: bool = False,
    ) -> list[Goal]:
        """Return active goals, optionally including inactive goals."""

    def update_goal(self, goal: Goal) -> None:
        """Persist edits to an existing goal."""

    def update_goal_status(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        reason: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, object]:
        """Apply a goal lifecycle transition and return transition metadata."""


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
                    reinforcement_score,
                    activation_count,
                    last_activated_at,
                    last_activated_event_id,
                    last_reinforced_at,
                    completion_score,
                    paused_reason,
                    completed_reason,
                    completed_at,
                    evidence_event_ids_json,
                    blocked_reason,
                    stale_score,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    payload["reinforcement_score"],
                    payload["activation_count"],
                    payload["last_activated_at"],
                    payload["last_activated_event_id"],
                    payload["last_reinforced_at"],
                    payload["completion_score"],
                    payload["paused_reason"],
                    payload["completed_reason"],
                    payload["completed_at"],
                    json.dumps(payload["evidence_event_ids"], sort_keys=True),
                    payload["blocked_reason"],
                    payload["stale_score"],
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
                    reinforcement_score,
                    activation_count,
                    last_activated_at,
                    last_activated_event_id,
                    last_reinforced_at,
                    completion_score,
                    paused_reason,
                    completed_reason,
                    completed_at,
                    evidence_event_ids_json,
                    blocked_reason,
                    stale_score,
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
                reinforcement_score,
                activation_count,
                last_activated_at,
                last_activated_event_id,
                last_reinforced_at,
                completion_score,
                paused_reason,
                completed_reason,
                completed_at,
                evidence_event_ids_json,
                blocked_reason,
                stale_score,
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

    def list_active_goals(
        self,
        limit: int,
        *,
        include_inactive: bool = False,
    ) -> list[Goal]:
        if include_inactive:
            return self.list_goals(limit=limit, status=None)
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
                    reinforcement_score = ?,
                    activation_count = ?,
                    last_activated_at = ?,
                    last_activated_event_id = ?,
                    last_reinforced_at = ?,
                    completion_score = ?,
                    paused_reason = ?,
                    completed_reason = ?,
                    completed_at = ?,
                    evidence_event_ids_json = ?,
                    blocked_reason = ?,
                    stale_score = ?,
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
                    payload["reinforcement_score"],
                    payload["activation_count"],
                    payload["last_activated_at"],
                    payload["last_activated_event_id"],
                    payload["last_reinforced_at"],
                    payload["completion_score"],
                    payload["paused_reason"],
                    payload["completed_reason"],
                    payload["completed_at"],
                    json.dumps(payload["evidence_event_ids"], sort_keys=True),
                    payload["blocked_reason"],
                    payload["stale_score"],
                    json.dumps(payload["metadata"], sort_keys=True),
                    payload["id"],
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Goal {goal.id!r} does not exist")
            connection.commit()

    def update_goal_status(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        reason: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, object]:
        goal = self.get_goal(goal_id)
        if goal is None:
            return {"ok": False, "reason": "goal_not_found", "goal_id": goal_id}
        previous_status = goal.status
        if previous_status == status:
            return {
                "ok": True,
                "goal_id": goal_id,
                "previous_status": previous_status.value,
                "new_status": status.value,
                "reason": "no_status_change",
                "event_id": event_id,
                "timestamp": utcnow().isoformat(),
            }
        allowed = {
            GoalStatus.ACTIVE: {GoalStatus.PAUSED, GoalStatus.COMPLETED},
            GoalStatus.PAUSED: {GoalStatus.ACTIVE, GoalStatus.COMPLETED},
            GoalStatus.COMPLETED: set(),
        }
        if status not in allowed.get(previous_status, set()):
            return {
                "ok": False,
                "reason": "invalid_status_transition",
                "goal_id": goal_id,
                "previous_status": previous_status.value,
                "new_status": status.value,
                "event_id": event_id,
                "timestamp": utcnow().isoformat(),
            }
        goal.status = status
        if status == GoalStatus.PAUSED:
            goal.paused_reason = reason
        elif status == GoalStatus.ACTIVE:
            goal.paused_reason = None
        elif status == GoalStatus.COMPLETED:
            goal.completed_at = utcnow()
            goal.completed_reason = reason
            goal.completion_score = 1.0
        if event_id:
            evidence = list(goal.evidence_event_ids)
            evidence.append(event_id)
            goal.evidence_event_ids = evidence[-20:]
        self.update_goal(goal)
        return {
            "ok": True,
            "goal_id": goal_id,
            "previous_status": previous_status.value,
            "new_status": status.value,
            "reason": reason,
            "event_id": event_id,
            "timestamp": utcnow().isoformat(),
        }

    def pause_goal(self, goal_id: str, reason: str | None = None) -> dict[str, object]:
        return self.update_goal_status(goal_id, GoalStatus.PAUSED, reason=reason)

    def resume_goal(self, goal_id: str, reason: str | None = None) -> dict[str, object]:
        return self.update_goal_status(goal_id, GoalStatus.ACTIVE, reason=reason)

    def complete_goal(
        self,
        goal_id: str,
        *,
        reason: str | None = None,
        evidence_event_id: str | None = None,
    ) -> dict[str, object]:
        return self.update_goal_status(
            goal_id,
            GoalStatus.COMPLETED,
            reason=reason,
            event_id=evidence_event_id,
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
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority REAL NOT NULL,
                    status TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reinforcement_score REAL NOT NULL DEFAULT 0.0,
                    activation_count INTEGER NOT NULL DEFAULT 0,
                    last_activated_at TEXT,
                    last_activated_event_id TEXT,
                    last_reinforced_at TEXT,
                    completion_score REAL NOT NULL DEFAULT 0.0,
                    paused_reason TEXT,
                    completed_reason TEXT,
                    completed_at TEXT,
                    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    blocked_reason TEXT,
                    stale_score REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT NOT NULL,
                    CHECK (priority >= 0.0 AND priority <= 1.0),
                    CHECK (reinforcement_score >= 0.0 AND reinforcement_score <= 1.0),
                    CHECK (completion_score >= 0.0 AND completion_score <= 1.0),
                    CHECK (stale_score >= 0.0 AND stale_score <= 1.0),
                    CHECK (status IN ('active', 'paused', 'completed')),
                    CHECK (source IN ('user', 'system'))
                )
                """
            )
            self._ensure_column(
                connection, "goals", "reinforcement_score", "REAL NOT NULL DEFAULT 0.0"
            )
            self._ensure_column(
                connection, "goals", "activation_count", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "goals", "last_activated_at", "TEXT")
            self._ensure_column(connection, "goals", "last_activated_event_id", "TEXT")
            self._ensure_column(connection, "goals", "last_reinforced_at", "TEXT")
            self._ensure_column(
                connection, "goals", "completion_score", "REAL NOT NULL DEFAULT 0.0"
            )
            self._ensure_column(connection, "goals", "paused_reason", "TEXT")
            self._ensure_column(connection, "goals", "completed_reason", "TEXT")
            self._ensure_column(connection, "goals", "completed_at", "TEXT")
            self._ensure_column(
                connection,
                "goals",
                "evidence_event_ids_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            self._ensure_column(connection, "goals", "blocked_reason", "TEXT")
            self._ensure_column(
                connection, "goals", "stale_score", "REAL NOT NULL DEFAULT 0.0"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_goals_status_priority_updated
                ON goals (status, priority DESC, updated_at DESC)
                """
            )
            connection.commit()

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {row["name"] for row in rows}
        if column_name in existing_columns:
            return
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
        )

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
                "reinforcement_score": row["reinforcement_score"],
                "activation_count": row["activation_count"],
                "last_activated_at": row["last_activated_at"],
                "last_activated_event_id": row["last_activated_event_id"],
                "last_reinforced_at": row["last_reinforced_at"],
                "completion_score": row["completion_score"],
                "paused_reason": row["paused_reason"],
                "completed_reason": row["completed_reason"],
                "completed_at": row["completed_at"],
                "evidence_event_ids": json.loads(row["evidence_event_ids_json"] or "[]"),
                "blocked_reason": row["blocked_reason"],
                "stale_score": row["stale_score"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
