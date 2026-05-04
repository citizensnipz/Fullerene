"""SQLite-backed policy storage for Fullerene Policy v0."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from fullerene.policy.models import PolicyRule, utcnow


class PolicyStore(Protocol):
    def add_policy(self, rule: PolicyRule) -> None:
        """Persist a policy rule."""

    def get_policy(self, policy_id: str) -> PolicyRule | None:
        """Fetch a policy rule by id."""

    def list_policies(
        self,
        limit: int,
        enabled_only: bool = False,
    ) -> list[PolicyRule]:
        """Return policy rules, optionally filtered to enabled rows only."""

    def update_policy(self, rule: PolicyRule) -> None:
        """Persist edits to an existing policy rule."""

    def delete_policy(self, policy_id: str) -> None:
        """Delete a persisted policy rule."""

    def list_enabled_policies(
        self,
        limit: int | None = None,
    ) -> list[PolicyRule]:
        """Return enabled policy rules, optionally bounded."""

    def count_enabled_policies(self) -> int:
        """Return the total number of enabled policy rules."""


class SQLitePolicyStore:
    """SQLite-backed source of truth for explicit Fullerene policy rules."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add_policy(self, rule: PolicyRule) -> None:
        payload = rule.to_dict()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO policies (
                    id,
                    name,
                    description,
                    rule_type,
                    target_type,
                    target,
                    conditions_json,
                    priority,
                    enabled,
                    source,
                    created_at,
                    updated_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["name"],
                    payload["description"],
                    payload["rule_type"],
                    payload["target_type"],
                    payload["target"],
                    json.dumps(payload["conditions"], sort_keys=True),
                    payload["priority"],
                    1 if payload["enabled"] else 0,
                    payload["source"],
                    payload["created_at"],
                    payload["updated_at"],
                    json.dumps(payload["metadata"], sort_keys=True),
                ),
            )
            connection.commit()

    def get_policy(self, policy_id: str) -> PolicyRule | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    name,
                    description,
                    rule_type,
                    target_type,
                    target,
                    conditions_json,
                    priority,
                    enabled,
                    source,
                    created_at,
                    updated_at,
                    metadata_json
                FROM policies
                WHERE id = ?
                """,
                (policy_id,),
            ).fetchone()
        return self._row_to_policy(row) if row else None

    def list_policies(
        self,
        limit: int,
        enabled_only: bool = False,
    ) -> list[PolicyRule]:
        bounded_limit = self._normalize_limit(limit)
        query = """
            SELECT
                id,
                name,
                description,
                rule_type,
                target_type,
                target,
                conditions_json,
                priority,
                enabled,
                source,
                created_at,
                updated_at,
                metadata_json
            FROM policies
        """
        params: list[object] = []
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY priority DESC, updated_at DESC, id DESC LIMIT ?"
        params.append(bounded_limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_policy(row) for row in rows]

    def update_policy(self, rule: PolicyRule) -> None:
        rule.updated_at = utcnow()
        payload = rule.to_dict()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE policies
                SET
                    name = ?,
                    description = ?,
                    rule_type = ?,
                    target_type = ?,
                    target = ?,
                    conditions_json = ?,
                    priority = ?,
                    enabled = ?,
                    source = ?,
                    updated_at = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    payload["name"],
                    payload["description"],
                    payload["rule_type"],
                    payload["target_type"],
                    payload["target"],
                    json.dumps(payload["conditions"], sort_keys=True),
                    payload["priority"],
                    1 if payload["enabled"] else 0,
                    payload["source"],
                    payload["updated_at"],
                    json.dumps(payload["metadata"], sort_keys=True),
                    payload["id"],
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Policy {rule.id!r} does not exist")
            connection.commit()

    def delete_policy(self, policy_id: str) -> None:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM policies WHERE id = ?",
                (policy_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Policy {policy_id!r} does not exist")
            connection.commit()

    def list_enabled_policies(
        self,
        limit: int | None = None,
    ) -> list[PolicyRule]:
        query = """
            SELECT
                id,
                name,
                description,
                rule_type,
                target_type,
                target,
                conditions_json,
                priority,
                enabled,
                source,
                created_at,
                updated_at,
                metadata_json
            FROM policies
            WHERE enabled = 1
            ORDER BY priority DESC, updated_at DESC, id DESC
        """
        params: list[object] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(self._normalize_limit(limit))
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_policy(row) for row in rows]

    def count_enabled_policies(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS enabled_count FROM policies WHERE enabled = 1"
            ).fetchone()
        if row is None:
            return 0
        return int(row["enabled_count"])

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
                CREATE TABLE IF NOT EXISTS policies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    priority REAL NOT NULL,
                    enabled INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    CHECK (
                        rule_type IN (
                            'allow',
                            'deny',
                            'require_approval',
                            'prefer'
                        )
                    ),
                    CHECK (
                        target_type IN (
                            'internal_state',
                            'file_write',
                            'file_delete',
                            'shell',
                            'network',
                            'message',
                            'git',
                            'tool',
                            'decision',
                            'tag',
                            'general'
                        )
                    ),
                    CHECK (enabled IN (0, 1)),
                    CHECK (source IN ('user', 'system'))
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_policies_enabled_priority_updated
                ON policies (enabled, priority DESC, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_policies_rule_target_enabled
                ON policies (rule_type, target_type, enabled)
                """
            )
            connection.commit()

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(int(limit), 1)

    @staticmethod
    def _row_to_policy(row: sqlite3.Row) -> PolicyRule:
        return PolicyRule.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "rule_type": row["rule_type"],
                "target_type": row["target_type"],
                "target": row["target"],
                "conditions": json.loads(row["conditions_json"]),
                "priority": row["priority"],
                "enabled": bool(row["enabled"]),
                "source": row["source"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
