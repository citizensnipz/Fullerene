"""SQLite-backed memory store for Fullerene memory records."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from fullerene.memory.models import MemoryRecord, MemoryType
from fullerene.memory.scoring import score_sort_key, tokenize
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
    """SQLite-backed source of truth for Fullerene memory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

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
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            connection.commit()

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    memory_type,
                    content,
                    source_event_id,
                    salience,
                    confidence,
                    tags_json,
                    metadata_json
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_recent(
        self,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        bounded_limit = self._normalize_limit(limit)
        query = """
            SELECT
                id,
                created_at,
                memory_type,
                content,
                source_event_id,
                salience,
                confidence,
                tags_json,
                metadata_json
            FROM memories
        """
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
                SELECT
                    id,
                    created_at,
                    memory_type,
                    content,
                    source_event_id,
                    salience,
                    confidence,
                    tags_json,
                    metadata_json
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
                    CHECK (memory_type IN ('working', 'episodic', 'semantic')),
                    CHECK (salience >= 0.0 AND salience <= 1.0),
                    CHECK (confidence >= 0.0 AND confidence <= 1.0)
                )
                """
            )
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
            connection.commit()

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(int(limit), 1)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
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
            }
        )
