"""Static context assembly for Fullerene Context v0."""

from __future__ import annotations

from typing import Sequence

from fullerene.context.models import (
    STATIC_RECENT_EPISODIC_V0,
    ContextItem,
    ContextItemType,
    ContextWindow,
)
from fullerene.memory import MemoryRecord, MemoryStore, MemoryType


class StaticContextAssembler:
    """Build a small, deterministic context window from recent episodic memory."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        max_items: int = 5,
    ) -> None:
        self.store = store
        self.max_items = max(int(max_items), 1)

    def assemble(
        self,
        recent_records: Sequence[MemoryRecord] | None = None,
    ) -> ContextWindow:
        records = self._load_recent_records(recent_records)
        items = [self._memory_to_context_item(record) for record in records]
        source_types = sorted({"episodic_memory"} if items else set())

        return ContextWindow(
            items=items,
            max_items=self.max_items,
            strategy=STATIC_RECENT_EPISODIC_V0,
            metadata={
                "source_types": source_types,
                "source_memory_type": MemoryType.EPISODIC.value,
                "item_count": len(items),
            },
        )

    def _load_recent_records(
        self,
        recent_records: Sequence[MemoryRecord] | None,
    ) -> list[MemoryRecord]:
        if recent_records is not None:
            episodic_records = [
                record
                for record in recent_records
                if record.memory_type == MemoryType.EPISODIC
            ]
            return list(episodic_records[: self.max_items])
        if self.store is None:
            return []
        return self.store.list_recent(
            limit=self.max_items,
            memory_type=MemoryType.EPISODIC,
        )

    @staticmethod
    def _memory_to_context_item(record: MemoryRecord) -> ContextItem:
        return ContextItem(
            id=record.id,
            item_type=ContextItemType.MEMORY,
            content=record.content,
            source_id=record.source_event_id,
            created_at=record.created_at,
            metadata={
                "memory_type": record.memory_type.value,
                "salience": record.salience,
                "confidence": record.confidence,
                "tags": list(record.tags),
                "memory_metadata": dict(record.metadata),
            },
        )
