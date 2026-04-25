"""Deterministic memory facet for Fullerene Memory v1."""

from __future__ import annotations

from pathlib import Path

from fullerene.memory import (
    MemoryRecord,
    MemoryStore,
    MemoryType,
    SQLiteMemoryStore,
    compute_salience,
    explain_salience,
    infer_tags,
    merge_tags,
)
from fullerene.memory.models import normalize_tags
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusState,
)


class MemoryFacet:
    """Persists episodic memory and retrieves a bounded memory view.

    Memory v1: when storing an event, the facet infers deterministic tags from
    the content, merges them with any explicit metadata-supplied tags, and
    computes a transparent salience score from a small, inspectable rule set.
    Explicit tags from event metadata are never overwritten - they are merged.
    """

    name = "memory"

    def __init__(
        self,
        store: MemoryStore,
        *,
        retrieve_limit: int = 3,
        working_limit: int = 3,
    ) -> None:
        self.store = store
        self.retrieve_limit = max(int(retrieve_limit), 1)
        self.working_limit = max(int(working_limit), 1)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        retrieve_limit: int = 3,
        working_limit: int = 3,
    ) -> "MemoryFacet":
        return cls(
            SQLiteMemoryStore(path),
            retrieve_limit=retrieve_limit,
            working_limit=working_limit,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state

        stored_memory = None
        if self._should_store_event(event):
            stored_memory = self._build_memory_record(event)
            self.store.add_memory(stored_memory)

        working_memories = self.store.list_recent(limit=self.working_limit)
        relevant_limit = self.retrieve_limit + (1 if stored_memory is not None else 0)
        relevant_memories = [
            memory
            for memory in self.store.retrieve_relevant(event, limit=relevant_limit)
            if stored_memory is None or memory.id != stored_memory.id
        ][: self.retrieve_limit]

        stored_summary = (
            f"stored episodic memory {stored_memory.id}"
            if stored_memory is not None
            else "stored nothing"
        )
        summary = (
            f"Memory facet {stored_summary}; "
            f"retrieved {len(relevant_memories)} relevant memories and "
            f"{len(working_memories)} working memories."
        )

        return FacetResult(
            facet_name=self.name,
            summary=summary,
            proposed_decision=(
                DecisionAction.RECORD if stored_memory is not None else None
            ),
            state_updates={
                "last_stored_memory_id": stored_memory.id if stored_memory else None,
                "last_working_memory_ids": [memory.id for memory in working_memories],
                "last_relevant_memory_ids": [memory.id for memory in relevant_memories],
            },
            metadata={
                "stored_memory": self._describe_memory(stored_memory)
                if stored_memory is not None
                else None,
                "working_memories": [
                    self._describe_memory(memory) for memory in working_memories
                ],
                "relevant_memories": [
                    self._describe_memory(memory) for memory in relevant_memories
                ],
            },
        )

    def _should_store_event(self, event: Event) -> bool:
        if event.event_type == EventType.USER_MESSAGE:
            return bool(event.content.strip())
        if event.event_type == EventType.SYSTEM_NOTE:
            return bool(event.content.strip() or event.metadata)
        return False

    def _build_memory_record(self, event: Event) -> MemoryRecord:
        metadata_tags = normalize_tags(event.metadata.get("tags", []))
        inferred_tags = infer_tags(event.content)
        # Explicit metadata tags lead so they retain priority in the merged
        # output; inferred tags only fill gaps.
        merged_tags = merge_tags(metadata_tags, inferred_tags)

        is_user_message = event.event_type == EventType.USER_MESSAGE
        salience = compute_salience(
            content=event.content,
            tags=merged_tags,
            is_user_message=is_user_message,
        )
        salience_breakdown = explain_salience(
            content=event.content,
            tags=merged_tags,
            is_user_message=is_user_message,
        )

        return MemoryRecord(
            memory_type=MemoryType.EPISODIC,
            content=event.content,
            source_event_id=event.event_id,
            salience=salience,
            confidence=1.0,
            tags=merged_tags,
            metadata={
                "event_type": event.event_type.value,
                "event_timestamp": event.timestamp.isoformat(),
                "event_metadata": event.metadata,
                "metadata_tags": metadata_tags,
                "inferred_tags": inferred_tags,
                "salience_breakdown": salience_breakdown,
            },
        )

    @staticmethod
    def _describe_memory(memory: MemoryRecord) -> dict[str, object]:
        return {
            "id": memory.id,
            "created_at": memory.created_at.isoformat(),
            "memory_type": memory.memory_type.value,
            "source_event_id": memory.source_event_id,
            "salience": memory.salience,
            "confidence": memory.confidence,
            "tags": list(memory.tags),
            "content_preview": memory.content[:120],
        }
