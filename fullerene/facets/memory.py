"""Deterministic memory facet for Fullerene Memory v1 / v2."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from fullerene.memory import (
    EmbeddingProvider,
    MemoryEdge,
    MemoryEdgeType,
    MemoryRecord,
    MemoryRole,
    MemoryStore,
    MemoryType,
    SQLiteMemoryStore,
    classify_memory_role,
    classify_query_intent,
    compute_salience,
    explain_hybrid_score,
    explain_salience,
    infer_domain,
    infer_tags,
    merge_tags,
    safe_embed,
    tokenize,
)
from fullerene.memory.models import normalize_tags
from fullerene.memory.scoring import extract_event_tags
from fullerene.nexus.models import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusState,
)


# Bounds on the write-time edge candidate sets. Memory v2 stays inspectable
# by capping each edge source so a single new memory does not kick off a
# graph-wide scan.
EDGE_RECENT_LIMIT = 20
EDGE_HIGH_SALIENCE_LIMIT = 20
EDGE_DOMAIN_LIMIT = 20

# Edge thresholds. Kept conservative on purpose: edges should be
# inspectable signals, not noise.
TAG_OVERLAP_EDGE_THRESHOLD = 1
KEYWORD_OVERLAP_EDGE_THRESHOLD = 0.3
SEMANTIC_EDGE_THRESHOLD = 0.55
TEMPORAL_PROXIMITY_WINDOW = timedelta(minutes=30)


class MemoryFacet:
    """Persists episodic memory and retrieves a bounded memory view.

    Memory v1: when storing an event, the facet infers deterministic tags
    from the content, merges them with any explicit metadata-supplied tags,
    and computes a transparent salience score from a small, inspectable
    rule set.

    Memory v2: the same store is now richer. Each new record carries a
    deterministic ``role`` and ``domain``; an optional embedding vector can
    be persisted as an *index* into the canonical SQLite row when an
    embedding provider is configured; bounded write-time edges are emitted
    against recent / high-salience / same-domain candidates; and retrieval
    uses a hybrid score that prefers preference / fact / task memories over
    repeated questions for the matching query intent. SQLite remains the
    source of truth, and missing embeddings always fall back to
    deterministic v1 retrieval through the same facet code path.
    """

    name = "memory"

    def __init__(
        self,
        store: MemoryStore,
        *,
        retrieve_limit: int = 3,
        working_limit: int = 3,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.retrieve_limit = max(int(retrieve_limit), 1)
        self.working_limit = max(int(working_limit), 1)
        self.embedding_provider = embedding_provider

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        retrieve_limit: int = 3,
        working_limit: int = 3,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "MemoryFacet":
        return cls(
            SQLiteMemoryStore(path),
            retrieve_limit=retrieve_limit,
            working_limit=working_limit,
            embedding_provider=embedding_provider,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state

        stored_memory: MemoryRecord | None = None
        stored_embedding_status: str | None = None
        stored_edges: list[MemoryEdge] = []
        if self._should_store_event(event):
            stored_memory = self._build_memory_record(event)
            self.store.add_memory(stored_memory)
            stored_embedding_status = self._maybe_store_embedding(stored_memory)
            stored_edges = self._maybe_store_edges(stored_memory)

        retrieval = self._retrieve_relevant(event, exclude_id=stored_memory.id if stored_memory else None)
        relevant_memories = retrieval["records"]
        retrieval_strategy = retrieval["strategy"]
        relevant_breakdowns = retrieval["breakdowns"]

        working_memories = self.store.list_recent(limit=self.working_limit)

        stored_summary = (
            f"stored episodic memory {stored_memory.id}"
            if stored_memory is not None
            else "stored nothing"
        )
        summary = (
            f"Memory facet {stored_summary}; "
            f"retrieved {len(relevant_memories)} relevant memories ({retrieval_strategy}) and "
            f"{len(working_memories)} working memories."
        )

        included_roles = sorted({memory.role or MemoryRole.UNKNOWN.value for memory in relevant_memories})
        included_domains = sorted({memory.domain for memory in relevant_memories if memory.domain})

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
                "last_retrieval_strategy": retrieval_strategy,
                "last_included_memory_roles": included_roles,
                "last_included_memory_domains": included_domains,
                "last_query_intent": retrieval["query_intent"],
                "last_event_domain": retrieval["event_domain"],
                "last_stored_memory_role": stored_memory.role if stored_memory else None,
                "last_stored_memory_domain": stored_memory.domain if stored_memory else None,
                "last_stored_embedding_status": stored_embedding_status,
                "last_stored_edge_count": len(stored_edges),
            },
            metadata={
                "stored_memory": self._describe_memory(stored_memory)
                if stored_memory is not None
                else None,
                "working_memories": [
                    self._describe_memory(memory) for memory in working_memories
                ],
                "relevant_memories": [
                    self._describe_memory(memory, breakdown=breakdown)
                    for memory, breakdown in zip(relevant_memories, relevant_breakdowns)
                ],
                "retrieval_strategy": retrieval_strategy,
                "query_intent": retrieval["query_intent"],
                "event_domain": retrieval["event_domain"],
                "included_memory_roles": included_roles,
                "included_memory_domains": included_domains,
                "stored_embedding_status": stored_embedding_status,
                "stored_edges": [edge.to_dict() for edge in stored_edges],
            },
        )

    # ---- Storage ------------------------------------------------------

    def _should_store_event(self, event: Event) -> bool:
        if event.event_type == EventType.USER_MESSAGE:
            return bool(event.content.strip())
        if event.event_type == EventType.SYSTEM_NOTE:
            return bool(event.content.strip() or event.metadata)
        return False

    def _build_memory_record(self, event: Event) -> MemoryRecord:
        metadata_tags, inferred_tags, merged_tags = self._derive_tags(event)
        salience, salience_breakdown = self._derive_salience(event, merged_tags)
        role = self._derive_role(event)
        domain = infer_domain(event.content, merged_tags)

        return MemoryRecord(
            memory_type=MemoryType.EPISODIC,
            content=event.content,
            source_event_id=event.event_id,
            salience=salience,
            confidence=1.0,
            tags=merged_tags,
            role=role.value,
            domain=domain,
            metadata={
                "event_type": event.event_type.value,
                "event_timestamp": event.timestamp.isoformat(),
                "event_metadata": event.metadata,
                "metadata_tags": metadata_tags,
                "inferred_tags": inferred_tags,
                "salience_breakdown": salience_breakdown,
                "memory_role": role.value,
                "memory_domain": domain,
            },
        )

    @staticmethod
    def _derive_tags(event: Event) -> tuple[list[str], list[str], list[str]]:
        metadata_tags = normalize_tags(event.metadata.get("tags", []))
        inferred_tags = infer_tags(event.content)
        merged_tags = merge_tags(metadata_tags, inferred_tags)
        return metadata_tags, inferred_tags, merged_tags

    @staticmethod
    def _derive_salience(
        event: Event,
        tags: list[str],
    ) -> tuple[float, dict[str, float]]:
        is_user_message = event.event_type == EventType.USER_MESSAGE
        salience = compute_salience(
            content=event.content,
            tags=tags,
            is_user_message=is_user_message,
        )
        salience_breakdown = explain_salience(
            content=event.content,
            tags=tags,
            is_user_message=is_user_message,
        )
        return salience, salience_breakdown

    @staticmethod
    def _derive_role(event: Event) -> MemoryRole:
        explicit = event.metadata.get("memory_role") if isinstance(event.metadata, dict) else None
        if explicit is not None:
            from fullerene.memory.roles import coerce_memory_role

            coerced = coerce_memory_role(explicit)
            if coerced != MemoryRole.UNKNOWN:
                return coerced
        return classify_memory_role(event.content)

    def _maybe_store_embedding(self, memory: MemoryRecord) -> str | None:
        provider = self.embedding_provider
        if provider is None:
            return None
        store = self.store
        if not hasattr(store, "add_memory_embedding"):
            return "store_unsupported"
        vector, error = safe_embed(provider, memory.content)
        if vector is None:
            return error or "no_vector"
        store.add_memory_embedding(
            memory_id=memory.id,
            model=provider.name,
            vector=vector,
        )
        return "stored"

    # ---- Edges --------------------------------------------------------

    def _maybe_store_edges(self, memory: MemoryRecord) -> list[MemoryEdge]:
        store = self.store
        if not hasattr(store, "add_memory_edge"):
            return []
        candidates = self._gather_edge_candidates(memory)
        if not candidates:
            return []

        emitted: list[MemoryEdge] = []
        seen_pairs: set[tuple[str, str, str]] = set()

        new_vector: list[float] | None = None
        new_vector_model: str | None = None
        if self.embedding_provider is not None and hasattr(store, "get_memory_embedding"):
            try:
                new_vector, new_vector_model = store.get_memory_embedding(
                    memory.id,
                    model=self.embedding_provider.name,
                )
            except Exception:  # noqa: BLE001 - defensive: never abort storage on edge errors
                new_vector = None

        candidate_vectors: dict[str, list[float]] = {}
        if new_vector is not None and hasattr(store, "list_memory_embeddings"):
            try:
                candidate_vectors = store.list_memory_embeddings(
                    [candidate.id for candidate in candidates],
                    model=new_vector_model,
                )
            except Exception:  # noqa: BLE001
                candidate_vectors = {}

        for candidate in candidates:
            if candidate.id == memory.id:
                continue
            for edge in self._derive_edges(
                memory,
                candidate,
                new_vector=new_vector,
                candidate_vector=candidate_vectors.get(candidate.id),
            ):
                key = (
                    edge.source_memory_id,
                    edge.target_memory_id,
                    edge.edge_type.value,
                )
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                try:
                    store.add_memory_edge(edge)
                except Exception:  # noqa: BLE001
                    continue
                emitted.append(edge)
        return emitted

    def _gather_edge_candidates(self, memory: MemoryRecord) -> list[MemoryRecord]:
        store = self.store
        seen: set[str] = {memory.id}
        candidates: list[MemoryRecord] = []
        if hasattr(store, "list_recent"):
            for record in store.list_recent(limit=EDGE_RECENT_LIMIT + 1):
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)
        if hasattr(store, "list_high_salience"):
            for record in store.list_high_salience(limit=EDGE_HIGH_SALIENCE_LIMIT):
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)
        if memory.domain and hasattr(store, "list_by_domain"):
            for record in store.list_by_domain(memory.domain, limit=EDGE_DOMAIN_LIMIT):
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)
        return candidates

    def _derive_edges(
        self,
        new_memory: MemoryRecord,
        other: MemoryRecord,
        *,
        new_vector: Sequence[float] | None,
        candidate_vector: Sequence[float] | None,
    ) -> list[MemoryEdge]:
        edges: list[MemoryEdge] = []
        new_metadata = new_memory.metadata if isinstance(new_memory.metadata, dict) else {}
        other_metadata = other.metadata if isinstance(other.metadata, dict) else {}

        new_goal_id = new_metadata.get("goal_id") or new_metadata.get("event_metadata", {}).get(
            "goal_id"
        ) if isinstance(new_metadata.get("event_metadata"), dict) else None
        other_goal_id = other_metadata.get("goal_id") or (
            other_metadata.get("event_metadata", {}).get("goal_id")
            if isinstance(other_metadata.get("event_metadata"), dict)
            else None
        )
        if new_goal_id and other_goal_id and new_goal_id == other_goal_id:
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.SAME_GOAL,
                    weight=1.0,
                    metadata={"goal_id": new_goal_id},
                )
            )

        new_tag_set = set(new_memory.tags)
        other_tag_set = set(other.tags)
        shared_tags = sorted(new_tag_set & other_tag_set)
        if len(shared_tags) >= TAG_OVERLAP_EDGE_THRESHOLD:
            denominator = len(new_tag_set | other_tag_set) or 1
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.TAG_OVERLAP,
                    weight=len(shared_tags) / denominator,
                    metadata={"shared_tags": shared_tags},
                )
            )

        if new_memory.domain and new_memory.domain == other.domain:
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.SAME_DOMAIN,
                    weight=1.0,
                    metadata={"domain": new_memory.domain},
                )
            )

        new_tokens = tokenize(new_memory.content)
        other_tokens = tokenize(other.content)
        keyword_overlap = 0.0
        if new_tokens and other_tokens:
            shared_tokens = new_tokens & other_tokens
            keyword_overlap = (
                len(shared_tokens) / len(new_tokens | other_tokens) if shared_tokens else 0.0
            )
        if keyword_overlap >= KEYWORD_OVERLAP_EDGE_THRESHOLD:
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.KEYWORD_SIMILARITY,
                    weight=keyword_overlap,
                    metadata={"keyword_overlap": round(keyword_overlap, 3)},
                )
            )

        delta = abs((new_memory.created_at - other.created_at).total_seconds())
        if delta <= TEMPORAL_PROXIMITY_WINDOW.total_seconds():
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.TEMPORAL_PROXIMITY,
                    weight=max(0.0, 1.0 - (delta / TEMPORAL_PROXIMITY_WINDOW.total_seconds())),
                    metadata={"seconds_apart": int(delta)},
                )
            )

        role_pair = self._role_related_pair(new_memory, other)
        if role_pair is not None:
            edges.append(
                MemoryEdge(
                    source_memory_id=new_memory.id,
                    target_memory_id=other.id,
                    edge_type=MemoryEdgeType.ROLE_RELATED,
                    weight=0.5,
                    metadata={"role_pair": role_pair},
                )
            )

        if new_vector is not None and candidate_vector is not None:
            from fullerene.memory.embeddings import cosine_similarity

            similarity = cosine_similarity(new_vector, candidate_vector)
            if similarity >= SEMANTIC_EDGE_THRESHOLD:
                edges.append(
                    MemoryEdge(
                        source_memory_id=new_memory.id,
                        target_memory_id=other.id,
                        edge_type=MemoryEdgeType.SEMANTIC_SIMILARITY,
                        weight=max(0.0, min(similarity, 1.0)),
                        metadata={"similarity": round(similarity, 3)},
                    )
                )

        return edges

    @staticmethod
    def _role_related_pair(new: MemoryRecord, other: MemoryRecord) -> str | None:
        roles = {(new.role or "unknown"), (other.role or "unknown")}
        if new.domain and new.domain == other.domain:
            if {"question", "preference"}.issubset(roles):
                return "question_preference"
            if {"task", "outcome"}.issubset(roles):
                return "task_outcome"
            if {"task", "feedback"}.issubset(roles):
                return "task_feedback"
        if {"goal", "task"}.issubset(roles):
            return "goal_task"
        return None

    # ---- Retrieval ----------------------------------------------------

    def _retrieve_relevant(
        self,
        event: Event,
        *,
        exclude_id: str | None,
    ) -> dict[str, Any]:
        store = self.store
        retrieve_limit = self.retrieve_limit + (1 if exclude_id is not None else 0)
        intent = classify_query_intent(event.content)
        event_domain = infer_domain(event.content, extract_event_tags(event))

        if hasattr(store, "hybrid_retrieve_relevant"):
            event_vector: list[float] | None = None
            embedding_model: str | None = None
            if self.embedding_provider is not None:
                vector, _error = safe_embed(self.embedding_provider, event.content)
                if vector is not None:
                    event_vector = vector
                    embedding_model = self.embedding_provider.name
            try:
                ranked_pairs = store.hybrid_retrieve_relevant(
                    event,
                    limit=retrieve_limit,
                    embedding_provider_name=embedding_model,
                    event_vector=event_vector,
                    domain_hint=event_domain,
                )
            except Exception:  # noqa: BLE001 - hybrid retrieval falls back to v1
                ranked_pairs = []
            else:
                filtered = [
                    pair for pair in ranked_pairs if pair[0].id != exclude_id
                ][: self.retrieve_limit]
                return {
                    "records": [pair[0] for pair in filtered],
                    "breakdowns": [pair[1] for pair in filtered],
                    "strategy": "hybrid_v2_with_embeddings"
                    if event_vector is not None
                    else "hybrid_v2_deterministic",
                    "query_intent": intent.value,
                    "event_domain": event_domain,
                }

        # Fallback: deterministic v1 retrieval, with on-the-fly hybrid
        # breakdowns so debug/score-breakdown surfaces stay populated.
        records = [
            memory
            for memory in store.retrieve_relevant(event, limit=retrieve_limit)
            if exclude_id is None or memory.id != exclude_id
        ][: self.retrieve_limit]
        breakdowns = [
            explain_hybrid_score(
                event,
                memory,
                event_vector=None,
                memory_vector=None,
                query_intent=intent,
                event_domain=event_domain,
            )
            for memory in records
        ]
        return {
            "records": records,
            "breakdowns": breakdowns,
            "strategy": "deterministic_v1_fallback",
            "query_intent": intent.value,
            "event_domain": event_domain,
        }

    # ---- Output -------------------------------------------------------

    @staticmethod
    def _describe_memory(
        memory: MemoryRecord | None,
        *,
        breakdown: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if memory is None:
            return None
        payload: dict[str, Any] = {
            "id": memory.id,
            "created_at": memory.created_at.isoformat(),
            "memory_type": memory.memory_type.value,
            "source_event_id": memory.source_event_id,
            "salience": memory.salience,
            "confidence": memory.confidence,
            "tags": list(memory.tags),
            "content_preview": memory.content[:120],
            "role": memory.role,
            "domain": memory.domain,
        }
        if breakdown is not None:
            payload["hybrid_score"] = round(float(breakdown.get("total", 0.0)), 4)
            payload["score_breakdown"] = breakdown
        return payload
