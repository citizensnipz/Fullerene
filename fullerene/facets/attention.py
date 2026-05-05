"""Deterministic attention facet for Fullerene Attention v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fullerene.attention import (
    AttentionBroadcast,
    AttentionConflict,
    AttentionHistoryEntry,
    AttentionItem,
    AttentionMode,
    AttentionResult,
    AttentionSource,
    FixedWeightAttentionScorer,
)
from fullerene.memory import MemoryRecord, MemoryStore, extract_event_tags
from fullerene.memory.scoring import explain_score
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState

ATTENTION_BROADCAST_RECIPIENTS = [
    "memory",
    "goals",
    "world_model",
    "behavior",
    "context",
    "nexus",
]
ATTENTION_CONFLICT_THRESHOLD = 0.05


class AttentionFacet:
    """Score current-cycle focus candidates and broadcast the winner."""

    name = "attention"

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        *,
        top_n: int = 3,
        memory_limit: int | None = None,
        history_size: int = 20,
        scorer: FixedWeightAttentionScorer | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.top_n = max(int(top_n), 1)
        self.memory_limit = max(int(memory_limit or max(self.top_n, 3)), 1)
        self.history_size = max(int(history_size), 1)
        self.scorer = scorer or FixedWeightAttentionScorer()

    def process(self, event: Event, state: NexusState) -> FacetResult:
        candidates, reasons, limitations = self._build_candidates(event, state)
        evaluation = self.scorer.evaluate(candidates, top_n=self.top_n)
        ranked = [self._annotate_attention_mode(item) for item in evaluation["ranked"]]
        conflict = self._detect_conflict(ranked, event)
        ranked = self._apply_conflict_resolution(ranked, conflict)
        focus_payloads = [
            item for item in ranked[: self.top_n] if float(item["score"]) > 0.0
        ]
        focus_items = [
            AttentionItem(
                id=item["id"],
                source=AttentionSource(item["source"]),
                source_id=item.get("source_id"),
                content=item["content"],
                score=item["score"],
                components=item["components"],
                dominant_component=item.get("dominant_component"),
                metadata=item.get("metadata", {}),
            )
            for item in focus_payloads
        ]
        dominant_source = focus_items[0].source if focus_items else None
        prior_history = self._history_from_state(state)
        winner_payload = ranked[0] if ranked and float(ranked[0]["score"]) > 0.0 else None
        broadcast = self._build_broadcast(
            winner_payload,
            conflict=conflict,
            history=prior_history,
            event=event,
        )
        attention_history = self._updated_history(prior_history, broadcast)
        attention_result = AttentionResult(
            focus_items=focus_items,
            scores={item["id"]: item["score"] for item in ranked},
            dominant_source=dominant_source,
            strategy=evaluation["strategy"],
            metadata={
                "candidate_count": len(candidates),
                "selected_count": len(focus_items),
                "ranked_item_ids": [item["id"] for item in ranked],
                "available_sources": sorted({item["source"] for item in ranked}),
                "limitations": list(limitations),
                "weights": dict(self.scorer.weights),
                "attention_version": "v1",
                "broadcast": broadcast.to_dict() if broadcast is not None else None,
                "broadcast_item_id": (
                    broadcast.item_id if broadcast is not None else None
                ),
                "broadcast_mode": (
                    broadcast.mode.value if broadcast is not None else None
                ),
                "recipients": (
                    list(broadcast.recipients) if broadcast is not None else []
                ),
                "pressure_contribution": (
                    broadcast.pressure_contribution if broadcast is not None else 0.0
                ),
                "attention_history_count": len(attention_history),
                "attention_conflict": conflict is not None,
                "conflict": conflict.to_dict() if conflict is not None else None,
                "conflict_items": (
                    list(conflict.item_ids) if conflict is not None else []
                ),
                "score_delta": (
                    conflict.score_delta if conflict is not None else None
                ),
            },
        )
        focus_item_payload = [item.to_dict() for item in focus_items]
        dominant_source_value = (
            dominant_source.value if dominant_source is not None else None
        )
        summary = self._summary(focus_items, candidate_count=len(candidates))
        return FacetResult(
            facet_name=self.name,
            summary=summary,
            proposed_decision=(
                DecisionAction.RECORD if focus_items else DecisionAction.WAIT
            ),
            state_updates={
                "last_attention_result": attention_result.to_dict(),
                "last_focus_item_ids": [item.id for item in focus_items],
                "last_dominant_source": dominant_source_value,
                "last_strategy": attention_result.strategy,
                "last_scores": dict(attention_result.scores),
                "last_attention_broadcast": (
                    broadcast.to_dict() if broadcast is not None else None
                ),
                "last_attention_broadcast_item_id": (
                    broadcast.item_id if broadcast is not None else None
                ),
                "last_attention_mode": (
                    broadcast.mode.value if broadcast is not None else None
                ),
                "last_attention_conflict": (
                    conflict.to_dict() if conflict is not None else None
                ),
                "attention_history": [
                    entry.to_dict() for entry in attention_history
                ],
                "last_attention_history_count": len(attention_history),
                "last_attention_pressure_contribution": (
                    broadcast.pressure_contribution if broadcast is not None else 0.0
                ),
            },
            metadata={
                "attention_result": attention_result.to_dict(),
                "focus_items": focus_item_payload,
                "scores": dict(attention_result.scores),
                "dominant_source": dominant_source_value,
                "strategy": attention_result.strategy,
                "top_n": self.top_n,
                "history_size": self.history_size,
                "weights": dict(self.scorer.weights),
                "broadcast": broadcast.to_dict() if broadcast is not None else None,
                "broadcast_item_id": (
                    broadcast.item_id if broadcast is not None else None
                ),
                "broadcast_mode": (
                    broadcast.mode.value if broadcast is not None else None
                ),
                "recipients": (
                    list(broadcast.recipients) if broadcast is not None else []
                ),
                "pressure_contribution": (
                    broadcast.pressure_contribution if broadcast is not None else 0.0
                ),
                "attention_conflict": conflict is not None,
                "conflict": conflict.to_dict() if conflict is not None else None,
                "conflict_items": (
                    list(conflict.item_ids) if conflict is not None else []
                ),
                "score_delta": (
                    conflict.score_delta if conflict is not None else None
                ),
                "attention_history": [
                    entry.to_dict() for entry in attention_history
                ],
                "attention_history_count": len(attention_history),
                "reasons": reasons,
                "limitations": limitations,
            },
        )

    def _build_candidates(
        self,
        event: Event,
        state: NexusState,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        reasons: list[str] = ["generated_event_candidate"]
        limitations: list[str] = []
        pressure = self._coerce_unit(event.metadata.get("pressure"))
        explicit_novelty = self._coerce_optional_unit(event.metadata.get("novelty"))
        event_tags = extract_event_tags(event)

        memory_records = self._memory_candidates(event)
        stored_event_memory = self._stored_event_memory(event)
        if memory_records:
            reasons.append(f"memory_candidates={len(memory_records)}")
        else:
            reasons.append("memory_candidates=0")
            limitations.append("memory_candidates_unavailable_or_no_matches")

        goal_matches = self._facet_matches(
            state,
            facet_name="goals",
            key="last_relevant_goals",
        )
        if goal_matches:
            reasons.append(f"goal_candidates={len(goal_matches)}")
        else:
            reasons.append("goal_candidates=0")
            limitations.append("goal_candidates_unavailable_or_no_matches")

        belief_matches = self._facet_matches(
            state,
            facet_name="world_model",
            key="last_relevant_beliefs",
        )
        if belief_matches:
            reasons.append(f"belief_candidates={len(belief_matches)}")
        else:
            reasons.append("belief_candidates=0")
            limitations.append("belief_candidates_unavailable_or_no_matches")

        execution_result = self._execution_result(state)
        if execution_result is not None:
            reasons.append("execution_candidate=1")
        else:
            reasons.append("execution_candidate=0")
            limitations.append("execution_candidate_unavailable")

        novelty = (
            explicit_novelty
            if explicit_novelty is not None
            else self._heuristic_event_novelty(
                event=event,
                event_tags=event_tags,
                memory_records=memory_records,
                goal_matches=goal_matches,
                belief_matches=belief_matches,
            )
        )
        if explicit_novelty is not None:
            reasons.append("used_explicit_event_novelty")
        if pressure > 0.0:
            reasons.append("used_event_pressure")

        top_goal_priority = self._top_numeric(goal_matches, key="priority")
        top_belief_uncertainty = self._top_uncertainty(belief_matches)
        execution_recency = self._execution_recency(
            execution_result,
            reference_time=event.timestamp,
        )

        candidates: list[dict[str, Any]] = [
            {
                "id": f"event:{event.event_id}",
                "source": AttentionSource.EVENT.value,
                "source_id": event.event_id,
                "content": event.content or event.event_type.value,
                "memory_salience": (
                    stored_event_memory.salience if stored_event_memory is not None else 0.0
                ),
                "goal_priority": top_goal_priority,
                "pressure": pressure,
                "novelty": novelty,
                "belief_uncertainty": top_belief_uncertainty,
                "execution_recency": execution_recency,
                "metadata": {
                    "event_type": event.event_type.value,
                    "event_tags": sorted(event_tags),
                    "has_stored_event_memory": stored_event_memory is not None,
                    "top_goal_priority": top_goal_priority,
                    "top_belief_uncertainty": top_belief_uncertainty,
                    "execution_recency": execution_recency,
                    "novelty_reason": (
                        "explicit_metadata"
                        if explicit_novelty is not None
                        else "heuristic_event_novelty_v0"
                    ),
                },
            }
        ]
        candidates.extend(
            self._memory_candidate_payloads(
                event,
                memory_records=memory_records,
                pressure=pressure,
            )
        )
        candidates.extend(
            self._goal_candidate_payloads(goal_matches, pressure=pressure)
        )
        candidates.extend(
            self._belief_candidate_payloads(belief_matches, pressure=pressure)
        )
        execution_candidate = self._execution_candidate_payload(
            execution_result,
            pressure=pressure,
            reference_time=event.timestamp,
        )
        if execution_candidate is not None:
            candidates.append(execution_candidate)
        return candidates, reasons, sorted(set(limitations))

    def _memory_candidates(self, event: Event) -> list[MemoryRecord]:
        if self.memory_store is None:
            return []
        records = self.memory_store.retrieve_relevant(event, limit=self.memory_limit)
        return [
            record
            for record in records
            if record.source_event_id != event.event_id
        ]

    def _stored_event_memory(self, event: Event) -> MemoryRecord | None:
        if self.memory_store is None:
            return None
        recent = self.memory_store.list_recent(limit=max(self.memory_limit * 2, 6))
        for record in recent:
            if record.source_event_id == event.event_id:
                return record
        return None

    @staticmethod
    def _facet_matches(
        state: NexusState,
        *,
        facet_name: str,
        key: str,
    ) -> list[dict[str, Any]]:
        facet_state = state.facet_state.get(facet_name)
        if not isinstance(facet_state, dict):
            return []
        raw_matches = facet_state.get(key)
        if not isinstance(raw_matches, list):
            return []
        return [
            dict(item)
            for item in raw_matches
            if isinstance(item, dict)
        ]

    @staticmethod
    def _execution_result(state: NexusState) -> dict[str, Any] | None:
        facet_state = state.facet_state.get("executor")
        if not isinstance(facet_state, dict):
            return None
        raw_result = facet_state.get("last_execution_result")
        if not isinstance(raw_result, dict):
            return None
        return dict(raw_result)

    def _memory_candidate_payloads(
        self,
        event: Event,
        *,
        memory_records: list[MemoryRecord],
        pressure: float,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for record in memory_records:
            payloads.append(
                {
                    "id": f"memory:{record.id}",
                    "source": AttentionSource.MEMORY.value,
                    "source_id": record.id,
                    "content": record.content,
                    "memory_salience": record.salience,
                    "pressure": pressure,
                    "metadata": {
                        "memory_type": record.memory_type.value,
                        "confidence": record.confidence,
                        "tags": list(record.tags),
                        "retrieval_breakdown": explain_score(event, record),
                    },
                }
            )
        return payloads

    @staticmethod
    def _goal_candidate_payloads(
        goal_matches: list[dict[str, Any]],
        *,
        pressure: float,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for match in goal_matches:
            payloads.append(
                {
                    "id": f"goal:{match['id']}",
                    "source": AttentionSource.GOAL.value,
                    "source_id": match["id"],
                    "content": str(match.get("description", "")),
                    "goal_priority": AttentionFacet._coerce_unit(match.get("priority")),
                    "pressure": pressure,
                    "metadata": {
                        "goal_match": dict(match),
                    },
                }
            )
        return payloads

    @staticmethod
    def _belief_candidate_payloads(
        belief_matches: list[dict[str, Any]],
        *,
        pressure: float,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for match in belief_matches:
            confidence = AttentionFacet._coerce_unit(match.get("confidence"))
            payloads.append(
                {
                    "id": f"belief:{match['id']}",
                    "source": AttentionSource.BELIEF.value,
                    "source_id": match["id"],
                    "content": str(match.get("claim", "")),
                    "belief_uncertainty": 1.0 - confidence,
                    "pressure": pressure,
                    "metadata": {
                        "belief_match": dict(match),
                    },
                }
            )
        return payloads

    def _execution_candidate_payload(
        self,
        execution_result: dict[str, Any] | None,
        *,
        pressure: float,
        reference_time: datetime,
    ) -> dict[str, Any] | None:
        if execution_result is None:
            return None
        plan_id = execution_result.get("plan_id")
        overall_status = str(execution_result.get("overall_status", "")).strip() or "unknown"
        reasons = execution_result.get("reasons", [])
        dry_run = bool(execution_result.get("dry_run", True))
        execution_recency = self._execution_recency(
            execution_result,
            reference_time=reference_time,
        )
        return {
            "id": f"execution:{plan_id or overall_status}",
            "source": AttentionSource.EXECUTION.value,
            "source_id": plan_id,
            "content": f"Execution {overall_status} ({'dry-run' if dry_run else 'live'})",
            "pressure": pressure,
            "execution_recency": execution_recency,
            "metadata": {
                "execution_result": execution_result,
                "reasons": list(reasons) if isinstance(reasons, list) else [],
            },
        }

    @staticmethod
    def _annotate_attention_mode(item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        components = dict(payload.get("components") or {})
        goal_priority = float(components.get("goal_priority", 0.0))
        novelty = float(components.get("novelty", 0.0))
        pressure = float(components.get("pressure", 0.0))
        execution_recency = float(components.get("execution_recency", 0.0))
        if goal_priority >= novelty and goal_priority > 0.0:
            mode = AttentionMode.TOP_DOWN
        elif pressure + goal_priority >= novelty + execution_recency:
            mode = AttentionMode.TOP_DOWN
        else:
            mode = AttentionMode.BOTTOM_UP
        metadata = dict(payload.get("metadata") or {})
        metadata["mode"] = mode.value
        payload["metadata"] = metadata
        payload["mode"] = mode.value
        return payload

    def _detect_conflict(
        self,
        ranked: list[dict[str, Any]],
        event: Event,
    ) -> AttentionConflict | None:
        if len(ranked) < 2:
            return None
        first, second = ranked[0], ranked[1]
        score_delta = abs(float(first.get("score", 0.0)) - float(second.get("score", 0.0)))
        if score_delta > ATTENTION_CONFLICT_THRESHOLD:
            return None
        return AttentionConflict(
            id=f"attention-conflict:{event.event_id}",
            item_ids=[str(first.get("id") or ""), str(second.get("id") or "")],
            score_delta=round(score_delta, 3),
            reason="close_score_competition",
            metadata={
                "modes": [
                    str(first.get("mode") or ""),
                    str(second.get("mode") or ""),
                ],
                "sources": [
                    str(first.get("source") or ""),
                    str(second.get("source") or ""),
                ],
            },
        )

    def _apply_conflict_resolution(
        self,
        ranked: list[dict[str, Any]],
        conflict: AttentionConflict | None,
    ) -> list[dict[str, Any]]:
        if conflict is None or len(ranked) < 2:
            return ranked
        first, second = ranked[0], ranked[1]
        preferred = self._preferred_conflict_item(first, second)
        if preferred is first:
            return ranked
        return [second, first, *ranked[2:]]

    @staticmethod
    def _preferred_conflict_item(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> dict[str, Any]:
        first_mode = str(first.get("mode") or AttentionMode.BOTTOM_UP.value)
        second_mode = str(second.get("mode") or AttentionMode.BOTTOM_UP.value)
        if first_mode != second_mode:
            if second_mode == AttentionMode.TOP_DOWN.value:
                return second
            return first
        first_score = float(first.get("score", 0.0))
        second_score = float(second.get("score", 0.0))
        if second_score > first_score:
            return second
        if first_score > second_score:
            return first
        if (str(second.get("source") or ""), str(second.get("id") or "")) < (
            str(first.get("source") or ""),
            str(first.get("id") or ""),
        ):
            return second
        return first

    def _build_broadcast(
        self,
        winner_payload: dict[str, Any] | None,
        *,
        conflict: AttentionConflict | None,
        history: list[AttentionHistoryEntry],
        event: Event,
    ) -> AttentionBroadcast | None:
        if winner_payload is None:
            return None
        normalized_content = self._normalize_content(
            str(winner_payload.get("content") or "")
        )
        repeated_count = self._repeated_count(
            history,
            item_id=str(winner_payload.get("id") or ""),
            normalized_content=normalized_content,
        )
        pressure_contribution = min(repeated_count * 0.05, 0.25)
        metadata = dict(winner_payload.get("metadata") or {})
        metadata["normalized_content"] = normalized_content
        metadata["repeated_count"] = repeated_count
        metadata["pressure_contribution"] = pressure_contribution
        return AttentionBroadcast(
            id=f"attention-broadcast:{event.event_id}",
            created_at=event.timestamp,
            item_id=str(winner_payload.get("id") or ""),
            source=AttentionSource(str(winner_payload.get("source") or "system")),
            source_id=winner_payload.get("source_id"),
            content=str(winner_payload.get("content") or ""),
            score=float(winner_payload.get("score", 0.0)),
            mode=AttentionMode(str(winner_payload.get("mode") or AttentionMode.BOTTOM_UP.value)),
            components=dict(winner_payload.get("components") or {}),
            metadata=metadata,
            recipients=list(ATTENTION_BROADCAST_RECIPIENTS),
            conflict_ids=[conflict.id] if conflict is not None else [],
            repeated_count=repeated_count,
            pressure_contribution=round(pressure_contribution, 3),
        )

    def _history_from_state(self, state: NexusState) -> list[AttentionHistoryEntry]:
        facet_state = state.facet_state.get(self.name)
        if not isinstance(facet_state, dict):
            return []
        raw_history = facet_state.get("attention_history")
        if not isinstance(raw_history, list):
            return []
        history: list[AttentionHistoryEntry] = []
        for raw_entry in raw_history:
            if not isinstance(raw_entry, dict):
                continue
            try:
                history.append(AttentionHistoryEntry.from_dict(raw_entry))
            except (KeyError, TypeError, ValueError):
                continue
        return history[-self.history_size :]

    def _updated_history(
        self,
        history: list[AttentionHistoryEntry],
        broadcast: AttentionBroadcast | None,
    ) -> list[AttentionHistoryEntry]:
        if broadcast is None:
            return history[-self.history_size :]
        next_history = list(history)
        next_history.append(
            AttentionHistoryEntry(
                broadcast_id=broadcast.id,
                item_id=broadcast.item_id,
                source=broadcast.source,
                source_id=broadcast.source_id,
                score=broadcast.score,
                created_at=broadcast.created_at,
                metadata={
                    "mode": broadcast.mode.value,
                    "normalized_content": broadcast.metadata.get("normalized_content", ""),
                    "repeated_count": broadcast.repeated_count,
                    "pressure_contribution": broadcast.pressure_contribution,
                },
            )
        )
        return next_history[-self.history_size :]

    def _repeated_count(
        self,
        history: list[AttentionHistoryEntry],
        *,
        item_id: str,
        normalized_content: str,
    ) -> int:
        repeated_count = 0
        for entry in history[-self.history_size :]:
            entry_normalized_content = str(
                entry.metadata.get("normalized_content", "")
            )
            if entry.item_id == item_id or (
                normalized_content and entry_normalized_content == normalized_content
            ):
                repeated_count += 1
        return repeated_count

    @staticmethod
    def _normalize_content(content: str) -> str:
        return " ".join(content.casefold().split())

    @staticmethod
    def _top_numeric(matches: list[dict[str, Any]], *, key: str) -> float:
        values = [
            AttentionFacet._coerce_unit(match.get(key))
            for match in matches
        ]
        return max(values, default=0.0)

    @staticmethod
    def _top_uncertainty(matches: list[dict[str, Any]]) -> float:
        uncertainties = [
            1.0 - AttentionFacet._coerce_unit(match.get("confidence"))
            for match in matches
        ]
        return max(uncertainties, default=0.0)

    @staticmethod
    def _execution_recency(
        execution_result: dict[str, Any] | None,
        *,
        reference_time: datetime,
    ) -> float:
        if execution_result is None:
            return 0.0
        records = execution_result.get("records", [])
        latest_record_time: datetime | None = None
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                raw_created_at = record.get("created_at")
                if not isinstance(raw_created_at, str):
                    continue
                try:
                    created_at = datetime.fromisoformat(raw_created_at)
                except ValueError:
                    continue
                if latest_record_time is None or created_at > latest_record_time:
                    latest_record_time = created_at
        if latest_record_time is None:
            return 0.5
        age_seconds = max((reference_time - latest_record_time).total_seconds(), 0.0)
        age_hours = age_seconds / 3600.0
        return 1.0 / (1.0 + age_hours)

    @staticmethod
    def _heuristic_event_novelty(
        *,
        event: Event,
        event_tags: set[str],
        memory_records: list[MemoryRecord],
        goal_matches: list[dict[str, Any]],
        belief_matches: list[dict[str, Any]],
    ) -> float:
        if not event.content.strip():
            return 0.0
        novelty = 0.0
        if not memory_records:
            novelty += 0.2
        if event_tags:
            known_tags = {
                tag
                for record in memory_records
                for tag in record.tags
            }
            if event_tags - known_tags:
                novelty += 0.1
        if not goal_matches and not belief_matches:
            novelty += 0.05
        return AttentionFacet._coerce_unit(novelty)

    @staticmethod
    def _coerce_unit(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_optional_unit(value: Any) -> float | None:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _summary(
        focus_items: list[AttentionItem],
        *,
        candidate_count: int,
    ) -> str:
        if not focus_items:
            return (
                "Attention facet scored the available candidates but selected no "
                f"focus items above zero from {candidate_count} candidates."
            )
        top_item = focus_items[0]
        return (
            f"Attention facet selected {len(focus_items)} focus item(s) from "
            f"{candidate_count} candidates; top source {top_item.source.value} "
            f"scored {top_item.score:.3f}."
        )
