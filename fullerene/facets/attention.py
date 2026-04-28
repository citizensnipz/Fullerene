"""Deterministic attention facet for Fullerene Attention v0."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fullerene.attention import (
    AttentionItem,
    AttentionResult,
    AttentionSource,
    FixedWeightAttentionScorer,
)
from fullerene.memory import MemoryRecord, MemoryStore, extract_event_tags
from fullerene.memory.scoring import explain_score
from fullerene.nexus.models import DecisionAction, Event, FacetResult, NexusState


class AttentionFacet:
    """Score current-cycle focus candidates without broadcasting them."""

    name = "attention"

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        *,
        top_n: int = 3,
        memory_limit: int | None = None,
        scorer: FixedWeightAttentionScorer | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.top_n = max(int(top_n), 1)
        self.memory_limit = max(int(memory_limit or max(self.top_n, 3)), 1)
        self.scorer = scorer or FixedWeightAttentionScorer()

    def process(self, event: Event, state: NexusState) -> FacetResult:
        candidates, reasons, limitations = self._build_candidates(event, state)
        evaluation = self.scorer.evaluate(candidates, top_n=self.top_n)
        ranked = evaluation["ranked"]
        focus_payloads = [
            item for item in evaluation["selected"] if float(item["score"]) > 0.0
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
        attention_result = AttentionResult(
            focus_items=focus_items,
            scores=evaluation["scores"],
            dominant_source=dominant_source,
            strategy=evaluation["strategy"],
            metadata={
                "candidate_count": len(candidates),
                "selected_count": len(focus_items),
                "ranked_item_ids": [item["id"] for item in ranked],
                "available_sources": sorted({item["source"] for item in ranked}),
                "limitations": list(limitations),
                "weights": dict(self.scorer.weights),
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
            },
            metadata={
                "attention_result": attention_result.to_dict(),
                "focus_items": focus_item_payload,
                "scores": dict(attention_result.scores),
                "dominant_source": dominant_source_value,
                "strategy": attention_result.strategy,
                "top_n": self.top_n,
                "weights": dict(self.scorer.weights),
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
