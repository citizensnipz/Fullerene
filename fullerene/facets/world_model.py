"""Deterministic world model facet for Fullerene World Model v0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, FacetResult, NexusState
from fullerene.world_model import Belief, SQLiteWorldModelStore, WorldModelStore


@dataclass(slots=True)
class _BeliefMatch:
    belief: Belief
    score: float
    tag_overlap: float
    keyword_overlap: float
    shared_tags: list[str]
    shared_keywords: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.belief.id,
            "claim": self.belief.claim,
            "confidence": self.belief.confidence,
            "status": self.belief.status.value,
            "score": round(self.score, 3),
            "tag_overlap": round(self.tag_overlap, 3),
            "keyword_overlap": round(self.keyword_overlap, 3),
            "shared_tags": list(self.shared_tags),
            "shared_keywords": list(self.shared_keywords),
            "source": self.belief.source.value,
        }


class WorldModelFacet:
    """Expose deterministic belief relevance signals without inference logic."""

    name = "world_model"

    def __init__(
        self,
        store: WorldModelStore,
        *,
        active_limit: int = 20,
        relevant_limit: int = 3,
    ) -> None:
        self.store = store
        self.active_limit = max(int(active_limit), 1)
        self.relevant_limit = max(int(relevant_limit), 1)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        active_limit: int = 20,
        relevant_limit: int = 3,
    ) -> "WorldModelFacet":
        return cls(
            SQLiteWorldModelStore(path),
            active_limit=active_limit,
            relevant_limit=relevant_limit,
        )

    def process(self, event: Event, state: NexusState) -> FacetResult:
        del state

        active_beliefs = self.store.list_active_beliefs(limit=self.active_limit)
        if not active_beliefs:
            return FacetResult(
                facet_name=self.name,
                summary="World model facet found no active beliefs.",
                state_updates={
                    "last_active_belief_ids": [],
                    "last_relevant_beliefs": [],
                    "last_relevance_score": 0.0,
                },
                metadata={
                    "active_belief_count": 0,
                    "relevant_beliefs": [],
                    "relevance_score": 0.0,
                    "score_formula": "tag_overlap + keyword_overlap + confidence",
                },
            )

        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        relevant_matches = [
            match
            for belief in active_beliefs
            if (
                match := self._score_belief(
                    belief,
                    event_tags=event_tags,
                    event_keywords=event_keywords,
                )
            )
            is not None
        ]
        relevant_matches.sort(
            key=lambda match: (
                match.score,
                match.belief.confidence,
                match.belief.updated_at.timestamp(),
                match.belief.id,
            ),
            reverse=True,
        )
        relevant_matches = relevant_matches[: self.relevant_limit]
        relevance_score = (
            round(relevant_matches[0].score, 3) if relevant_matches else 0.0
        )

        if relevant_matches:
            summary = (
                f"World model facet matched {len(relevant_matches)} active beliefs; "
                f"top relevance score {relevance_score:.3f}."
            )
        else:
            summary = (
                f"World model facet checked {len(active_beliefs)} active beliefs "
                "and found no relevant matches."
            )

        relevant_belief_payload = [match.to_dict() for match in relevant_matches]
        return FacetResult(
            facet_name=self.name,
            summary=summary,
            state_updates={
                "last_active_belief_ids": [belief.id for belief in active_beliefs],
                "last_relevant_beliefs": relevant_belief_payload,
                "last_relevance_score": relevance_score,
            },
            metadata={
                "active_belief_count": len(active_beliefs),
                "relevant_beliefs": relevant_belief_payload,
                "relevance_score": relevance_score,
                "event_tags": sorted(event_tags),
                "event_keywords": sorted(event_keywords),
                "score_formula": "tag_overlap + keyword_overlap + confidence",
            },
        )

    @staticmethod
    def _score_belief(
        belief: Belief,
        *,
        event_tags: set[str],
        event_keywords: set[str],
    ) -> _BeliefMatch | None:
        belief_tags = set(belief.tags)
        belief_keywords = tokenize(belief.claim)
        shared_tags = sorted(event_tags & belief_tags)
        shared_keywords = sorted(event_keywords & belief_keywords)

        if not shared_tags and not shared_keywords:
            return None

        tag_overlap = len(shared_tags) / len(belief_tags) if belief_tags else 0.0
        keyword_overlap = (
            len(shared_keywords) / len(belief_keywords) if belief_keywords else 0.0
        )
        score = tag_overlap + keyword_overlap + belief.confidence

        return _BeliefMatch(
            belief=belief,
            score=score,
            tag_overlap=tag_overlap,
            keyword_overlap=keyword_overlap,
            shared_tags=shared_tags,
            shared_keywords=shared_keywords,
        )
