"""Deterministic world model facet for Fullerene World Model v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from fullerene.memory import extract_event_tags, tokenize
from fullerene.nexus.models import Event, FacetResult, NexusState
from fullerene.world_model import Belief, BeliefSource, BeliefStatus, SQLiteWorldModelStore, WorldModelStore
from fullerene.world_model.models import BeliefType, normalize_statement, stable_belief_id, utcnow


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
    """Expose deterministic belief lifecycle and relevance signals."""

    name = "world_model"

    def __init__(
        self,
        store: WorldModelStore,
        *,
        active_limit: int = 20,
        relevant_limit: int = 3,
        support_weight: float = 0.2,
        contradiction_weight: float = 0.3,
        contradiction_threshold: int = 2,
        low_confidence_threshold: float = 0.35,
    ) -> None:
        self.store = store
        self.active_limit = max(int(active_limit), 1)
        self.relevant_limit = max(int(relevant_limit), 1)
        self.support_weight = max(0.0, min(float(support_weight), 1.0))
        self.contradiction_weight = max(0.0, min(float(contradiction_weight), 1.0))
        self.contradiction_threshold = max(int(contradiction_threshold), 1)
        self.low_confidence_threshold = max(0.0, min(float(low_confidence_threshold), 1.0))

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
        world_updates = self._ingest_event_as_belief(event, state)
        active_beliefs = self.store.list_beliefs(limit=self.active_limit)
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
                    "world_model_updates": world_updates,
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
                "world_model_updates": world_updates,
                "contradiction_signals": world_updates.get("pressure_signals", []),
            },
        )

    def _ingest_event_as_belief(self, event: Event, state: NexusState) -> dict[str, object]:
        content = event.content.strip()
        if not content:
            return {"updated_belief_ids": [], "pressure_signals": []}
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        explicit_belief_create = bool(metadata.get("create_belief"))
        is_question_like = content.endswith("?")
        if not explicit_belief_create and is_question_like:
            return {"updated_belief_ids": [], "pressure_signals": []}
        normalized_key = normalize_statement(content)
        if not normalized_key:
            return {"updated_belief_ids": [], "pressure_signals": []}
        belief = self.store.get_belief_by_normalized_key(normalized_key)
        if belief is None:
            belief = self._find_contradicted_match(content, normalized_key)
        elif explicit_belief_create and belief.source_event_id == event.event_id:
            # CLI metadata-created belief already exists for this event; avoid
            # counting the same event twice.
            return {"updated_belief_ids": [belief.id], "pressure_signals": []}
        if belief is None:
            belief = Belief(
                id=stable_belief_id(normalized_key),
                claim=content,
                confidence=0.6,
                status=BeliefStatus.VALID,
                source=BeliefSource.MEMORY,
                source_event_id=event.event_id,
                normalized_key=normalized_key,
                support_count=1,
                contradiction_count=0,
                last_support_event_id=event.event_id,
                last_updated_event_id=event.event_id,
                belief_type=self._classify_belief_type(content),
                sources=[event.event_id],
                metadata={
                    "origin": "memory",
                    "domain": metadata.get("event_domain"),
                    "tags": sorted(extract_event_tags(event)),
                },
            )
            self.store.add_belief(belief)
            self._upsert_edges(belief)
            return {
                "updated_belief_ids": [belief.id],
                "pressure_signals": [],
                "created_belief_id": belief.id,
            }

        contradiction, contradiction_kind = self._is_contradiction(
            belief.claim,
            content,
            belief.normalized_key,
            normalized_key,
        )
        if contradiction:
            belief.contradiction_count += 1
            belief.confidence = self._clamp01(
                belief.confidence - (belief.confidence * self.contradiction_weight)
            )
            belief.last_contradiction_event_id = event.event_id
            if belief.contradiction_count >= self.contradiction_threshold:
                belief.status = BeliefStatus.CONTRADICTED
            belief.metadata["last_contradiction_kind"] = contradiction_kind
        else:
            belief.support_count += 1
            belief.confidence = self._clamp01(
                belief.confidence + ((1.0 - belief.confidence) * self.support_weight)
            )
            belief.last_support_event_id = event.event_id
            if normalized_key == belief.normalized_key:
                belief.status = BeliefStatus.REDUNDANT
            elif belief.status == BeliefStatus.CONTRADICTED and belief.confidence >= 0.5:
                belief.status = BeliefStatus.VALID

        belief.last_updated_event_id = event.event_id
        belief.updated_at = utcnow()
        belief.sources = sorted(set([*belief.sources, event.event_id]))
        belief.metadata["last_updated_timestamp"] = belief.updated_at.isoformat()
        self.store.update_belief(belief)
        self._upsert_edges(belief)
        pressure_signals = self._pressure_signals_for_belief(belief)
        return {
            "updated_belief_ids": [belief.id],
            "pressure_signals": pressure_signals,
            "contradiction_detected": contradiction,
        }

    def _find_contradicted_match(self, incoming_claim: str, incoming_key: str) -> Belief | None:
        incoming_tokens = tokenize(incoming_claim)
        for candidate in self.store.list_beliefs(limit=100):
            contradiction, _ = self._is_contradiction(
                candidate.claim,
                incoming_claim,
                candidate.normalized_key,
                incoming_key,
            )
            if not contradiction:
                continue
            candidate_tokens = tokenize(candidate.claim)
            overlap = len(incoming_tokens & candidate_tokens)
            union = len(incoming_tokens | candidate_tokens) or 1
            if (overlap / union) >= 0.25:
                return candidate
        return None

    def _upsert_edges(self, anchor: Belief) -> None:
        if not hasattr(self.store, "add_belief_edge"):
            return
        candidates = self.store.list_beliefs(limit=40)
        anchor_tokens = set(tokenize(anchor.claim))
        for belief in candidates:
            if belief.id == anchor.id:
                continue
            weight = 0.0
            edge_type = "related"
            shared = anchor_tokens & set(tokenize(belief.claim))
            if shared:
                weight = max(weight, len(shared) / max(len(anchor_tokens | set(tokenize(belief.claim))), 1))
            if set(anchor.sources) & set(belief.sources):
                weight = max(weight, 0.7)
                edge_type = "temporal"
            if weight >= 0.2:
                self.store.add_belief_edge(
                    source_belief_id=anchor.id,
                    target_belief_id=belief.id,
                    edge_type=edge_type,
                    weight=weight,
                    metadata={"shared_keywords": sorted(shared)},
                )

    def _pressure_signals_for_belief(self, belief: Belief) -> list[dict[str, object]]:
        signals: list[dict[str, object]] = []
        if belief.contradiction_count > 0:
            signals.append(
                {
                    "source": "world_model",
                    "entry_type": "contradiction",
                    "source_id": belief.id,
                    "description": f"Belief contradiction: {belief.claim[:100]}",
                    "metadata": {
                        "belief_id": belief.id,
                        "confidence": belief.confidence,
                        "contradiction_count": belief.contradiction_count,
                        "support_count": belief.support_count,
                        "priority": belief.priority,
                    },
                }
            )
        if belief.contradiction_count >= self.contradiction_threshold:
            signals[-1]["metadata"]["severity"] = "high"
        if belief.confidence <= self.low_confidence_threshold:
            signals.append(
                {
                    "source": "world_model",
                    "entry_type": "uncertainty",
                    "source_id": belief.id,
                    "description": f"Belief uncertainty: {belief.claim[:100]}",
                    "metadata": {"belief_id": belief.id, "belief_confidence": belief.confidence},
                }
            )
        return signals

    @staticmethod
    def _classify_belief_type(content: str) -> BeliefType:
        text = content.lower()
        if any(token in text for token in ("i like", "i prefer", "favorite")):
            return BeliefType.PREFERENCE
        if any(token in text for token in ("can ", "able to", "capable")):
            return BeliefType.CAPABILITY
        if text:
            return BeliefType.FACT
        return BeliefType.UNKNOWN

    @staticmethod
    def _is_contradiction(
        existing_claim: str,
        incoming_claim: str,
        existing_key: str,
        incoming_key: str,
    ) -> tuple[bool, str | None]:
        if existing_key == incoming_key:
            return False, None
        existing = existing_claim.lower()
        incoming = incoming_claim.lower()
        neg_pairs = [(" is ", " is not "), (" has ", " does not have "), (" can ", " cannot ")]
        for pos, neg in neg_pairs:
            if (pos in existing and neg in incoming) or (neg in existing and pos in incoming):
                return True, "direct_negation"
        num_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        existing_nums = num_pattern.findall(existing)
        incoming_nums = num_pattern.findall(incoming)
        if existing_nums and incoming_nums and existing_nums != incoming_nums:
            base_existing = re.sub(num_pattern, "<num>", existing_key)
            base_incoming = re.sub(num_pattern, "<num>", incoming_key)
            if base_existing == base_incoming:
                return True, "numeric_conflict"
        if any(word in incoming.split() for word in ("not", "never", "no")) and existing_key in incoming_key:
            return True, "keyword_negation"
        return False, None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

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
