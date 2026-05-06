"""Deterministic reference-anchor extraction from bounded working memory."""

from __future__ import annotations

from collections import Counter
from typing import Any
import re

from fullerene.context.models import ConversationContinuity, ReferenceAnchor

REFERENTIAL_TOKENS = {
    "one",
    "ones",
    "it",
    "that",
    "this",
    "those",
    "them",
    "they",
    "there",
    "again",
    "he",
    "she",
    "him",
    "her",
}

STOP_TERMS = {
    "the",
    "a",
    "an",
    "my",
    "your",
    "this",
    "that",
    "these",
    "those",
    "for",
    "with",
    "from",
    "about",
    "into",
    "onto",
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "did",
    "does",
    "can",
    "could",
    "would",
    "should",
    "to",
    "and",
    "or",
    "of",
    "in",
    "on",
}


def derive_reference_anchors(
    current_text: str,
    working_turns: list[Any],
    *,
    max_anchors: int = 5,
) -> ConversationContinuity:
    text = str(current_text or "")
    turns = list(working_turns or [])
    max_anchor_count = max(int(max_anchors or 5), 0)
    referential_hits = _referential_hits(text)
    candidate_pool = _candidate_phrases(turns)
    topic_terms, topic_hint = _topic_terms(turns, candidate_pool)

    anchors: list[ReferenceAnchor] = []
    unresolved: list[str] = []
    for hit in referential_hits:
        ranked = _rank_candidates_for_surface(hit["token"], candidate_pool, max_anchor_count)
        if not ranked:
            unresolved.append(hit["token"])
            continue
        for rank in ranked:
            anchors.append(
                ReferenceAnchor(
                    anchor_id=f"anchor-{hit['token']}-{rank['turn_id']}-{rank['phrase']}".lower().replace(" ", "-"),
                    surface_form=hit["token"],
                    referent_text=rank["phrase"],
                    referent_source_turn_id=rank["turn_id"],
                    referent_source_role=rank["role"],
                    confidence=rank["confidence"],
                    reason=rank["reason"],
                    current_message_fragment=hit["fragment"],
                    metadata={
                        "recency_rank": rank["recency_rank"],
                        "match_kind": rank["match_kind"],
                        "source": "working_memory",
                    },
                )
            )

    anchors = sorted(
        anchors,
        key=lambda item: (float(item.confidence), item.referent_source_turn_id or "", item.referent_text),
        reverse=True,
    )[:max_anchor_count]
    unresolved_unique = sorted(set(unresolved))
    continuity_confidence = _continuity_confidence(anchors, unresolved_unique, len(turns))

    return ConversationContinuity(
        current_topic_hint=topic_hint,
        topic_terms=topic_terms,
        reference_anchors=anchors,
        unresolved_references=unresolved_unique,
        continuity_confidence=continuity_confidence,
        working_memory_turn_count=len(turns),
        source="working_memory",
    )


def _referential_hits(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if not text.strip():
        return hits
    for match in re.finditer(r"\b(one|ones|it|that|this|those|them|they|there|again|he|she|him|her)\b", text, re.IGNORECASE):
        token = match.group(1).lower()
        left = max(match.start() - 18, 0)
        right = min(match.end() + 18, len(text))
        fragment = text[left:right].strip()
        hits.append({"token": token, "fragment": fragment})
    return hits


def _candidate_phrases(working_turns: list[Any]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for idx, turn in enumerate(working_turns):
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("content") or "").strip()
        if not text:
            continue
        metadata = turn.get("metadata", {})
        role = str((metadata.get("dialogue_role") if isinstance(metadata, dict) else turn.get("role")) or "").strip().lower()
        turn_id = str(turn.get("id") or turn.get("source_id") or f"turn-{idx + 1}")
        recency_rank = max(len(working_turns) - idx, 1)
        for phrase, kind, clarity in _extract_phrases(text):
            pool.append(
                {
                    "phrase": phrase,
                    "turn_id": turn_id,
                    "role": role or "unknown",
                    "recency_rank": recency_rank,
                    "match_kind": kind,
                    "clarity": clarity,
                }
            )
    return pool


def _extract_phrases(text: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for match in re.finditer(r"\"([^\"]{2,60})\"|'([^']{2,60})'", text):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            out.append((phrase, "quoted_phrase", 0.85))
    for match in re.finditer(r"\b(?:a|an|the|my|your|this|that)\s+([a-zA-Z][a-zA-Z0-9\-]*(?:\s+[a-zA-Z][a-zA-Z0-9\-]*){0,3})", text):
        phrase = (match.group(1) or "").strip()
        if phrase and len(phrase) > 1:
            out.append((phrase, "article_noun_phrase", 0.65))
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        phrase = (match.group(1) or "").strip()
        if phrase and phrase.lower() not in STOP_TERMS:
            out.append((phrase, "capitalized_span", 0.58))
    return _dedupe_phrase_rows(out)


def _dedupe_phrase_rows(rows: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    best: dict[str, tuple[str, str, float]] = {}
    for phrase, kind, score in rows:
        key = phrase.lower()
        prev = best.get(key)
        if prev is None or score > prev[2]:
            best[key] = (phrase, kind, score)
    return list(best.values())


def _rank_candidates_for_surface(
    surface: str,
    candidates: list[dict[str, Any]],
    max_count: int,
) -> list[dict[str, Any]]:
    if not candidates or max_count <= 0:
        return []
    phrase_counts = Counter(item["phrase"].lower() for item in candidates if item.get("phrase"))
    scored: list[dict[str, Any]] = []
    total = max(len(candidates), 1)
    for row in candidates:
        phrase = str(row["phrase"])
        repeated = phrase_counts[phrase.lower()] > 1
        repeated_bonus = 0.1 if repeated else 0.0
        recency = 1.0 - ((row["recency_rank"] - 1) / max(total, 1))
        recency_bonus = recency * 0.1
        base = float(row["clarity"])
        conf = max(0.0, min(base + repeated_bonus + recency_bonus, 0.9))
        reason = "quoted_term" if row["match_kind"] == "quoted_phrase" else (
            "repeated_recent_phrase" if repeated else "recent_noun_phrase"
        )
        if row["recency_rank"] > 6:
            conf = max(0.35, conf - 0.2)
            reason = "older_candidate"
        scored.append(
            {
                "phrase": phrase,
                "turn_id": row["turn_id"],
                "role": row["role"],
                "recency_rank": row["recency_rank"],
                "match_kind": row["match_kind"],
                "confidence": round(conf, 2),
                "reason": reason,
                "surface": surface,
            }
        )
    ranked = sorted(
        scored,
        key=lambda item: (-item["confidence"], item["recency_rank"], item["phrase"]),
    )
    unique: list[dict[str, Any]] = []
    seen = set()
    for row in ranked:
        key = (row["phrase"].lower(), row["turn_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= max_count:
            break
    return unique


def _topic_terms(
    turns: list[Any],
    candidate_pool: list[dict[str, Any]],
) -> tuple[list[str], str | None]:
    if not turns:
        return [], None
    role_terms: dict[str, set[str]] = {"user": set(), "assistant": set()}
    freq: Counter[str] = Counter()
    for row in candidate_pool:
        phrase = str(row.get("phrase") or "").strip().lower()
        if not phrase:
            continue
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", phrase) if w not in STOP_TERMS]
        for word in words[:2]:
            freq[word] += 1
            role = str(row.get("role") or "")
            if role in role_terms:
                role_terms[role].add(word)
    cross_role = [term for term, count in freq.most_common() if term in role_terms["user"] and term in role_terms["assistant"]]
    ordered = cross_role[:3]
    if not ordered:
        ordered = [term for term, _ in freq.most_common(3)]
    if not ordered:
        return [], None
    hint = f"recent discussion about {'/'.join(ordered[:2])}"
    return ordered, hint


def _continuity_confidence(
    anchors: list[ReferenceAnchor],
    unresolved: list[str],
    turn_count: int,
) -> float:
    if not turn_count:
        return 0.0
    if anchors:
        avg_anchor = sum(float(anchor.confidence) for anchor in anchors) / max(len(anchors), 1)
        penalty = min(len(unresolved) * 0.12, 0.35)
        return round(max(0.0, min(avg_anchor - penalty, 1.0)), 2)
    if unresolved:
        return 0.3
    return 0.45 if turn_count > 0 else 0.0
