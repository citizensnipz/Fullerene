"""Deterministic belief rule evaluation (World Model v2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fullerene.world_model.models import Belief, BeliefType, normalize_statement, stable_belief_id


@dataclass(slots=True)
class BeliefRulePattern:
    """Single antecedent match constraint."""

    text_contains: tuple[str, ...] = ()
    min_confidence: float = 0.6
    belief_types: tuple[str, ...] | None = None  # subset of BeliefType values or None=all
    status_in: tuple[str, ...] = ("valid",)

    def matches(self, belief: Belief) -> bool:
        if belief.confidence < self.min_confidence:
            return False
        st = belief.status.value if hasattr(belief.status, "value") else str(belief.status)
        if self.status_in and st.lower() not in {x.lower() for x in self.status_in}:
            return False
        if self.belief_types is not None:
            bt = belief.belief_type.value if isinstance(belief.belief_type, BeliefType) else str(belief.belief_type)
            if bt.lower() not in {x.lower() for x in self.belief_types}:
                return False
        text = belief.claim.lower()
        nk = belief.normalized_key.lower()
        for frag in self.text_contains:
            f = frag.lower()
            if f not in text and f not in nk:
                return False
        return True


@dataclass(slots=True)
class RuleEvalConfig:
    max_rule_evaluations: int = 50
    max_inferred_beliefs_per_cycle: int = 5
    min_confidence_for_rule: float = 0.6


def _rule_effective_weight(rule_row: dict[str, Any]) -> float:
    cw = float(rule_row.get("confidence_weight") or 0.5)
    hv = float(rule_row.get("historical_validity") or 0.5)
    return max(0.0, min(1.0, cw * hv))


def belief_matches_builtin_negative(
    belief_a: Belief,
    belief_b: Belief,
    *,
    subject_fn: Callable[[Belief], str] | None = None,
) -> bool:
    """Generic complementary capability / negation heuristic (synthetic-safe)."""

    def _subject(b: Belief) -> str:
        if subject_fn:
            return subject_fn(b)
        parts = (b.normalized_key or "").split()
        sig = [p for p in parts if len(p) > 2][:4]
        return " ".join(sig)

    sa, sb = _subject(belief_a), _subject(belief_b)
    if not sa or sa != sb:
        return False

    def _neg_polarity(text: str) -> bool:
        t = text.lower()
        return any(x in t for x in (" not ", "no ", "cannot ", "can't ", "never "))

    pa = _neg_polarity(belief_a.claim)
    pb = _neg_polarity(belief_b.claim)
    return pa != pb


def increment_rule_support(rule_id: str, store: Any) -> None:
    if hasattr(store, "increment_belief_rule_support"):
        store.increment_belief_rule_support(rule_id)


def increment_rule_failure(rule_id: str, store: Any) -> None:
    if hasattr(store, "increment_belief_rule_failure"):
        store.increment_belief_rule_failure(rule_id)


def evaluate_enabled_rules_bounded(
    store: Any,
    candidate_beliefs: list[Belief],
    *,
    config: RuleEvalConfig | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Return effect summary: inferred belief ids, contradictions strengthened, evaluations used."""
    cfg = config or RuleEvalConfig()
    out: dict[str, Any] = {
        "evaluations_run": 0,
        "inferred_belief_ids": [],
        "applied_rule_ids": [],
        "negative_hits": [],
    }
    if not hasattr(store, "list_enabled_belief_rules"):
        return out

    rules = store.list_enabled_belief_rules(limit=cfg.max_rule_evaluations)
    inferred_count = 0
    evaluations = 0

    for rule in rules:
        if evaluations >= cfg.max_rule_evaluations:
            break
        evaluations += 1
        rule_id = str(rule.get("rule_id") or "")
        enabled = bool(rule.get("enabled", True))
        if not enabled or not rule_id:
            continue
        rtype = str(rule.get("rule_type") or "").strip().lower()
        ants = rule.get("antecedent_patterns") or []
        cons = rule.get("consequent_pattern") or ""

        ew = _rule_effective_weight(rule)
        if ew <= 0.01:
            continue

        if rtype == "positive":
            patterns: list[BeliefRulePattern] = []
            for raw in ants:
                if not isinstance(raw, dict):
                    continue
                fragments = ()
                tc = raw.get("text_contains")
                if isinstance(tc, str):
                    fragments = (tc,)
                elif isinstance(tc, list):
                    fragments = tuple(str(x) for x in tc)
                mc = float(raw.get("min_confidence") or cfg.min_confidence_for_rule)
                bt = raw.get("belief_types")
                btypes = tuple(str(x) for x in bt) if isinstance(bt, list) else None
                patterns.append(
                    BeliefRulePattern(
                        text_contains=fragments,
                        min_confidence=mc,
                        belief_types=btypes,
                        status_in=tuple(str(x) for x in raw["status_in"])
                        if isinstance(raw.get("status_in"), list)
                        else ("valid",),
                    )
                )
            if len(patterns) < 2:
                continue
            bound: list[Belief] = []
            consumed: set[str] = set()
            ok = True
            for patt in patterns:
                found_b: Belief | None = None
                for belief in candidate_beliefs:
                    if belief.id in consumed:
                        continue
                    if patt.matches(belief):
                        found_b = belief
                        break
                if found_b is None:
                    ok = False
                    break
                bound.append(found_b)
                consumed.add(found_b.id)
            if not ok or inferred_count >= cfg.max_inferred_beliefs_per_cycle:
                continue

            if len(bound) < 2 or bound[0].id == bound[1].id:
                continue
            bid0, bid1 = bound[0], bound[1]

            cons_text = str(cons).strip()
            if not cons_text:
                continue
            nk_infer = normalize_statement(cons_text)
            existing = (
                store.get_belief_by_normalized_key(nk_infer) if hasattr(store, "get_belief_by_normalized_key") else None
            )
            if existing:
                increment_rule_failure(rule_id, store)
                continue

            confidence = round(max(0.35, min(0.95, 0.45 + ew * 0.4)), 4)
            new_belief = Belief(
                id=stable_belief_id(nk_infer + "|inferred|" + rule_id),
                claim=cons_text,
                confidence=confidence,
                sources=[bid0.id, bid1.id, rule_id],
                normalized_key=nk_infer,
                belief_type=BeliefType.FACT,
                metadata={
                    "inferred": True,
                    "inference_rule_id": rule_id,
                    "source_belief_ids": sorted([bid0.id, bid1.id]),
                    "world_model_rule_version": "v2",
                },
            )
            store.add_belief(new_belief)
            if hasattr(store, "add_belief_edge"):
                for src_id in (bid0.id, bid1.id):
                    store.add_belief_edge(
                        source_belief_id=src_id,
                        target_belief_id=new_belief.id,
                        edge_type="inferred_from",
                        weight=round(ew * 0.5 + 0.2, 4),
                        metadata={"rule_id": rule_id},
                    )
            inferred_count += 1
            out["inferred_belief_ids"].append(new_belief.id)
            out["applied_rule_ids"].append(rule_id)
            increment_rule_support(rule_id, store)

        elif rtype == "negative":
            # Pair scan for complementary claims
            matched = False
            for i, ba in enumerate(candidate_beliefs):
                if ba.confidence < cfg.min_confidence_for_rule:
                    continue
                for j in range(i + 1, len(candidate_beliefs)):
                    bb = candidate_beliefs[j]
                    if bb.confidence < cfg.min_confidence_for_rule:
                        continue
                    if belief_matches_builtin_negative(ba, bb):
                        if hasattr(store, "strengthen_belief_edge_pair"):
                            store.strengthen_belief_edge_pair(
                                ba.id,
                                bb.id,
                                "contradicting",
                                amount=0.15 * ew,
                                provenance={"rule_ids": [rule_id]},
                            )
                        matched = True
                        out["negative_hits"].append({"belief_a": ba.id, "belief_b": bb.id, "rule_id": rule_id})
                        increment_rule_support(rule_id, store)
                        break
                if matched:
                    break
            if not matched:
                increment_rule_failure(rule_id, store)

    out["evaluations_run"] = evaluations
    return out


def seed_builtin_belief_rules(store: Any) -> int:
    """Insert minimal generic rules if table empty. Returns rules added."""
    if not hasattr(store, "count_belief_rules") or not hasattr(store, "add_belief_rule"):
        return 0
    if store.count_belief_rules() > 0:
        return 0
    added = 0
    # Positive: two distinct token markers (tests use token_a / token_b)
    store.add_belief_rule(
        rule_id="builtin_positive_synergy_v0",
        rule_type="positive",
        antecedent_patterns=[
            {"text_contains": ["token_a"], "min_confidence": 0.6, "status_in": ["valid"]},
            {"text_contains": ["token_b"], "min_confidence": 0.6, "status_in": ["valid"]},
        ],
        consequent_pattern="combined token_a token_b holds by rule synergy",
        confidence_weight=0.7,
        historical_validity=1.0,
        enabled=True,
        metadata={"builtin": True, "category": "synthetic_compose"},
    )
    added += 1
    # Negative: capability-style polarity (subject line match via shared normalized prefix)
    store.add_belief_rule(
        rule_id="builtin_negative_capability_polarity_v0",
        rule_type="negative",
        antecedent_patterns=[],
        consequent_pattern="",
        confidence_weight=0.8,
        historical_validity=1.0,
        enabled=True,
        metadata={"builtin": True, "category": "complementary_polarity"},
    )
    added += 1
    return added
