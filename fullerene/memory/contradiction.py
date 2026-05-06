"""Deterministic memory contradiction / refinement heuristics (no LLM)."""

from __future__ import annotations

import re
from typing import Any

from fullerene.memory.scoring import tokenize


class ContradictionStatus:
    NONE = "none"
    CONTRADICTED = "contradicted"
    REFINED = "refined"
    SUPERSEDED = "superseded"


_NEGATORS = re.compile(
    r"\b(no|not|never|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|cannot|can't)\b",
    re.IGNORECASE,
)


def extract_numeric_claims(text: str) -> dict[str, float]:
    """Simple key:value or 'temperature is 30' style numbers for conflict checks."""
    out: dict[str, float] = {}
    for m in re.finditer(
        r"([a-z_][a-z0-9_]*)\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        text.lower(),
    ):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


def simple_content_contradiction(
    a: str,
    b: str,
) -> tuple[bool, str | None]:
    """Negation flip or numeric key clash between similar token sets."""
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return False, None
    overlap = ta & tb
    if len(overlap) < 2:
        return False, None
    neg_a = bool(_NEGATORS.search(a))
    neg_b = bool(_NEGATORS.search(b))
    if neg_a != neg_b:
        return True, "negation_polarity"

    nums_a = extract_numeric_claims(a)
    nums_b = extract_numeric_claims(b)
    for k in set(nums_a) & set(nums_b):
        if abs(nums_a[k] - nums_b[k]) > 1e-6:
            return True, f"numeric_conflict:{k}"
    return False, None


def merge_contradiction_metadata(
    existing: dict[str, Any],
    *,
    status: str,
    peer_ids: list[str],
    score_delta: float,
    reason: str,
) -> dict[str, Any]:
    md = dict(existing)
    md["contradiction_status"] = status
    cur = list(md.get("contradicted_by_memory_ids") or [])
    for pid in peer_ids:
        if pid and pid not in cur:
            cur.append(pid)
    md["contradicted_by_memory_ids"] = cur[:20]
    md["contradiction_score"] = max(
        float(md.get("contradiction_score") or 0.0),
        float(score_delta),
    )
    reasons = list(md.get("contradiction_reasons") or [])
    if reason and reason not in reasons:
        reasons.append(reason)
    md["contradiction_reasons"] = reasons[:12]
    return md
