"""Belief-graph community detection (World Model v2).

Default strategy: deterministic contradiction connected-components over high-weight
`contradicting` edges (Leiden seam reserved; not required for tests).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol, Sequence


def stable_cluster_id(member_ids: Sequence[str], *, cluster_type: str, strategy: str) -> str:
    payload = {
        "members": sorted({str(m) for m in member_ids if m}),
        "cluster_type": str(cluster_type),
        "strategy": str(strategy),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()[:32]


class BeliefCommunityDetector(Protocol):
    @property
    def strategy_name(self) -> str: ...

    def detect_contradiction_clusters(
        self,
        *,
        belief_ids: Sequence[str],
        contradicting_edges: Sequence[tuple[str, str, float]],
    ) -> list[frozenset[str]]:
        ...


@dataclass(slots=True)
class DeterministicContradictionComponentDetector:
    """Groups beliefs connected by contradicting edges above a minimum weight."""

    min_edge_weight: float = 0.25
    max_graph_edges: int = 2000
    strategy_name: str = "deterministic_contradiction_components_v0"

    def detect_contradiction_clusters(
        self,
        *,
        belief_ids: Sequence[str],
        contradicting_edges: Sequence[tuple[str, str, float]],
    ) -> list[frozenset[str]]:
        ids = sorted({str(bid) for bid in belief_ids if bid})
        if not ids:
            return []

        trimmed: list[tuple[str, str, float]] = []
        for a, b, w in contradicting_edges:
            if float(w) < self.min_edge_weight:
                continue
            x, y = (a, b) if a < b else (b, a)
            trimmed.append((x, y, float(w)))
        trimmed.sort(key=lambda row: (row[0], row[1], -row[2]))
        seen_e: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str, float]] = []
        for a, b, w in trimmed:
            if (a, b) in seen_e:
                continue
            seen_e.add((a, b))
            deduped.append((a, b, w))
            if len(deduped) >= self.max_graph_edges:
                break

        adj: dict[str, set[str]] = defaultdict(set)
        for a, b, _w in deduped:
            adj[a].add(b)
            adj[b].add(a)
        for bid in ids:
            adj.setdefault(bid, set())

        visited: set[str] = set()
        components: list[frozenset[str]] = []
        for start in sorted(adj.keys()):
            if start in visited:
                continue
            stack = [start]
            comp: set[str] = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in sorted(adj.get(node, ())):
                    if nb not in visited:
                        stack.append(nb)
            if len(comp) >= 2:
                components.append(frozenset(comp))

        components.sort(key=lambda fs: (min(fs), -len(fs), sorted(fs)))
        return components
