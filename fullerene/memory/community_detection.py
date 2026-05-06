"""Pluggable community detection for Memory v3.

Default: deterministic_connected_components_v0 (not Leiden).
Future: leiden when an optional backend is wired.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Protocol, Sequence


class CommunityDetector(Protocol):
    """Detect memory communities from a bounded weighted edge list."""

    @property
    def strategy_name(self) -> str: ...

    def detect(
        self,
        *,
        memory_ids: Sequence[str],
        edges: Sequence[tuple[str, str, str, float]],
        memory_tags: dict[str, list[str]],
        memory_domains: dict[str, str | None],
    ) -> list[frozenset[str]]:
        """Return disjoint member sets (each frozenset is one community)."""


@dataclass(slots=True)
class DeterministicConnectedComponentsDetector:
    """Split memories into connected components over edges >= min_weight.

    Very large components may be partitioned by dominant domain when simple.
    """

    min_edge_weight: float = 0.15
    max_graph_edges: int = 8000
    max_community_members: int = 40
    strategy_name: str = "deterministic_connected_components_v0"

    def detect(
        self,
        *,
        memory_ids: Sequence[str],
        edges: Sequence[tuple[str, str, str, float]],
        memory_tags: dict[str, list[str]],
        memory_domains: dict[str, str | None],
    ) -> list[frozenset[str]]:
        """Edges: (src, tgt, edge_type, weight) undirected."""

        ids = sorted({str(mid) for mid in memory_ids if mid})
        if not ids:
            return []

        # Build adjacency for edges above threshold (lexicographic tie-break stable)
        trimmed: list[tuple[str, str, float]] = []
        for src, tgt, _etype, w in edges:
            if float(w) < self.min_edge_weight:
                continue
            a, b = (src, tgt) if src < tgt else (tgt, src)
            trimmed.append((a, b, float(w)))
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
        for mid in ids:
            adj.setdefault(mid, set())

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
            if not comp:
                continue
            # Split oversized by dominant domain
            if len(comp) > self.max_community_members:
                subparts = self._split_by_domain(comp, memory_domains)
                for part in subparts:
                    if part:
                        components.append(frozenset(part))
            else:
                components.append(frozenset(comp))

        orphans = set(ids) - visited
        for mid in sorted(orphans):
            components.append(frozenset({mid}))

        # Deterministic sort: by min id, then size
        components.sort(key=lambda fs: (min(fs), -len(fs), sorted(fs)))
        return components

    def _split_by_domain(
        self,
        members: set[str],
        memory_domains: dict[str, str | None],
    ) -> list[set[str]]:
        buckets: dict[str, set[str]] = defaultdict(set)
        for mid in members:
            dom = memory_domains.get(mid) or ""
            dom = str(dom).strip().lower() or "__none__"
            buckets[dom].add(mid)
        parts = list(buckets.values())
        parts.sort(key=lambda s: (-len(s), min(s)))
        out: list[set[str]] = []
        for p in parts:
            if len(p) <= self.max_community_members:
                out.append(set(p))
            else:
                # last resort: chunk by sorted id
                ordered = sorted(p)
                chunk: list[str] = []
                for mid in ordered:
                    chunk.append(mid)
                    if len(chunk) >= self.max_community_members:
                        out.append(set(chunk))
                        chunk = []
                if chunk:
                    out.append(set(chunk))
        return out


def stable_community_id(
    members: Sequence[str],
    top_tags: Sequence[str],
    *,
    strategy: str,
) -> str:
    """Deterministic id from sorted members + tags + strategy."""
    payload = {
        "members": sorted({str(m) for m in members if m}),
        "tags": sorted({str(t).lower() for t in top_tags if str(t).strip()}),
        "strategy": strategy,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()[:32]


def label_from_tags_domains(
    top_tags: Sequence[str],
    top_domains: Sequence[str],
) -> str:
    tags = [t for t in top_tags if t][:3]
    doms = [d for d in top_domains if d][:2]
    parts = tags + doms
    if not parts:
        return "concern_area"
    return " / ".join(parts)


def aggregate_top_tags_domains_roles(
    members: Sequence[str],
    tag_fn: Any,
    domain_fn: Any,
    role_fn: Any,
) -> tuple[list[str], list[str], list[str]]:
    """tag_fn(mid) -> list[str], domain_fn(mid) -> str|None, role_fn(mid)->str."""
    tag_counts: Counter[str] = Counter()
    dom_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for mid in members:
        for t in tag_fn(mid) or []:
            tag_counts[str(t).lower()] += 1
        d = domain_fn(mid)
        if d:
            dom_counts[str(d).lower()] += 1
        r = role_fn(mid)
        if r:
            role_counts[str(r).lower()] += 1
    top_tags = [t for t, _ in tag_counts.most_common(6)]
    top_doms = [t for t, _ in dom_counts.most_common(3)]
    top_roles = [t for t, _ in role_counts.most_common(3)]
    return top_tags, top_doms, top_roles


class LeidenCommunityDetector:
    """Placeholder seam for future Leiden /igraph wiring (not implemented)."""

    strategy_name: str = "leiden"

    def detect(
        self,
        *,
        memory_ids: Sequence[str],
        edges: Sequence[tuple[str, str, str, float]],
        memory_tags: dict[str, list[str]],
        memory_domains: dict[str, str | None],
    ) -> list[frozenset[str]]:
        raise NotImplementedError("Leiden backend not enabled; use fallback detector.")
