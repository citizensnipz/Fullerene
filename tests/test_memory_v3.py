"""Memory v3 — communities, retrieval bonuses, LPB, salience, contradictions (synthetic)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fullerene.memory.communities import MemoryCommunity
from fullerene.memory.community_detection import (
    DeterministicConnectedComponentsDetector,
    stable_community_id,
)
from fullerene.memory.contradiction import (
    ContradictionStatus,
    merge_contradiction_metadata,
    simple_content_contradiction,
)
from fullerene.memory.models import MemoryLayer, MemoryRecord, MemoryType
from fullerene.memory.store import SQLiteMemoryStore
from fullerene.memory import v3 as mem_v3
from fullerene.memory.edges import MemoryEdge, MemoryEdgeType
from fullerene.nexus.models import Event, EventType
from fullerene.nexus.runtime import NexusRuntime
from fullerene.facets.echo import EchoFacet
from fullerene.facets.memory import MemoryFacet
from fullerene.signals.latent_pressure.buffer import should_ingest_signal_on_tick


def _evt(txt: str, eid: str = "e1") -> Event:
    return Event(
        event_id=eid,
        event_type=EventType.USER_MESSAGE,
        content=txt,
        metadata={},
    )


class MemoryV3SchemaTests(unittest.TestCase):
    def test_community_round_trip(self) -> None:
        mc = MemoryCommunity(
            community_id="c1",
            label="books / reading_books",
            member_count=2,
            top_tags=["reading"],
            top_domains=["reading_books"],
        )
        back = MemoryCommunity.from_dict(mc.to_dict())
        self.assertEqual(back.community_id, "c1")
        self.assertEqual(back.label, mc.label)


class MemoryV3CommunityDetectionTests(unittest.TestCase):
    def test_connected_components_and_stable_id(self) -> None:
        det = DeterministicConnectedComponentsDetector(min_edge_weight=0.1)
        mems = ["a", "b", "c", "d"]
        edges = [
            ("a", "b", MemoryEdgeType.TAG_OVERLAP.value, 0.4),
            ("b", "c", MemoryEdgeType.KEYWORD_SIMILARITY.value, 0.5),
        ]
        comps = det.detect(
            memory_ids=mems,
            edges=edges,
            memory_tags={m: [] for m in mems},
            memory_domains={m: None for m in mems},
        )
        flat = {frozenset(x) for x in comps}
        self.assertIn(frozenset({"a", "b", "c"}), flat)
        cid = stable_community_id(["a", "b", "c"], ["t1"], strategy="deterministic_connected_components_v0")
        self.assertEqual(len(cid), 32)

    def test_weak_edges_do_not_connect(self) -> None:
        det = DeterministicConnectedComponentsDetector(min_edge_weight=0.5)
        mems = ["a", "b"]
        edges = [("a", "b", MemoryEdgeType.TAG_OVERLAP.value, 0.1)]
        comps = det.detect(
            memory_ids=mems,
            edges=edges,
            memory_tags={"a": [], "b": []},
            memory_domains={"a": None, "b": None},
        )
        sizes = sorted(len(c) for c in comps)
        self.assertEqual(sizes, [1, 1])


class MemoryV3StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.sqlite3"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_migration_preserves_rows(self) -> None:
        store = SQLiteMemoryStore(self.path)
        r = MemoryRecord(content="legacy", memory_type=MemoryType.EPISODIC)
        store.add_memory(r)
        store2 = SQLiteMemoryStore(self.path)
        got = store2.get_memory(r.id)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.content, "legacy")

    def test_rebuild_and_neighbors_bounded(self) -> None:
        store = SQLiteMemoryStore(self.path)
        ids = []
        for i in range(4):
            m = MemoryRecord(
                content=f"topic alpha chunk {i}",
                memory_type=MemoryType.EPISODIC,
                tags=["alpha"],
                role="fact",
                domain="project_software",
            )
            store.add_memory(m)
            ids.append(m.id)
        for a, b in zip(ids, ids[1:]):
            store.add_memory_edge(
                MemoryEdge(
                    source_memory_id=a,
                    target_memory_id=b,
                    edge_type=MemoryEdgeType.TAG_OVERLAP,
                    weight=0.6,
                )
            )
        n = store.rebuild_memory_communities(limit=500)
        self.assertGreaterEqual(n, 1)
        neigh = store.get_memory_neighbors(ids[0], depth=1, limit=10)
        self.assertIn(ids[1], neigh)
        self.assertLessEqual(len(neigh), 10)


class MemoryV3HybridTests(unittest.TestCase):
    def test_v3_bonuses_bounded(self) -> None:
        b = mem_v3.v3_retrieval_bonuses(
            community_activation=1.0,
            community_pressure=1.0,
            member_overlap_ratio=1.0,
            best_neighbor_weight_norm=1.0,
        )
        self.assertLessEqual(b["community_activation_bonus"], 0.10 + 1e-6)
        self.assertLessEqual(b["community_pressure_bonus"], 0.10 + 1e-6)
        self.assertLessEqual(b["direct_neighbor_bonus"], 0.05 + 1e-6)


class MemoryV3LPBTests(unittest.TestCase):
    def test_memory_cluster_suppressed_on_tick(self) -> None:
        from fullerene.nexus.models import NexusState

        sig = {
            "source": "memory",
            "entry_type": "memory_cluster_activation",
            "source_id": "x",
            "description": "test",
            "metadata": {},
        }
        tick = Event(
            event_id="t1",
            event_type=EventType.SYSTEM_TICK,
            content="",
            metadata={},
        )
        ok, reason = should_ingest_signal_on_tick(sig, tick, NexusState())
        self.assertFalse(ok)
        self.assertIn("memory_cluster", reason)


class MemoryV3ContradictionTests(unittest.TestCase):
    def test_negation_heuristic(self) -> None:
        ok, why = simple_content_contradiction(
            "the server is online",
            "the server is not online",
        )
        self.assertTrue(ok)
        self.assertEqual(why, "negation_polarity")

    def test_merge_metadata(self) -> None:
        md = merge_contradiction_metadata(
            {},
            status=ContradictionStatus.CONTRADICTED,
            peer_ids=["m2"],
            score_delta=0.3,
            reason="test",
        )
        self.assertEqual(md["contradiction_status"], "contradicted")


class MemoryV3IntegrationSmoke(unittest.TestCase):
    def test_nexus_memory_facet_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mem.sqlite3"
            store = SQLiteMemoryStore(p)
            facet = MemoryFacet(store, retrieve_limit=2)
            rt = NexusRuntime(facets=[facet, EchoFacet()])
            rec = rt.process_event(_evt("hello v3 synthetic test"))
            self.assertTrue(any(fr.facet_name == "memory" for fr in rec.facet_results))


if __name__ == "__main__":
    unittest.main()
