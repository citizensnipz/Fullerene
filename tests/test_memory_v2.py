"""Tests for Fullerene Memory v2 (role/domain, hybrid retrieval, edges)."""

from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fullerene.cli import _build_model_prompt, main as cli_main
from fullerene.context import ContextAssemblyConfig, ContextItemType
from fullerene.facets import ContextFacet, EchoFacet, MemoryFacet
from fullerene.memory import (
    DeterministicHashEmbeddingProvider,
    MemoryEdgeType,
    MemoryRecord,
    MemoryRole,
    MemoryType,
    QueryIntent,
    SQLiteMemoryStore,
    classify_memory_role,
    classify_query_intent,
    cosine_similarity,
    explain_hybrid_score,
    infer_domain,
    infer_tags,
)
from fullerene.memory.models import utcnow
from fullerene.nexus import Event, EventType, NexusRuntime, NexusState
from fullerene.state import FileStateStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    return workspace_state_root() / f".test-memory-v2-{uuid4().hex}"


# ---------------------------------------------------------------------------
# Section A: deterministic role / tag / domain classification
# ---------------------------------------------------------------------------


class RoleClassificationTests(unittest.TestCase):
    def test_preference_phrasing_classifies_as_preference(self) -> None:
        self.assertEqual(
            classify_memory_role("I like to read sci-fi novels and non-fiction autobiographies"),
            MemoryRole.PREFERENCE,
        )
        self.assertEqual(
            classify_memory_role("I really enjoy long walks at the beach"),
            MemoryRole.PREFERENCE,
        )
        self.assertEqual(
            classify_memory_role("My favorite books are biographies"),
            MemoryRole.PREFERENCE,
        )

    def test_question_phrasing_classifies_as_question(self) -> None:
        self.assertEqual(
            classify_memory_role("What kind of book should I read next?"),
            MemoryRole.QUESTION,
        )
        self.assertEqual(
            classify_memory_role("How do I configure this?"),
            MemoryRole.QUESTION,
        )

    def test_task_phrasing_classifies_as_task(self) -> None:
        self.assertEqual(
            classify_memory_role("I need to finish Fullerene"),
            MemoryRole.TASK,
        )
        self.assertEqual(
            classify_memory_role("I should remember to ship the release"),
            MemoryRole.TASK,
        )

    def test_feedback_phrasing_classifies_as_feedback(self) -> None:
        self.assertEqual(classify_memory_role("that worked"), MemoryRole.FEEDBACK)
        self.assertEqual(
            classify_memory_role("that's wrong"),
            MemoryRole.FEEDBACK,
        )

    def test_outcome_phrasing_classifies_as_outcome(self) -> None:
        self.assertEqual(
            classify_memory_role("the deploy succeeded yesterday"),
            MemoryRole.OUTCOME,
        )
        self.assertEqual(
            classify_memory_role("the build failed earlier"),
            MemoryRole.OUTCOME,
        )

    def test_factual_statement_falls_back_to_fact(self) -> None:
        self.assertEqual(
            classify_memory_role("SQLite is the canonical memory store"),
            MemoryRole.FACT,
        )

    def test_empty_content_is_unknown(self) -> None:
        self.assertEqual(classify_memory_role(""), MemoryRole.UNKNOWN)
        self.assertEqual(classify_memory_role("   "), MemoryRole.UNKNOWN)


class DomainInferenceTests(unittest.TestCase):
    def test_preference_about_books_infers_reading_books_domain(self) -> None:
        domain = infer_domain(
            "I like to read sci-fi novels and non-fiction autobiographies",
        )
        self.assertEqual(domain, "reading_books")

    def test_question_about_books_infers_reading_books_domain(self) -> None:
        domain = infer_domain("What kind of book should I read next?")
        self.assertEqual(domain, "reading_books")

    def test_outdoors_water_domain_for_scuba(self) -> None:
        domain = infer_domain("we should plan a scuba diving trip soon")
        self.assertEqual(domain, "outdoors_water")

    def test_project_software_domain_for_fullerene(self) -> None:
        domain = infer_domain("I need to finish Fullerene")
        self.assertEqual(domain, "project_software")

    def test_empty_content_returns_none(self) -> None:
        self.assertIsNone(infer_domain(""))

    def test_unrelated_topic_returns_none(self) -> None:
        self.assertIsNone(infer_domain("the weather looks great today"))


class TagDomainTogetherTests(unittest.TestCase):
    def test_book_preference_event_extracts_expected_tags_and_domain(self) -> None:
        content = "I like to read sci-fi novels and non-fiction autobiographies"
        tags = infer_tags(content)
        self.assertIn("read", tags)
        self.assertIn("book", tags)
        self.assertIn("sci-fi", tags)
        self.assertEqual(infer_domain(content, tags), "reading_books")
        self.assertEqual(classify_memory_role(content), MemoryRole.PREFERENCE)


class QueryIntentTests(unittest.TestCase):
    def test_recommendation_intent(self) -> None:
        self.assertEqual(
            classify_query_intent("What kind of book should I read next?"),
            QueryIntent.RECOMMENDATION,
        )
        self.assertEqual(
            classify_query_intent("Recommend a movie for tonight"),
            QueryIntent.RECOMMENDATION,
        )

    def test_planning_intent(self) -> None:
        self.assertEqual(
            classify_query_intent("What are the next steps?"),
            QueryIntent.PLANNING,
        )

    def test_factual_intent(self) -> None:
        self.assertEqual(
            classify_query_intent("What is Fullerene?"),
            QueryIntent.FACTUAL,
        )

    def test_unknown_intent(self) -> None:
        self.assertEqual(
            classify_query_intent("hello there"),
            QueryIntent.UNKNOWN,
        )


# ---------------------------------------------------------------------------
# Section B: retrieval grounding (preference outranks prior questions)
# ---------------------------------------------------------------------------


class HybridRetrievalGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.store = SQLiteMemoryStore(self.root / "memory.sqlite3")

    def _store(self, content: str, *, created_at_offset: timedelta = timedelta()) -> MemoryRecord:
        from fullerene.memory.inference import infer_tags as _infer_tags

        tags = _infer_tags(content)
        domain = infer_domain(content, tags)
        record = MemoryRecord(
            created_at=utcnow() - created_at_offset,
            memory_type=MemoryType.EPISODIC,
            content=content,
            salience=0.6,
            confidence=1.0,
            tags=tags,
            role=classify_memory_role(content).value,
            domain=domain,
        )
        self.store.add_memory(record)
        return record

    def test_preference_outranks_prior_question_for_recommendation_query(self) -> None:
        preference = self._store(
            "I like to read sci-fi novels and non-fiction autobiographies",
            created_at_offset=timedelta(days=2),
        )
        prior_question = self._store(
            "What kind of book should I read next?",
            created_at_offset=timedelta(days=1),
        )
        unrelated_question = self._store(
            "What should I do next?",
            created_at_offset=timedelta(hours=1),
        )

        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="What kind of book should I read next?",
        )
        ranked = self.store.hybrid_retrieve_relevant(event, limit=3)

        self.assertEqual([record.id for record, _ in ranked][0], preference.id)
        ids = [record.id for record, _ in ranked]
        self.assertIn(prior_question.id, ids)
        self.assertNotEqual(ids[0], prior_question.id)
        del unrelated_question  # silence unused var

        breakdown_by_id = {record.id: breakdown for record, breakdown in ranked}
        preference_breakdown = breakdown_by_id[preference.id]
        prior_breakdown = breakdown_by_id[prior_question.id]
        self.assertEqual(preference_breakdown["memory_role"], MemoryRole.PREFERENCE.value)
        self.assertEqual(preference_breakdown["query_intent"], QueryIntent.RECOMMENDATION.value)
        self.assertGreater(preference_breakdown["role_bonus_raw"], 0.0)
        self.assertGreater(preference_breakdown["domain_match"], 0.0)
        self.assertGreater(prior_breakdown["role_penalty_raw"], 0.0)

    def test_negative_control_without_preference_signals_insufficient_support(self) -> None:
        prior = self._store(
            "What kind of book should I read next?",
            created_at_offset=timedelta(days=1),
        )
        event = Event(
            event_type=EventType.USER_MESSAGE,
            content="What kind of book should I read next?",
        )
        ranked = self.store.hybrid_retrieve_relevant(event, limit=3)

        breakdowns = {record.id: breakdown for record, breakdown in ranked}
        self.assertIn(prior.id, breakdowns)
        prior_breakdown = breakdowns[prior.id]
        # Without a preference memory, the prior question should still get
        # a role penalty for a recommendation query.
        self.assertGreater(prior_breakdown["role_penalty_raw"], 0.0)
        self.assertEqual(prior_breakdown["memory_role"], MemoryRole.QUESTION.value)


# ---------------------------------------------------------------------------
# Section C: context integration
# ---------------------------------------------------------------------------


class ContextIntegrationTests(unittest.TestCase):
    def test_context_includes_preference_memory_for_recommendation_query(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        preference = MemoryRecord(
            created_at=utcnow() - timedelta(days=2),
            memory_type=MemoryType.EPISODIC,
            content="I like to read sci-fi novels and non-fiction autobiographies",
            salience=0.7,
            confidence=1.0,
            tags=infer_tags("I like to read sci-fi novels and non-fiction autobiographies"),
            role=MemoryRole.PREFERENCE.value,
            domain="reading_books",
        )
        prior_question = MemoryRecord(
            created_at=utcnow() - timedelta(days=1),
            memory_type=MemoryType.EPISODIC,
            content="What kind of book should I read next?",
            salience=0.4,
            confidence=1.0,
            tags=infer_tags("What kind of book should I read next?"),
            role=MemoryRole.QUESTION.value,
            domain="reading_books",
        )
        unrelated_question = MemoryRecord(
            created_at=utcnow() - timedelta(hours=1),
            memory_type=MemoryType.EPISODIC,
            content="What should I do next?",
            salience=0.4,
            confidence=1.0,
            tags=[],
            role=MemoryRole.QUESTION.value,
            domain=None,
        )
        for record in (preference, prior_question, unrelated_question):
            memory_store.add_memory(record)

        facet = ContextFacet(
            memory_store,
            config=ContextAssemblyConfig(
                max_memories=3,
                include_policy_summary=False,
                include_signal_summaries=False,
            ),
        )
        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What kind of book should I read next?",
            ),
            NexusState(),
        )

        memory_items = [
            item
            for item in result.metadata["context_window"]["items"]
            if item["item_type"] == "memory"
        ]
        memory_ids = [item["id"] for item in memory_items]
        self.assertIn(preference.id, memory_ids)
        self.assertEqual(memory_ids[0], preference.id)
        self.assertIn(MemoryRole.PREFERENCE.value, result.metadata["included_memory_roles"])
        self.assertIn("reading_books", result.metadata["included_memory_domains"])
        self.assertEqual(result.metadata["query_intent"], QueryIntent.RECOMMENDATION.value)
        self.assertEqual(result.metadata["retrieval_strategy"], "hybrid_v2_deterministic")


# ---------------------------------------------------------------------------
# Section D: model prompt grounding
# ---------------------------------------------------------------------------


class ModelPromptGroundingTests(unittest.TestCase):
    def test_prompt_includes_preference_memory_as_grounding(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")

        preference = MemoryRecord(
            created_at=utcnow() - timedelta(days=2),
            memory_type=MemoryType.EPISODIC,
            content="I like to read sci-fi novels and non-fiction autobiographies",
            salience=0.7,
            confidence=1.0,
            tags=infer_tags("I like to read sci-fi novels and non-fiction autobiographies"),
            role=MemoryRole.PREFERENCE.value,
            domain="reading_books",
        )
        memory_store.add_memory(preference)

        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    memory_store,
                    config=ContextAssemblyConfig(
                        max_memories=3,
                        include_policy_summary=False,
                        include_signal_summaries=False,
                    ),
                ),
                MemoryFacet(memory_store, retrieve_limit=3, working_limit=3),
                EchoFacet(),
            ],
            store=state_store,
        )
        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What kind of book should I read next?",
            )
        )

        prompt = _build_model_prompt(
            record,
            {
                "query_intent": "recommendation",
                "response_template": "grounded_response_available",
            },
        )

        self.assertIn("Current working context:", prompt)
        self.assertIn("- current event: What kind of book should I read next?", prompt)
        self.assertIn("sci-fi novels and non-fiction autobiographies", prompt)
        self.assertIn("role=preference", prompt)
        self.assertIn("domain=reading_books", prompt)
        self.assertNotIn("\"context_window\"", prompt)


# ---------------------------------------------------------------------------
# Section E: edge creation
# ---------------------------------------------------------------------------


class WriteTimeEdgeTests(unittest.TestCase):
    def test_storing_related_memories_creates_role_related_or_same_domain_edge(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        facet = MemoryFacet(store, retrieve_limit=3, working_limit=3)

        first = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="I like to read sci-fi novels and non-fiction autobiographies",
            ),
            NexusState(),
        )
        second = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What kind of book should I read next?",
            ),
            NexusState(),
        )
        del first  # silence unused var warning

        new_memory_id = second.state_updates["last_stored_memory_id"]
        self.assertIsNotNone(new_memory_id)

        edges = store.list_memory_edges(memory_id=new_memory_id, limit=20)
        self.assertGreater(len(edges), 0)
        edge_types = {edge.edge_type for edge in edges}
        self.assertTrue(
            MemoryEdgeType.SAME_DOMAIN in edge_types
            or MemoryEdgeType.ROLE_RELATED in edge_types
        )

    def test_edge_creation_is_bounded_by_recent_window(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        facet = MemoryFacet(store)

        for index in range(40):
            facet.process(
                Event(
                    event_type=EventType.USER_MESSAGE,
                    content=f"unrelated note {index}",
                ),
                NexusState(),
            )

        last_event = Event(
            event_type=EventType.USER_MESSAGE,
            content="another unrelated note finale",
        )
        result = facet.process(last_event, NexusState())
        new_memory_id = result.state_updates["last_stored_memory_id"]
        # Recent + high-salience + same-domain caps each at 20; even if the
        # store had hundreds of records we should not produce more edges than
        # the bounded candidate sets allow. Each candidate yields at most ~6
        # edge types, so the upper bound is comfortably under 200.
        edges = store.list_memory_edges(memory_id=new_memory_id, limit=500)
        self.assertLess(len(edges), 200)


# ---------------------------------------------------------------------------
# Section F: fallback / embedding path coverage
# ---------------------------------------------------------------------------


class FallbackAndEmbeddingTests(unittest.TestCase):
    def test_no_embedding_provider_uses_deterministic_fallback(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        facet = MemoryFacet(store, retrieve_limit=3, working_limit=3)

        facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="I like to read sci-fi novels and non-fiction autobiographies",
            ),
            NexusState(),
        )
        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What kind of book should I read next?",
            ),
            NexusState(),
        )

        self.assertEqual(
            result.metadata["retrieval_strategy"],
            "hybrid_v2_deterministic",
        )
        self.assertIsNone(result.state_updates["last_stored_embedding_status"])

    def test_deterministic_embedding_provider_stores_vectors_and_appears_in_breakdown(
        self,
    ) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        provider = DeterministicHashEmbeddingProvider()
        facet = MemoryFacet(
            store,
            retrieve_limit=3,
            working_limit=3,
            embedding_provider=provider,
        )

        first_result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="I like to read sci-fi novels and non-fiction autobiographies",
            ),
            NexusState(),
        )
        self.assertEqual(first_result.state_updates["last_stored_embedding_status"], "stored")

        stored_id = first_result.state_updates["last_stored_memory_id"]
        vector, model_name = store.get_memory_embedding(stored_id)
        self.assertIsNotNone(vector)
        self.assertEqual(model_name, provider.name)
        self.assertGreater(len(vector or []), 0)

        result = facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="What kind of book should I read next?",
            ),
            NexusState(),
        )
        relevant = result.metadata["relevant_memories"]
        self.assertGreater(len(relevant), 0)
        breakdown = relevant[0]["score_breakdown"]
        self.assertIn("semantic_similarity", breakdown)
        self.assertEqual(
            result.metadata["retrieval_strategy"],
            "hybrid_v2_with_embeddings",
        )

    def test_cosine_similarity_handles_mismatched_inputs(self) -> None:
        self.assertEqual(cosine_similarity([], [1.0, 2.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)


# ---------------------------------------------------------------------------
# Section G: CLI smoke
# ---------------------------------------------------------------------------


class CLIMemoryV2SmokeTests(unittest.TestCase):
    def test_cli_memory_embeddings_flag_enables_embedding_index(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--memory-embeddings",
                    "--json",
                    "--content",
                    "I like to read sci-fi novels and non-fiction autobiographies",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        memory_result = next(
            r for r in payload["facet_results"] if r["facet_name"] == "memory"
        )
        self.assertEqual(
            memory_result["state_updates"]["last_stored_embedding_status"],
            "stored",
        )

    def test_cli_full_run_falls_back_when_embeddings_disabled(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--context",
                    "--json",
                    "--content",
                    "I like to read sci-fi novels and non-fiction autobiographies",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        memory_result = next(
            r for r in payload["facet_results"] if r["facet_name"] == "memory"
        )
        self.assertIsNone(
            memory_result["state_updates"]["last_stored_embedding_status"]
        )


if __name__ == "__main__":
    unittest.main()
