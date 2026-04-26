from __future__ import annotations

import io
import json
import shutil
import sqlite3
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.scratch import scratch_root
from fullerene.facets import (
    BehaviorFacet,
    EchoFacet,
    GoalsFacet,
    MemoryFacet,
    PolicyFacet,
    WorldModelFacet,
)
from fullerene.goals import Goal, SQLiteGoalStore
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import DecisionAction, Event, EventType, NexusRuntime, NexusState
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicySource,
    PolicyStatus,
    PolicyTargetType,
    SQLitePolicyStore,
)
from fullerene.state import FileStateStore, InMemoryStateStore
from fullerene.world_model import Belief, SQLiteWorldModelStore


def make_tempdir_path() -> Path:
    return scratch_root() / f".test-policy-{uuid4().hex}"


class PolicyRuleModelTests(unittest.TestCase):
    def test_policy_rule_round_trips_through_dict(self) -> None:
        rule = PolicyRule(
            id="policy-1",
            name="Allow internal writes",
            description="Allow internal state CRUD in the configured sandbox.",
            rule_type=PolicyRuleType.ALLOW,
            target_type=PolicyTargetType.INTERNAL_STATE,
            target="state-dir",
            conditions={"within_state_dir": True},
            priority=9.5,
            enabled=True,
            source=PolicySource.SYSTEM,
            metadata={"origin": "bootstrap"},
        )

        round_tripped = PolicyRule.from_dict(rule.to_dict())

        self.assertEqual(round_tripped, rule)


class SQLitePolicyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.db_path = self.root / "policy.sqlite3"
        self.store = SQLitePolicyStore(self.db_path)

    def test_initializes_schema(self) -> None:
        self.assertTrue(self.db_path.exists())

        with sqlite3.connect(self.db_path) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

        self.assertIn("policies", table_names)

    def test_add_get_list_update_and_delete_policy(self) -> None:
        low = PolicyRule(
            id="policy-low",
            name="Low priority allow",
            description="Allow internal writes.",
            rule_type=PolicyRuleType.ALLOW,
            target_type=PolicyTargetType.INTERNAL_STATE,
            target="state-dir",
            priority=1.0,
        )
        high = PolicyRule(
            id="policy-high",
            name="High priority approval",
            description="Require approval before shell commands.",
            rule_type=PolicyRuleType.REQUIRE_APPROVAL,
            target_type=PolicyTargetType.SHELL,
            target="*",
            priority=9.0,
        )

        self.store.add_policy(low)
        self.store.add_policy(high)

        fetched = self.store.get_policy("policy-high")
        listed = self.store.list_policies(limit=5)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "policy-high")
        self.assertEqual([rule.id for rule in listed], ["policy-high", "policy-low"])

        high.enabled = False
        high.priority = 3.0
        self.store.update_policy(high)
        updated = self.store.get_policy("policy-high")

        self.assertIsNotNone(updated)
        self.assertFalse(updated.enabled)
        self.assertEqual(updated.priority, 3.0)

        self.store.delete_policy("policy-low")
        self.assertIsNone(self.store.get_policy("policy-low"))

    def test_list_enabled_policies_excludes_disabled_rules(self) -> None:
        enabled = PolicyRule(
            id="policy-enabled",
            name="Enabled policy",
            description="Enabled policy row.",
            rule_type=PolicyRuleType.ALLOW,
            target_type=PolicyTargetType.INTERNAL_STATE,
            target="state-dir",
            enabled=True,
        )
        disabled = PolicyRule(
            id="policy-disabled",
            name="Disabled policy",
            description="Disabled policy row.",
            rule_type=PolicyRuleType.DENY,
            target_type=PolicyTargetType.SHELL,
            target="*",
            enabled=False,
        )

        self.store.add_policy(enabled)
        self.store.add_policy(disabled)

        enabled_rules = self.store.list_enabled_policies()

        self.assertEqual([rule.id for rule in enabled_rules], ["policy-enabled"])


class PolicyFacetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.store = SQLitePolicyStore(self.root / "policy.sqlite3")
        self.facet = PolicyFacet(self.store, state_dir=self.root)

    def test_returns_no_match_when_no_policies_or_target_apply(self) -> None:
        result = self.facet.process(
            Event(event_type=EventType.USER_MESSAGE, content="hello policy"),
            NexusState(),
        )

        self.assertEqual(result.metadata["policy_status"], PolicyStatus.NO_MATCH.value)
        self.assertEqual(result.metadata["matched_policies"], [])
        self.assertIsNone(result.proposed_decision)

    def test_allows_internal_state_crud_inside_state_dir(self) -> None:
        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="update runtime state",
                metadata={
                    "explicit_action": True,
                    "target_type": "internal_state",
                    "target": "state-dir",
                    "operation": "update",
                },
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["policy_status"], PolicyStatus.ALLOWED.value)
        self.assertTrue(result.metadata["is_internal_state_action"])
        self.assertIsNone(result.proposed_decision)
        self.assertEqual(
            result.metadata["matched_policies"][0]["id"],
            "builtin-allow-internal-state",
        )

    def test_require_approval_policy_returns_approval_required_and_proposes_ask(
        self,
    ) -> None:
        self.store.add_policy(
            PolicyRule(
                id="shell-approval",
                name="Shell approval",
                description="Require approval before shell commands.",
                rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=10.0,
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            ),
            NexusState(),
        )

        self.assertEqual(
            result.metadata["policy_status"],
            PolicyStatus.APPROVAL_REQUIRED.value,
        )
        self.assertEqual(result.proposed_decision, DecisionAction.ASK)
        self.assertEqual(result.metadata["effective_policy"]["id"], "shell-approval")

    def test_deny_policy_overrides_allow_and_prefer(self) -> None:
        self.store.add_policy(
            PolicyRule(
                id="allow-shell",
                name="Allow shell",
                description="Allow shell use.",
                rule_type=PolicyRuleType.ALLOW,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=5.0,
            )
        )
        self.store.add_policy(
            PolicyRule(
                id="deny-shell",
                name="Deny shell",
                description="Deny shell use.",
                rule_type=PolicyRuleType.DENY,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=8.0,
            )
        )
        self.store.add_policy(
            PolicyRule(
                id="prefer-shell",
                name="Prefer shell logging",
                description="Prefer logging shell activity.",
                rule_type=PolicyRuleType.PREFER,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=2.0,
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["policy_status"], PolicyStatus.DENIED.value)
        self.assertEqual(result.proposed_decision, DecisionAction.RECORD)
        self.assertEqual(result.metadata["effective_policy"]["id"], "deny-shell")
        self.assertIn("prefer-shell", result.metadata["matched_policy_ids"])

    def test_prefer_policy_annotates_metadata_but_does_not_override_deny(self) -> None:
        self.store.add_policy(
            PolicyRule(
                id="deny-network",
                name="Deny network",
                description="Deny network calls.",
                rule_type=PolicyRuleType.DENY,
                target_type=PolicyTargetType.NETWORK,
                target="*",
                priority=4.0,
            )
        )
        self.store.add_policy(
            PolicyRule(
                id="prefer-network",
                name="Prefer network logging",
                description="Prefer extra network logging.",
                rule_type=PolicyRuleType.PREFER,
                target_type=PolicyTargetType.NETWORK,
                target="*",
                priority=9.0,
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="call an API",
                metadata={"explicit_action": True, "target_type": "network"},
            ),
            NexusState(),
        )

        self.assertEqual(result.metadata["policy_status"], PolicyStatus.DENIED.value)
        self.assertIn("prefer-network", result.metadata["matched_policy_ids"])
        self.assertEqual(result.metadata["effective_policy"]["id"], "deny-network")

    def test_disabled_policy_does_not_match(self) -> None:
        self.store.add_policy(
            PolicyRule(
                id="disabled-shell-allow",
                name="Disabled shell allow",
                description="Disabled rule.",
                rule_type=PolicyRuleType.ALLOW,
                target_type=PolicyTargetType.SHELL,
                target="*",
                enabled=False,
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            ),
            NexusState(),
        )

        self.assertEqual(
            result.metadata["policy_status"],
            PolicyStatus.APPROVAL_REQUIRED.value,
        )
        self.assertNotIn("disabled-shell-allow", result.metadata["matched_policy_ids"])

    def test_priority_resolves_same_type_matches(self) -> None:
        self.store.add_policy(
            PolicyRule(
                id="approval-low",
                name="Low approval",
                description="Low-priority approval rule.",
                rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=1.0,
            )
        )
        self.store.add_policy(
            PolicyRule(
                id="approval-high",
                name="High approval",
                description="High-priority approval rule.",
                rule_type=PolicyRuleType.REQUIRE_APPROVAL,
                target_type=PolicyTargetType.SHELL,
                target="*",
                priority=9.0,
            )
        )

        result = self.facet.process(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run a shell command",
                metadata={"explicit_action": True, "target_type": "shell"},
            ),
            NexusState(),
        )

        self.assertEqual(
            result.metadata["policy_status"],
            PolicyStatus.APPROVAL_REQUIRED.value,
        )
        self.assertEqual(result.metadata["effective_policy"]["id"], "approval-high")


class PolicyRuntimeIntegrationTests(unittest.TestCase):
    def test_nexus_runs_with_memory_goals_world_behavior_policy_and_echo_facets(
        self,
    ) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        state_store = FileStateStore(root)
        memory_store = SQLiteMemoryStore(root / "memory.sqlite3")
        goal_store = SQLiteGoalStore(root / "goals.sqlite3")
        world_store = SQLiteWorldModelStore(root / "world.sqlite3")
        policy_store = SQLitePolicyStore(root / "policy.sqlite3")
        goal_store.add_goal(
            Goal(
                id="goal-1",
                description="Keep the internal state healthy",
                priority=0.9,
                tags=["state"],
            )
        )
        world_store.add_belief(
            Belief(
                id="belief-1",
                claim="The configured state directory is safe for internal CRUD",
                confidence=0.9,
                tags=["state"],
            )
        )
        runtime = NexusRuntime(
            facets=[
                MemoryFacet(memory_store, retrieve_limit=2, working_limit=2),
                GoalsFacet(goal_store, active_limit=5, relevant_limit=3),
                WorldModelFacet(world_store, active_limit=5, relevant_limit=3),
                BehaviorFacet(),
                PolicyFacet(policy_store, state_dir=root),
                EchoFacet(),
            ],
            store=state_store,
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="update the runtime state",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "internal_state",
                    "target": "state-dir",
                    "tags": ["state"],
                },
            )
        )
        policy_result = next(
            result for result in record.facet_results if result.facet_name == "policy"
        )

        self.assertEqual(len(record.facet_results), 6)
        self.assertEqual(
            [result.facet_name for result in record.facet_results],
            ["memory", "goals", "world_model", "behavior", "policy", "echo"],
        )
        self.assertEqual(record.decision.action, DecisionAction.ACT)
        self.assertEqual(policy_result.metadata["policy_status"], "allowed")

    def test_policy_can_downgrade_act_to_ask_when_approval_is_required(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(SQLitePolicyStore(root / "policy.sqlite3"), state_dir=root),
                EchoFacet(),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="run the shell command",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "shell",
                },
            )
        )

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(record.decision.source_facets, ["policy"])

    def test_internal_state_actions_are_allowed_without_approval(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(SQLitePolicyStore(root / "policy.sqlite3"), state_dir=root),
                EchoFacet(),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="update the state snapshot",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "internal_state",
                    "target": "state-dir",
                },
            )
        )
        policy_result = next(
            result for result in record.facet_results if result.facet_name == "policy"
        )

        self.assertEqual(record.decision.action, DecisionAction.ACT)
        self.assertEqual(policy_result.metadata["policy_status"], "allowed")
        self.assertFalse(policy_result.metadata["approval_required"])

    def test_external_side_effect_like_metadata_requires_approval_by_default(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        runtime = NexusRuntime(
            facets=[
                BehaviorFacet(),
                PolicyFacet(SQLitePolicyStore(root / "policy.sqlite3"), state_dir=root),
                EchoFacet(),
            ],
            store=InMemoryStateStore(),
        )

        record = runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="modify the repository",
                metadata={
                    "explicit_action": True,
                    "low_risk": True,
                    "target_type": "git",
                },
            )
        )
        policy_result = next(
            result for result in record.facet_results if result.facet_name == "policy"
        )

        self.assertEqual(record.decision.action, DecisionAction.ASK)
        self.assertEqual(policy_result.metadata["policy_status"], "approval_required")
        self.assertEqual(
            policy_result.metadata["matched_policies"][0]["id"],
            "builtin-require-approval-git",
        )


class CLIPolicyIntegrationTests(unittest.TestCase):
    def test_cli_with_policy_creates_policy_sqlite_under_state_dir_by_default(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--policy",
                    "--content",
                    "policy smoke",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue((root / "policy.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_cli_policy_db_flag_overrides_default_path(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        custom_db = root / "custom" / "policy.sqlite3"

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--policy",
                    "--content",
                    "policy smoke",
                    "--state-dir",
                    str(root),
                    "--policy-db",
                    str(custom_db),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(custom_db.exists())
        self.assertFalse((root / "policy.sqlite3").exists())
        self.assertTrue((root / "state.json").exists())
        self.assertTrue((root / "runtime-log.jsonl").exists())
        self.assertEqual(payload["decision"]["action"], "record")

    def test_create_policy_metadata_creates_a_policy(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--policy",
                    "--content",
                    "Require approval before shell commands",
                    "--metadata",
                    '{"create_policy": true, "rule_type": "require_approval", "target_type": "shell", "target": "*"}',
                    "--state-dir",
                    str(root),
                ]
            )

        store = SQLitePolicyStore(root / "policy.sqlite3")
        policies = store.list_policies(limit=5)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0].rule_type, PolicyRuleType.REQUIRE_APPROVAL)
        self.assertEqual(policies[0].target_type, PolicyTargetType.SHELL)
        self.assertEqual(policies[0].target, "*")


if __name__ == "__main__":
    unittest.main()
