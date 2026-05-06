from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from fullerene.executor import ExecutionMode, ExecutionStatus, InternalActionExecutor, SkillManifestEntry, resolve_sandbox_path
from fullerene.facets import ExecutorFacet
from fullerene.nexus import Event, EventType, NexusState
from fullerene.planner import Plan, PlanStep
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    root = workspace_state_root() / f".test-executor-v1-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_plan(*steps: PlanStep) -> Plan:
    return Plan(id="plan-v1", source_event_id="event-v1", title="plan", steps=list(steps), reasons=["test"])


class ExecutorV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.executor = InternalActionExecutor(state_dir=self.root)

    def test_registry_lists_builtin_file_skills(self) -> None:
        names = {entry.skill_name for entry in self.executor.registry.list_skills()}
        self.assertIn("file_read", names)
        self.assertIn("file_write", names)
        self.assertIn("file_list", names)

    def test_unknown_skill_is_refused(self) -> None:
        plan = make_plan(
            PlanStep(
                id="s1",
                description="unknown",
                order=1,
                target_type="file",
                policy_status="allowed",
                metadata={"action_type": "file_read", "skill_name": "unknown_skill", "path": "a.txt"},
            )
        )
        result = self.executor.execute(plan, mode=ExecutionMode.DRY_RUN)
        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertEqual(result.records[0].metadata.get("reason"), "skill_not_registered")

    def test_external_skill_must_be_registered(self) -> None:
        self.executor.register_skill(
            SkillManifestEntry(
                skill_name="test_external_skill",
                version="v1",
                action_types=["external_echo"],
                target_types=["general"],
                dry_run_supported=True,
                live_supported=True,
            ),
            lambda **kwargs: {"success": True, "echo": kwargs.get("payload", {})},
        )
        plan = make_plan(
            PlanStep(
                id="s1",
                description="external",
                order=1,
                target_type="general",
                policy_status="allowed",
                metadata={"action_type": "external_echo", "skill_name": "test_external_skill", "value": "ok"},
            )
        )
        result = self.executor.execute(plan, mode=ExecutionMode.DRY_RUN)
        self.assertEqual(result.overall_status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.records[0].skill_name, "test_external_skill")

    def test_sandbox_path_blocks_traversal(self) -> None:
        with self.assertRaises(ValueError):
            resolve_sandbox_path(self.root / "sandbox", "../outside.txt")

    def test_file_write_dry_run_does_not_write(self) -> None:
        plan = make_plan(
            PlanStep(
                id="w1",
                description="write",
                order=1,
                target_type="file",
                policy_status="allowed",
                metadata={"action_type": "file_write", "skill_name": "file_write", "path": "notes/a.txt", "content": "hello", "create_parent_dirs": True},
            )
        )
        result = self.executor.execute(plan, mode=ExecutionMode.DRY_RUN)
        self.assertEqual(result.overall_status, ExecutionStatus.SUCCESS)
        self.assertFalse((self.root / "sandbox" / "notes" / "a.txt").exists())

    def test_file_write_live_writes_inside_sandbox(self) -> None:
        plan = make_plan(
            PlanStep(
                id="w1",
                description="write",
                order=1,
                target_type="file",
                policy_status="allowed",
                metadata={"action_type": "file_write", "skill_name": "file_write", "path": "notes/live.txt", "content": "hello", "create_parent_dirs": True},
            )
        )
        result = self.executor.execute(plan, mode=ExecutionMode.LIVE)
        self.assertEqual(result.overall_status, ExecutionStatus.SUCCESS)
        self.assertEqual((self.root / "sandbox" / "notes" / "live.txt").read_text(encoding="utf-8"), "hello")

    def test_file_read_dry_run_no_content(self) -> None:
        target = self.root / "sandbox" / "x.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("hello", encoding="utf-8")
        plan = make_plan(
            PlanStep(
                id="r1",
                description="read",
                order=1,
                target_type="file",
                policy_status="allowed",
                metadata={"action_type": "file_read", "skill_name": "file_read", "path": "x.txt"},
            )
        )
        result = self.executor.execute(plan, mode=ExecutionMode.DRY_RUN)
        self.assertNotIn("content", result.records[0].metadata)

    def test_file_operation_log_excludes_content(self) -> None:
        plan = make_plan(
            PlanStep(
                id="w1",
                description="write",
                order=1,
                target_type="file",
                policy_status="allowed",
                metadata={"action_type": "file_write", "skill_name": "file_write", "path": "audit.txt", "content": "secret", "overwrite": True},
            )
        )
        self.executor.execute(plan, mode=ExecutionMode.DRY_RUN)
        row = self.executor.file_operation_log[-1]
        self.assertNotIn("content", row)
        self.assertEqual(row["operation"], "write")

    def test_approval_pending_then_timeout(self) -> None:
        step = PlanStep(
            id="a1",
            description="needs approval",
            order=1,
            target_type="file",
            requires_approval=True,
            policy_status="approval_required",
            metadata={"action_type": "file_list", "skill_name": "file_list", "path": ".", "cycle_id": 1},
        )
        first = self.executor.execute(make_plan(step), mode=ExecutionMode.DRY_RUN)
        self.assertEqual(first.overall_status, ExecutionStatus.PENDING_APPROVAL)
        step.metadata["cycle_id"] = 10
        second = self.executor.execute(make_plan(step), mode=ExecutionMode.DRY_RUN)
        self.assertEqual(second.overall_status, ExecutionStatus.APPROVAL_TIMEOUT)

    def test_no_partial_execution_after_failure_marks_remaining_skipped(self) -> None:
        plan = make_plan(
            PlanStep(id="s1", description="bad", order=1, target_type="file", policy_status="allowed", metadata={"action_type": "file_write", "skill_name": "file_write", "path": "../bad.txt", "content": "x"}),
            PlanStep(id="s2", description="later", order=2, target_type="file", policy_status="allowed", metadata={"action_type": "file_list", "skill_name": "file_list", "path": "."}),
        )
        result = self.executor.execute(plan, mode=ExecutionMode.LIVE)
        self.assertEqual(result.records[-1].metadata.get("reason"), "skipped_due_to_prior_failure")

    def test_live_without_policy_allowed_fails_closed(self) -> None:
        plan = make_plan(
            PlanStep(id="s1", description="write", order=1, target_type="file", metadata={"action_type": "file_write", "skill_name": "file_write", "path": "a.txt", "content": "x"})
        )
        result = self.executor.execute(plan, mode=ExecutionMode.LIVE)
        self.assertEqual(result.overall_status, ExecutionStatus.FAILED)
        self.assertEqual(result.records[0].metadata.get("reason"), "policy_not_allowed_for_live")


class ExecutorFacetFeedbackTests(unittest.TestCase):
    def test_planner_feedback_metadata_is_exposed(self) -> None:
        facet = ExecutorFacet(state_dir=make_tempdir_path())
        state = NexusState(
            facet_state={
                "planner": {
                    "last_plan": make_plan(
                        PlanStep(
                            id="s1",
                            description="list",
                            order=1,
                            target_type="file",
                            policy_status="allowed",
                            metadata={"action_type": "file_list", "skill_name": "file_list", "path": "."},
                        )
                    ).to_dict()
                }
            }
        )
        result = facet.process(Event(event_type=EventType.USER_MESSAGE, content="go", metadata={"execute_plan": True}), state)
        self.assertIn("last_step_results", result.metadata)
        self.assertIn("requires_plan_reevaluation", result.metadata)


if __name__ == "__main__":
    unittest.main()
