"""Executor v1 built-in skill registrations."""

from __future__ import annotations

from typing import Any

from fullerene.executor.file_ops import file_list, file_read, file_write
from fullerene.executor.models import SkillManifestEntry
from fullerene.executor.registry import SkillRegistry


def register_builtin_skills(registry: SkillRegistry) -> None:
    registry.register_skill(
        SkillManifestEntry(
            skill_name="memory_write",
            version="v0",
            action_types=["update_memory"],
            target_types=["memory", "internal_state"],
            dry_run_supported=True,
            live_supported=False,
            description="Legacy memory update dry-run skill.",
        ),
        _stub_builtin,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="goal_update",
            version="v0",
            action_types=["update_goal"],
            target_types=["goal", "internal_state"],
            dry_run_supported=True,
            live_supported=True,
            description="Legacy goal update skill.",
        ),
        _stub_builtin,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="internal_event",
            version="v0",
            action_types=["emit_event"],
            target_types=["event", "internal_state"],
            dry_run_supported=True,
            live_supported=True,
            description="Legacy internal event emission skill.",
        ),
        _stub_builtin,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="world_model_belief_update",
            version="v0",
            action_types=["update_belief"],
            target_types=["belief", "internal_state"],
            dry_run_supported=True,
            live_supported=True,
            description="Legacy world model update skill.",
        ),
        _stub_builtin,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="file_read",
            version="v1",
            action_types=["file_read"],
            target_types=["file"],
            dry_run_supported=True,
            live_supported=True,
            sandbox_required=True,
            description="Read text files inside sandbox.",
        ),
        _file_read_handler,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="file_write",
            version="v1",
            action_types=["file_write"],
            target_types=["file"],
            dry_run_supported=True,
            live_supported=True,
            sandbox_required=True,
            description="Write text files inside sandbox.",
        ),
        _file_write_handler,
    )
    registry.register_skill(
        SkillManifestEntry(
            skill_name="file_list",
            version="v1",
            action_types=["file_list"],
            target_types=["file"],
            dry_run_supported=True,
            live_supported=True,
            sandbox_required=True,
            description="List files inside sandbox.",
        ),
        _file_list_handler,
    )


def _stub_builtin(*, payload: dict[str, Any], dry_run: bool, **_: Any) -> dict[str, Any]:
    return {"success": True, "dry_run": dry_run, "payload": dict(payload)}


def _file_read_handler(*, payload: dict[str, Any], dry_run: bool, sandbox_root, **_: Any) -> dict[str, Any]:
    return file_read(sandbox_root=sandbox_root, payload=payload, dry_run=dry_run)


def _file_write_handler(*, payload: dict[str, Any], dry_run: bool, sandbox_root, **_: Any) -> dict[str, Any]:
    return file_write(sandbox_root=sandbox_root, payload=payload, dry_run=dry_run)


def _file_list_handler(*, payload: dict[str, Any], dry_run: bool, sandbox_root, **_: Any) -> dict[str, Any]:
    return file_list(sandbox_root=sandbox_root, payload=payload, dry_run=dry_run)
