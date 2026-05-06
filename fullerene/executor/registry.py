"""Explicit manifest-backed skill registry for Executor v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fullerene.executor.models import SkillManifestEntry

SkillHandler = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class RegisteredSkill:
    entry: SkillManifestEntry
    handler: SkillHandler


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}

    def register_skill(self, entry: SkillManifestEntry, handler: SkillHandler) -> None:
        key = entry.skill_name.strip().lower()
        if not key:
            raise ValueError("skill_name is required.")
        self._skills[key] = RegisteredSkill(entry=entry, handler=handler)

    def get_skill(self, skill_name: str) -> RegisteredSkill | None:
        return self._skills.get(str(skill_name or "").strip().lower())

    def list_skills(self) -> list[SkillManifestEntry]:
        return [registered.entry for registered in sorted(self._skills.values(), key=lambda item: item.entry.skill_name)]

    def validate_skill_invocation(
        self,
        *,
        skill_name: str,
        action_type: str,
        target_type: str,
    ) -> tuple[bool, str]:
        registered = self.get_skill(skill_name)
        if registered is None:
            return False, "skill_not_registered"
        entry = registered.entry
        normalized_action = str(action_type or "").strip().lower()
        normalized_target = str(target_type or "").strip().lower()
        if normalized_action and entry.action_types and normalized_action not in entry.action_types:
            return False, "action_type_not_supported"
        if normalized_target and entry.target_types and normalized_target not in entry.target_types:
            return False, "target_type_not_supported"
        return True, "ok"
