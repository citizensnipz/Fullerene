"""Sandbox helpers for Executor v1 file operations."""

from __future__ import annotations

from pathlib import Path


class SandboxViolationError(ValueError):
    pass


def resolve_sandbox_path(sandbox_root: Path, requested_path: str) -> Path:
    root = sandbox_root.expanduser().resolve()
    raw = str(requested_path or "").strip()
    if not raw:
        raw = "."
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SandboxViolationError("outside_sandbox") from exc
    return resolved


def relative_sandbox_path(sandbox_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(sandbox_root.expanduser().resolve())).replace("\\", "/")
