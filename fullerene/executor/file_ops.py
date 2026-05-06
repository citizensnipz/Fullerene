"""Sandboxed file operations for Executor v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fullerene.executor.sandbox import SandboxViolationError, relative_sandbox_path, resolve_sandbox_path


def file_read(*, sandbox_root: Path, payload: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    max_bytes = int(payload.get("max_bytes", 65536) or 65536)
    encoding = str(payload.get("encoding", "utf-8") or "utf-8")
    requested_path = str(payload.get("path", "."))
    try:
        resolved = resolve_sandbox_path(sandbox_root, requested_path)
    except SandboxViolationError:
        return {"success": False, "reason": "outside_sandbox"}
    rel = relative_sandbox_path(sandbox_root, resolved)
    if dry_run:
        return {"success": True, "would_read": True, "resolved_relative_path": rel, "bytes_read": 0}
    if not resolved.exists() or not resolved.is_file():
        return {"success": False, "reason": "file_not_found", "resolved_relative_path": rel}
    size = resolved.stat().st_size
    if size > max_bytes:
        return {"success": False, "reason": "file_too_large", "resolved_relative_path": rel, "bytes": size}
    try:
        content = resolved.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return {"success": False, "reason": "file_unreadable", "resolved_relative_path": rel}
    return {"success": True, "resolved_relative_path": rel, "content": content, "bytes_read": len(content.encode(encoding, errors="ignore"))}


def file_write(*, sandbox_root: Path, payload: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    requested_path = str(payload.get("path", "")).strip()
    encoding = str(payload.get("encoding", "utf-8") or "utf-8")
    content = str(payload.get("content", ""))
    create_parent_dirs = bool(payload.get("create_parent_dirs", False))
    overwrite = bool(payload.get("overwrite", False))
    try:
        resolved = resolve_sandbox_path(sandbox_root, requested_path)
    except SandboxViolationError:
        return {"success": False, "reason": "outside_sandbox"}
    rel = relative_sandbox_path(sandbox_root, resolved)
    parent = resolved.parent
    if not parent.exists() and not create_parent_dirs:
        return {"success": False, "reason": "parent_missing", "resolved_relative_path": rel}
    if resolved.exists() and not overwrite:
        return {"success": False, "reason": "overwrite_not_allowed", "resolved_relative_path": rel}
    byte_count = len(content.encode(encoding, errors="ignore"))
    if dry_run:
        return {"success": True, "would_write": True, "resolved_relative_path": rel, "bytes_written": byte_count}
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding=encoding)
    return {"success": True, "resolved_relative_path": rel, "bytes_written": byte_count}


def file_list(*, sandbox_root: Path, payload: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    requested_path = str(payload.get("path", "."))
    recursive = bool(payload.get("recursive", False))
    max_entries = max(1, int(payload.get("max_entries", 100) or 100))
    try:
        resolved = resolve_sandbox_path(sandbox_root, requested_path)
    except SandboxViolationError:
        return {"success": False, "reason": "outside_sandbox"}
    rel = relative_sandbox_path(sandbox_root, resolved)
    if dry_run:
        return {"success": True, "would_list": True, "resolved_relative_path": rel, "entries": []}
    if not resolved.exists() or not resolved.is_dir():
        return {"success": False, "reason": "directory_not_found", "resolved_relative_path": rel}
    it = resolved.rglob("*") if recursive else resolved.iterdir()
    entries: list[str] = []
    for item in it:
        if len(entries) >= max_entries:
            break
        if item.is_file() or item.is_dir():
            entries.append(relative_sandbox_path(sandbox_root, item))
    return {"success": True, "resolved_relative_path": rel, "entries": entries, "truncated": len(entries) >= max_entries}
