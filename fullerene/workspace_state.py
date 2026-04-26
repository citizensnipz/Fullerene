"""Top-level ``state/`` directory for local runtime, tests, and smokes.

All process-local files live under ``state/`` at the repository working directory
(see ``.gitignore``). This is *not* the same concept as a Nexus **State-dir**
(``--state-dir``), which is usually a subdirectory of this tree, e.g.
``state/.fullerene-state``).
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE_STATE_DIR_NAME = "state"

# Default for ``fullerene`` CLI ``--state-dir`` (relative to the process CWD).
DEFAULT_STATE_DIR = f"{WORKSPACE_STATE_DIR_NAME}/.fullerene-state"


def workspace_state_root(cwd: Path | None = None) -> Path:
    """Return ``<cwd>/state``; *cwd* defaults to :func:`pathlib.Path.cwd`."""
    base = Path.cwd() if cwd is None else cwd
    return base / WORKSPACE_STATE_DIR_NAME
