"""Project-local paths for runtime and test ephemera.

All process-local state, smoke output, and test fixtures should live under
``scratch/`` at the repository root (see ``.gitignore``) so the working tree
stays clean and agents do not sprawl new dot-directories at the project root.
"""

from __future__ import annotations

from pathlib import Path

SCRATCH_DIR_NAME = "scratch"

# Default for ``fullerene`` CLI ``--state-dir`` (relative to the process CWD).
DEFAULT_STATE_DIR = f"{SCRATCH_DIR_NAME}/.fullerene-state"


def scratch_root(cwd: Path | None = None) -> Path:
    """Return ``<cwd>/scratch``; *cwd* defaults to :func:`pathlib.Path.cwd`."""
    base = Path.cwd() if cwd is None else cwd
    return base / SCRATCH_DIR_NAME
