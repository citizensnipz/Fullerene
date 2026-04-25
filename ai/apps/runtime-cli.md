# App — Python runtime & CLI (v0)

Single primary surface for v0: **library + CLI** driving the Conductor locally.

## Purpose

- Run Fullerene continuously or stepwise on one machine.
- Expose operator-facing commands (status, inspect state, trigger tasks — exact commands TBD).

## Responsibilities

- Start/stop or tick the **Conductor loop**.
- Load configuration (paths, model endpoints — see `operations/env-vars.md`).
- Persist state via **SQLite** (`operations/database.md`).
- Integrate **Ollama** (or stub) for LLM calls invoked by Planner/Policy paths.

## Boundaries

- **In scope**: local process, structured logging to operator, facet orchestration.
- **Out of scope (v0)**: remote multi-tenant serving, voice, camera, robotics, unattended risky tool use.

## Known patterns (to document when implemented)

- Entrypoint: _TBD_ (e.g. `python -m fullerene` or `fullerene` console script).
- Config resolution order: _TBD_.
- Graceful shutdown: _TBD_ (flush SQLite, stop loop).

## Related harness files

- `ai/project/architecture.md`
- `ai/operations/commands.md`
- `ai/operations/verification.md`
