# App — Python runtime and CLI (v0)

Single primary surface for v0: **library + CLI** driving the Conductor on one machine.

## Purpose

- Run Fullerene continuously or stepwise locally.
- Expose operator commands (exact verbs and flags **TBD**).

## Responsibilities

- Start, stop, or tick the **Conductor** loop.
- Load configuration (see `ai/operations/env-vars.md` — **TBD**).
- Persist state via **SQLite** (`ai/operations/database.md` — **TBD** schema).
- Integrate **Ollama** (or a stub) for LLM calls on planner/policy paths (**TBD**).

## Boundaries

| In scope (v0 intent) | Out of scope (v0) |
|----------------------|-------------------|
| Local process, operator logging, facet orchestration | Remote multi-tenant serving |
| | Voice, camera, robotics |
| | Unattended risky tool use |

## When implemented, document here

| Topic | Status |
|-------|--------|
| Entrypoint (e.g. `python -m …`, console script) | **TBD** |
| Config resolution order | **TBD** |
| Graceful shutdown (flush DB, stop loop) | **TBD** |

## Related harness files

- `ai/project/architecture.md`
- `ai/operations/commands.md`
- `ai/operations/verification.md`
