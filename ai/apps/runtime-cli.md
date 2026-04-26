# App - Python runtime and CLI (v0)

Single primary surface for v0: **library + CLI** driving the Nexus on one machine.

## Purpose

- Run Fullerene locally in a simple, inspectable way.
- Expose an operator-friendly command for processing events through the runtime.

## Responsibilities

- Process a local event through the **Nexus** loop.
- Persist state through a local snapshot/log directory.
- Stay model-agnostic; no provider integration yet.

## Boundaries

| In scope (v0) | Out of scope (v0) |
|---------------|-------------------|
| Local process, operator logging, facet orchestration | Remote multi-tenant serving |
| Explicit local persistence path | Voice, camera, robotics |
| Typed decisions without side effects | Unattended risky tool use |

## Current implementation

| Topic | Status |
|-------|--------|
| Entrypoint | `python -m fullerene --event-type user_message --content "hello"` |
| Config resolution order | CLI flags only (v0) |
| Graceful shutdown | Single-event command; no long-running loop yet |
| Local persistence path | `--state-dir` (defaults to `state/.fullerene-state`) |

## Related harness files

- `ai/project/architecture.md`
- `ai/operations/commands.md`
- `ai/operations/verification.md`
