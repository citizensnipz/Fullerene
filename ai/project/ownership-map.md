# Ownership map — Fullerene

Template for when multiple people own parts of the codebase. Replace **TBD** with names or teams.

## How to use

- One **primary** owner per area (accountable).
- **Deputies** may merge small changes; escalate to the primary when unclear.
- Link runbooks from `ai/operations/` when they exist.

## Areas

| Area | Primary | Deputies | Notes |
|------|---------|----------|-------|
| Conductor / orchestration | **TBD** | **TBD** | Core loop |
| Memory + SQLite schema | **TBD** | **TBD** | Migrations |
| Policy / skills / executor | **TBD** | **TBD** | Safety-sensitive |
| CLI / operator UX | **TBD** | **TBD** | User-facing |
| Ollama / model adapters | **TBD** | **TBD** | Swappable backends |
| AI harness (`ai/`) | **TBD** | **TBD** | Stays aligned with repo truth |

## Escalation

- **Safety or data loss** — Primary for policy/executor plus the verification path in `ai/operations/verification.md`.
- **Schema migrations** — Coordinate with memory/SQLite owner; document in `ai/operations/database.md`.
