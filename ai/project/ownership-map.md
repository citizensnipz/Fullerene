# Ownership map — Fullerene

Useful when multiple humans or teams touch the codebase. **Template** — replace placeholders with real owners.

## How to use

- One **primary owner** per area (accountable).
- **Deputies** can merge small PRs; escalations go to primary.
- Link runbooks in `ai/operations/` when they exist.

## Areas

| Area | Primary | Deputies | Notes |
|------|---------|----------|-------|
| Conductor / orchestration | _TBD_ | _TBD_ | core loop |
| Memory + SQLite schema | _TBD_ | _TBD_ | migrations |
| Policy / skills / executor | _TBD_ | _TBD_ | safety-critical |
| CLI / UX | _TBD_ | _TBD_ | user-facing |
| Ollama / model adapters | _TBD_ | _TBD_ | swappable backends |
| AI harness (`ai/`) | _TBD_ | _TBD_ | keep in sync with truth |

## Escalation

- **Safety or data loss risk** → primary for Executor/Policy + documented verification path.
- **Schema migration** → coordinate with Memory owner; document in `operations/database.md`.
