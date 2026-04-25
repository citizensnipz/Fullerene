# Deployment & runtime topology

## v0 expectation

- **Local process**: operator runs the Python runtime/CLI on their machine.
- No production deployment target is implied by the product description yet.

## When deployment expands

Add sections as they become true:

- **Target environment** (OS, container, single binary, etc.)
- **Process model** (single worker vs pool)
- **Persistence** (SQLite file location, backups)
- **Model serving** (Ollama local vs remote — security implications)
- **Rollback** procedure

## Checklist (template)

- [ ] Health check / “is Conductor alive” command documented
- [ ] Backup strategy for SQLite
- [ ] Upgrade path (schema migrations)

## Current truth

- _Document actual deployment steps when they exist._
