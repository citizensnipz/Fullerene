# Deployment and runtime topology

## v0 expectation

- **Local process** — Operator runs the Python runtime / CLI on their machine.
- No production deployment target is implied by the product description yet.

## When deployment grows

Add sections as they become true:

- Target environment (OS, container, binary, …)
- Process model (single worker vs pool)
- Persistence (SQLite path, backups)
- Model serving (Ollama local vs remote — security notes)
- Rollback procedure

## Checklist (template)

- [ ] Health or “is Conductor alive” check documented  
- [ ] SQLite backup strategy  
- [ ] Upgrade path (schema migrations)

## Current truth

**TBD** — Document real steps when deployment exists.
