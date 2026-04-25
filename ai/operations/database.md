# Database — SQLite (v0 intent)

## Role

- Durable **state** for memory, goals, world model, policy artifacts, etc., **as implemented**.
- Local-first v0 often uses a single DB file; confirm in code when it exists.

## Schema documentation rule

- When tables exist, list each with a **one-line** purpose.
- Record the migration approach (tool and workflow) once chosen.

## Tables (template)

| Table | Owner / module | Purpose |
|-------|------------------|---------|
| **TBD** | | |

## Migrations

| Step | Status |
|------|--------|
| Tool | **TBD** |
| Create migration | **TBD** |
| Apply migration | **TBD** |

## Integrity and backups

- **WAL** — Note here if SQLite WAL mode is used.
- **Backup** — Document approach (e.g. copy when stopped, or hot backup tool).

## Do not

- Commit production databases or PII dumps to the repository.
