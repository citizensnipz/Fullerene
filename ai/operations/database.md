# Database — SQLite (v0)

## Role

- Durable **state** for Memory, Goals, World Model, Policy artifacts, etc., as implemented.
- Single-file or single-DB deployment for local-first v0 is typical; confirm actual design in code.

## Schema documentation rule

- When tables exist, list them here with **one-line purpose** each.
- Link or embed migration tool (Alembic, ad-hoc SQL, etc.) once chosen.

## Template: tables

| Table | Owner facet / module | Purpose |
|-------|----------------------|---------|
| _TBD_ | | |

## Migrations

- Tool: _TBD_
- How to create a migration: _TBD_
- How to apply: _TBD_

## Integrity & backups

- **WAL**: note if SQLite WAL mode is used.
- **Backup**: copy file when Conductor stopped, or use documented hot backup approach.

## Do not

- Commit production databases or PII dumps to the repo.
