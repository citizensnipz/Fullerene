# Verification — definition of “done”

A task is not complete until the checks below (plus any task-specific checks) pass.

## Always

- **Truth** — Claims match repo and, where relevant, runtime. List what was **not** run if anything was skipped.
- **Scope** — Diff matches the request; no unrelated files.
- **Harness** — Meaningful work includes an update to `ai/logs/CHANGELOG_AI.md` (see that file for format).

## When Python code exists

- Run unit/integration tests — exact command in `commands.md` (**TBD** until defined).
- Run linter / typecheck if the repo configures them (**TBD**).
- Manual smoke of the operator CLI path — document bullet steps here once known (**TBD**).

## Facet and Conductor changes (when applicable)

- [ ] Loop terminates or idles safely (no busy spin) — refine when design is fixed  
- [ ] SQLite transactions — no partial writes on simulated crash (if applicable)  
- [ ] Policy / executor — dangerous operations gated or absent for v0  

## Database migrations (when applicable)

- [ ] Migration applies on empty DB and on previous version  
- [ ] Rollback noted in `database.md` if supported  

## Do not

- Mark complete on “looks fine” without at least one automated or documented manual check.
- Invent test commands — add them to `commands.md` first, then reference them here.
