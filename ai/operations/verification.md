# Verification - definition of "done"

A task is not complete until the checks below (plus any task-specific checks) pass.

## Always

- **Truth** - Claims match repo and, where relevant, runtime. List what was not run if anything was skipped.
- **Scope** - Diff matches the request; no unrelated files.
- **Harness** - Meaningful work includes an update to `ai/logs/CHANGELOG_AI.md`.

## When Python code exists

- Run unit/integration tests - use the command in `ai/operations/commands.md`.
- Run linter / typecheck if the repo configures them.
- Manual smoke of the operator CLI path:
  - `python -m fullerene --event-type user_message --content "hello nexus" --state-dir .fullerene-state`
  - Confirm `.fullerene-state/state.json` and `.fullerene-state/runtime-log.jsonl` were written.

## Facet and Nexus changes (when applicable)

- [ ] Loop terminates or idles safely (no busy spin)
- [ ] Snapshot/log writes stay inside the explicit state directory
- [ ] Policy / executor dangerous operations stay gated or absent for v0

## Database migrations (when applicable)

- [ ] Migration applies on empty DB and on previous version
- [ ] Rollback noted in `database.md` if supported

## Do not

- Mark complete on "looks fine" without at least one automated or documented manual check.
- Invent test commands - add them to `commands.md` first, then reference them here.
