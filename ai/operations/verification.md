# Verification — definition of “done”

No task is complete until checks here (or task-specific additions) pass.

## Always

- **Truth**: claims about behavior match repo + runtime observation; if not verified, say what was not run.
- **Scope**: no unrelated files changed; diff matches the request.
- **Harness**: meaningful work → `ai/logs/CHANGELOG_AI.md` updated (see that file for format).

## When Python code exists

- Run **unit/integration tests** (document exact command in `commands.md`).
- Run **linter/typecheck** if configured.
- **Manual**: smoke the CLI path used by operators (document steps in a bullet list here once known).

## Facet / Conductor changes

- [ ] Loop still terminates or idles safely (no busy spin) — adjust criterion when design is fixed
- [ ] SQLite transactions: no partial writes on simulated crash (if applicable)
- [ ] Policy/Executor: dangerous operations gated or absent in v0

## Database migrations

- [ ] Migration applies on empty DB and on previous version
- [ ] Rollback strategy noted in `database.md` if supported

## What not to do

- Do not mark complete on “looks fine” without at least one automated or documented manual check.
- Do not invent test commands; update `commands.md` first, then reference them here.
