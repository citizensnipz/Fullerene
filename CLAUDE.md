# Fullerene — Claude

Use the shared AI harness under `ai/`. Do not bypass it.

## First reads

1. `ai/MEMORY.md` — navigation and categories
2. `ai/project/overview.md` — scope and goals
3. `ai/operations/verification.md` — before claiming “done”

## Rules (summary)

- Prefer small, scoped changes; no architecture rewrites unless explicitly requested.
- Verify behavior (tests, CLI, DB checks) before completion; do not guess internals.
- After meaningful work: update `ai/logs/CHANGELOG_AI.md` and append `ai/logs/SESSION_LOG.md` if session-level notes help the next run.
- Record non-obvious decisions in `ai/knowledge/decisions.md`.

Full workflow and prompts live in `ai/prompts/` and `ai/MEMORY.md`.
