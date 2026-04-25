# Prompt: implement

Use for features, small refactors, or new modules.

## Preconditions

- `ai/MEMORY.md` → `project/overview.md` (scope) + `conventions.md`
- If touching persistence or ops: `operations/database.md`, `operations/env-vars.md`

## Plan (short)

- Goal
- Non-goals (explicit)
- Files expected to touch
- Risk to state/data (SQLite, migrations)

## Execution rules

- **Small diffs**; one logical change per PR/commit when possible.
- **No architecture rewrites** unless the user explicitly requested them.
- **Interfaces first** for facets: clarify inputs/outputs before bulk code.
- **Executor / skills**: default deny; permissions documented.

## Verification

- Follow `ai/operations/verification.md`
- Add/adjust tests when behavior is non-trivial.

## Closeout (mandatory)

- `ai/logs/CHANGELOG_AI.md` entry
- If decision is seminal: `ai/knowledge/decisions.md`
- If new commands: `ai/operations/commands.md`

## Output shape

- Summary of change
- Files touched
- How to verify
- Harness updates done
