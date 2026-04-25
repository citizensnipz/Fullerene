# Prompt: implement

Use for features, focused refactors, or new modules.

## Preconditions

- `ai/MEMORY.md`  
- `ai/project/overview.md` (scope) and `ai/project/conventions.md`  
- If touching persistence or ops: `ai/operations/database.md`, `ai/operations/env-vars.md`  

## Plan (short)

- Goal  
- Non-goals (explicit)  
- Files you expect to touch  
- Risk to state or data (SQLite, migrations)  

## Execution rules

- **Small diffs** — Prefer one logical change per commit or PR when practical.
- **No architecture rewrites** unless the user explicitly requested that scope.
- **Interfaces** — For facets, clarify inputs and outputs before large code dumps.
- **Executor and skills** — Default deny; document permissions when adding tools.

## Verification

- Follow `ai/operations/verification.md`.
- Add or adjust tests when behavior is non-trivial (**TBD** until test commands exist).

## Closeout

- `ai/logs/CHANGELOG_AI.md`  
- `ai/knowledge/decisions.md` for significant product or engineering choices  
- `ai/operations/commands.md` when new commands are introduced  

## Output shape

- Summary of change  
- Files touched  
- How to verify  
- Harness updates done  
