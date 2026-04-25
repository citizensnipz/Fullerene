# Prompt: explore (codebase / behavior)

Use this when mapping Fullerene before changing code.

## Preconditions

- Read `ai/MEMORY.md` and pick **only** linked files needed for the question.
- Do not load entire repo into context; use search and targeted reads.

## Steps

1. **State the goal** in one sentence (what you need to know).
2. **List hypotheses** (max 3) about where logic lives.
3. **Verify** with file paths and symbols (grep/read); cite paths.
4. **Summarize**:
   - entrypoints
   - data flow (Conductor → facets → SQLite if applicable)
   - open questions marked **unknown** with how to verify

## Constraints

- No code edits in explore mode unless the user asked for fixes.
- No invented modules — if absent, say “not implemented.”

## Output shape

- **Map** (bullets)
- **Key files** (paths)
- **Risks / unknowns**
- **Suggested next prompt** (`implement.md` / `debug.md`)
