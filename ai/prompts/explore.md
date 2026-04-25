# Prompt: explore (codebase or behavior)

Use when mapping Fullerene **before** changing code.

## Preconditions

- Read `ai/MEMORY.md` and open **only** linked files needed for the question.
- Do not load the whole repository into context; prefer search and narrow reads.

## Steps

1. **State the goal** in one sentence.
2. **List up to three hypotheses** about where logic or config lives.
3. **Verify** with paths and symbols (search, read); cite paths.
4. **Summarize**
   - Entrypoints  
   - Data flow (Conductor, facets, SQLite — **if** present in repo)  
   - Open questions marked **unknown**, each with a verification step  

## Constraints

- No code edits unless the user asked for fixes.
- If a module does not exist, say **not implemented** — do not invent file trees.

## Output shape

- Map (bullets)  
- Key files (paths)  
- Risks and unknowns  
- Suggested next prompt (`implement.md` or `debug.md`)  
