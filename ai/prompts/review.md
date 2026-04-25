# Prompt: review (PR / diff)

Use before merge or after an AI-generated large diff.

## Preconditions

- Diff scope from user or VCS
- `ai/project/conventions.md` + relevant `apps/*.md`

## Checklist

- **Scope**: matches request; no unrelated files.
- **Correctness**: edge cases, error paths, transaction boundaries (SQLite).
- **Safety**: Executor/skills — destructive ops guarded?
- **Performance**: obvious N+1, unbounded loops in Conductor?
- **Tests**: new behavior covered; renames reflected.
- **Docs**: harness updated if behavior or commands changed.

## Tone

- Be specific: file:region, not vague praise.
- Separate **must-fix** vs **nit** vs **follow-up**.

## Output shape

- Summary judgment (merge / merge with fixes / request changes)
- Must-fix bullets (path + issue)
- Nits (optional)
- Harness / changelog gaps
