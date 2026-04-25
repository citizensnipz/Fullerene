# Prompt: review (PR or diff)

Use before merge or after a large AI-generated diff.

## Preconditions

- Diff scope from the user or VCS  
- `ai/project/conventions.md` and relevant `ai/apps/*.md`  

## Checklist

- **Scope** — Matches request; no unrelated files.  
- **Correctness** — Edge cases, errors, SQLite transaction boundaries when relevant.  
- **Safety** — Executor or skills: destructive paths guarded?  
- **Performance** — Obvious N+1 or unbounded loops in hot paths?  
- **Tests** — New behavior covered; renames reflected.  
- **Harness** — Behavior or commands changed → docs updated.  

## Tone

- Specific references (file and region), not vague praise.  
- Split **must-fix**, **nit**, and **follow-up**.  

## Output shape

- Summary judgment (merge / merge with fixes / request changes)  
- Must-fix list (path + issue)  
- Nits (optional)  
- Harness or changelog gaps  
