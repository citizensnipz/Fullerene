# Prompt: debug

Use for defects, flaky behavior, or “why does it do X?”

## Preconditions

- `ai/knowledge/known-issues.md`
- `ai/operations/verification.md`
- `ai/project/architecture.md` (vocabulary) or the relevant code path

## Steps

1. **Repro** — Exact commands or UI steps; smallest case.
2. **Evidence** — Logs, stack traces, failing test name (path or excerpt).
3. **Isolate** — Bisect commits or components when useful.
4. **Fix** — Smallest change; avoid mixing refactors with the bugfix.
5. **Verify** — Commands from `ai/operations/commands.md`; update that file if commands were wrong or missing.

## Constraints

- Trace deterministic code and config before blaming the model.
- If repro is impossible, list **specific** inputs you need from the user.

## After

- Update `known-issues.md` if a sharp edge remains.
- Update `ai/logs/CHANGELOG_AI.md` for meaningful fixes.

## Output shape

- Repro  
- Root cause (path:line if possible)  
- Fix summary  
- Verification performed  
