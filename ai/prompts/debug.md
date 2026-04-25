# Prompt: debug

Use for defects, flaky behavior, or “why does it do X?”

## Preconditions

- `ai/knowledge/known-issues.md`
- `ai/operations/verification.md`
- Relevant facet doc in `ai/project/architecture.md` or code path

## Steps

1. **Repro**: exact command or steps; minimal case.
2. **Evidence**: logs, stack traces, failing test name — paste or path.
3. **Isolate**: binary search (commits, components) when helpful.
4. **Fix**: smallest change; avoid refactors mixed with bugfix.
5. **Verify**: run checks from `ai/operations/commands.md` (update file if commands were wrong/missing).

## Constraints

- Do not blame the model; trace deterministic code + config first.
- If repro impossible, list what you need from the user **specifically**.

## After

- Update `ai/knowledge/known-issues.md` if new sharp edge remains.
- Update `ai/logs/CHANGELOG_AI.md` for meaningful fixes.

## Output shape

- Repro
- Root cause (with path:line if possible)
- Fix summary
- Verification performed
