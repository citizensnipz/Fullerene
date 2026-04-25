# Fullerene — Claude Code rule (harness)

Claude Code loads rules from `.claude/rules/`. This file **does not replace** root `CLAUDE.md`; it reinforces the same contract.

## Source of truth

- **Index**: `ai/MEMORY.md` — open this first, then task-specific files from its tables.
- **Adapter**: root `CLAUDE.md` — keep it lightweight; expand detail in `ai/` not in chat.

## Discipline

- Verify before completion (`ai/operations/verification.md`).
- Prefer minimal context: targeted reads/search, not whole-repo dumps.
- After substantive edits: `ai/logs/CHANGELOG_AI.md`; significant product choices → `ai/knowledge/decisions.md`.
- Do not fabricate implementation details; unknown → say how to confirm in code or docs.

## Optional next steps for this folder

- Add `.claude/settings.json` when the team needs shared permissions, hooks, or env (see Claude Code docs).
- Add `.claude/skills/` for slash-invoked workflows that wrap `ai/prompts/*.md`.
