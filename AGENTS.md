# Fullerene — Codex / OpenAI-style agents

This repository uses a **shared harness** under `ai/`. All agents should load context from there instead of duplicating project lore in chat.

## Bootstrap

1. Read `ai/MEMORY.md`
2. Skim `ai/project/overview.md` and `ai/project/architecture.md`
3. Check `ai/knowledge/known-issues.md` before debugging

## Operating rules

- **Verify** before stating completion (see `ai/operations/verification.md`).
- **Scope**: implement the smallest change that satisfies the request.
- **No invention**: if a fact is not in repo or harness docs, say unknown and how to verify.
- **Harness hygiene**: after substantive changes, append `ai/logs/CHANGELOG_AI.md`; use `ai/retros/TEMPLATE.md` after large or ambiguous work.

## Prompts

Reusable task prompts: `ai/prompts/explore.md`, `debug.md`, `implement.md`, `review.md`, `retro.md`.
