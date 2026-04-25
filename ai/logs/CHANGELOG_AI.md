# AI-facing changelog

Changes that matter for future AI coding sessions (layout, commands, invariants). Not a substitute for user-facing release notes.

## Instructions

- Newest first.
- One line or bullet per change set; prefer paths over long prose.

## Changelog

### 2026-04-25 (e)

- Hardened `fullerene/nexus/runtime.py` so facet exceptions are isolated into sanitized `FacetResult` error entries instead of aborting the event loop.
- Extended `tests/test_nexus_runtime.py` to verify failed facets do not crash Nexus, other facets still run, and persistence still happens.

### 2026-04-25 (d)

- Added initial runtime package under `fullerene/` with `nexus/`, `facets/`, `state/`, `cli.py`, and `__main__.py`.
- Added `tests/test_nexus_runtime.py` and documented the runnable commands in `ai/operations/commands.md`.
- Updated harness terminology from Conductor to Nexus in `ai/project/overview.md`, `ai/project/architecture.md`, `ai/apps/runtime-cli.md`, `ai/operations/verification.md`, and `ai/knowledge/glossary.md`.
- Recorded the v0 runtime persistence decision in `ai/knowledge/decisions.md` and session handoff notes in `ai/logs/SESSION_LOG.md`.

### 2026-04-25 (c)

- Harness cleanup: reformatted `ai/**/*.md`; shortened `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.cursor/rules/fullerene-harness.mdc`, `.claude/rules/fullerene-harness.md`, `.codex/AGENTS.md`.
- Added harness-vs-repo truth rule and consolidated working agreements in `ai/MEMORY.md`; added `ai/README.md`.
- Clarified `ai/project/architecture.md` as harness vocabulary, not an implementation spec.

### 2026-04-25 (b)

- Tool-native hooks: `.cursor/rules/fullerene-harness.mdc`, `.claude/rules/fullerene-harness.md`, `.codex/AGENTS.md`; documented in `ai/MEMORY.md`; gitignore `.claude/settings.local.json`.

### 2026-04-25

- Initial `ai/` harness and root `CLAUDE.md`, `AGENTS.md`, `.cursorrules`.
