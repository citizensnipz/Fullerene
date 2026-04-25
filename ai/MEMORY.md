# AI memory index — Fullerene

Central map for Claude, Codex, Cursor, and other tools. **Read this file first**, then open only what the task needs.

## Tool-native entrypoints (optional layers)

These folders **mirror** the harness so each tool can load rules the way it expects. They should stay thin; `ai/` remains the shared brain.

| Location | Tool | Role |
|----------|------|------|
| `.cursor/rules/fullerene-harness.mdc` | Cursor | `alwaysApply` rule; delegates here |
| `.claude/rules/fullerene-harness.md` | Claude Code | Extra rule file alongside root `CLAUDE.md` |
| `.codex/AGENTS.md` | Codex CLI / IDE | Points at root `AGENTS.md` + this file |

Root `CLAUDE.md`, `AGENTS.md`, and `.cursorrules` are the **portable** adapters; add `.claude/settings.json` or `.codex/config.toml` only when the team needs tool-specific permissions or defaults.

## Workflow (every non-trivial task)

1. **Orient** — `project/overview.md`, `project/architecture.md` (skim)
2. **Operate** — `operations/commands.md`, `operations/verification.md`
3. **Change** — follow `prompts/implement.md` or `prompts/debug.md`
4. **Close** — verify, then `logs/CHANGELOG_AI.md` (+ `logs/SESSION_LOG.md` if handoff matters)
5. **Learn** — `knowledge/decisions.md`, `knowledge/known-issues.md` as needed

## Project

| File | Purpose |
|------|---------|
| `project/overview.md` | What Fullerene is, v0 scope, non-goals |
| `project/architecture.md` | Facets, conductor, data flow (high level) |
| `project/conventions.md` | Code style, naming, where things live |
| `project/ownership-map.md` | Who owns what when team grows (template) |

## Apps / surfaces

| File | Purpose |
|------|---------|
| `apps/runtime-cli.md` | Python runtime, SQLite, CLI, Ollama boundary |

Add more `apps/*.md` when new deployable surfaces exist.

## Operations

| File | Purpose |
|------|---------|
| `operations/env-vars.md` | Local config and secrets handling |
| `operations/deployment.md` | How/where it runs (v0: local) |
| `operations/verification.md` | Definition of “done,” checks |
| `operations/commands.md` | Install, run, test, lint |
| `operations/database.md` | SQLite schema/migrations notes |
| `operations/auth.md` | Auth model (v0: likely none — document truth) |
| `operations/payments.md` | N/A unless product adds billing |

## Knowledge

| File | Purpose |
|------|---------|
| `knowledge/tools-and-packages.md` | Python, Ollama, key libs |
| `knowledge/design-system.md` | UI only — placeholder until UI exists |
| `knowledge/known-issues.md` | Bugs, sharp edges, workarounds |
| `knowledge/decisions.md` | ADR-style decision log |
| `knowledge/glossary.md` | Terms: facet, conductor, affect, etc. |

## Logs & retros

| File | Purpose |
|------|---------|
| `logs/SESSION_LOG.md` | Short session trail for the next agent |
| `logs/CHANGELOG_AI.md` | AI-facing change log (not user release notes) |
| `retros/TEMPLATE.md` | Post-task or post-milestone retro |

## Prompts

| File | Use when |
|------|----------|
| `prompts/explore.md` | Mapping codebase or behavior |
| `prompts/debug.md` | Reproducing and fixing defects |
| `prompts/implement.md` | Feature or refactor with harness discipline |
| `prompts/review.md` | Reviewing a PR or diff |
| `prompts/retro.md` | After milestone or messy task |

## Global rules

- **No guessing** — cite repo paths or harness docs; otherwise mark unknown.
- **Minimal context** — open only files relevant to the current step.
- **Verification-first** — see `operations/verification.md` before completion.
- **Harness updates** — meaningful work → update changelog (and issues/decisions when appropriate).
