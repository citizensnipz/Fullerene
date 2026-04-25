# `ai/` — AI harness

This directory is **committed on purpose**. It is **shared project memory** for AI-assisted development: navigation, workflows, ops templates, and prompts so tools load less chat context and stay aligned.

## Who should use it

| Audience | Role |
|----------|------|
| **Human contributors** | May ignore `ai/`; it does not replace normal docs or code review. |
| **AI tools (Claude, Codex, Cursor, …)** | Should start here and follow `MEMORY.md`. |

## How agents should work

1. Open **`MEMORY.md`** first.
2. Open **only** files linked from it that match the current task.
3. If something in `ai/` disagrees with the repo, treat **`ai/` as stale** — verify in code, tests, package manifests, or runtime — then update the harness.

## Layout (summary)

| Area | Path (from repo root) |
|------|-------------------------|
| Index | `ai/MEMORY.md` |
| Product / engineering context | `ai/project/` |
| Surfaces (e.g. CLI) | `ai/apps/` |
| Run, verify, ship | `ai/operations/` |
| Decisions, glossary, issues | `ai/knowledge/` |
| Session handoff, AI changelog | `ai/logs/` |
| Retros | `ai/retros/` |
| Reusable task prompts | `ai/prompts/` |

Root **`CLAUDE.md`**, **`AGENTS.md`**, and **`.cursorrules`** stay short and point here.
