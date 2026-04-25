# Fullerene — Codex (project-scoped)

Codex loads project instructions from the repo root **`AGENTS.md`** and may also load **`.codex/AGENTS.md`** when the project is **trusted**.

This file exists so Codex users who rely on `.codex/` still land on the same harness.

## What to read

1. `../AGENTS.md` (root) — short bootstrap
2. `../ai/MEMORY.md` — full index and workflow

Keep root `AGENTS.md` and `ai/` in sync; avoid duplicating long policies in three places.

## Project config

- Optional team overrides: `.codex/config.toml` (only if you want committed Codex defaults). Codex skips project `.codex/` layers on untrusted clones — mark the repo trusted when appropriate.
