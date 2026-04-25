# AI memory index — Fullerene

> **Harness vs repo:** Harness docs are project guidance, not unquestionable truth. If harness docs conflict with source code, tests, package files, or runtime behavior, treat the harness as stale, verify against the repo, and update the harness.

Central map for Claude, Codex, Cursor, and other tools. **Read this file first**, then open only what the task needs.

See **`ai/README.md`** for why this folder exists and who should use it.

**Path convention:** File references below are from the **repository root** (for example `ai/project/overview.md`).

---

## Working agreements (all AI tools)

- **Minimal context** — Open only files relevant to the current step; use search and targeted reads, not whole-repo dumps.
- **No guessing** — If a fact is not in the repo or harness, say it is unknown and how to verify. Do not invent implementation details.
- **Small, scoped changes** — Prefer the smallest diff that satisfies the request; avoid drive-by refactors.
- **Verification before completion** — Follow `ai/operations/verification.md`; do not claim “done” without checks that match the task.
- **Harness updates after meaningful work** — Update `ai/logs/CHANGELOG_AI.md`; add session notes to `ai/logs/SESSION_LOG.md` when handoff helps; use `ai/knowledge/decisions.md` for significant choices.
- **No architecture rewrites without explicit instruction** — Do not reshape the system or harness-level product layout unless the user asked for that scope.

---

## Tool-native entrypoints (optional)

These paths let each tool load rules its own way. Keep them **thin**; `ai/` stays the shared brain.

| Location | Tool | Role |
|----------|------|------|
| `.cursor/rules/fullerene-harness.mdc` | Cursor | `alwaysApply`; delegates here |
| `.claude/rules/fullerene-harness.md` | Claude Code | Extra rule next to root `CLAUDE.md` |
| `.codex/AGENTS.md` | Codex | Points at root `AGENTS.md` and this file |

Portable adapters: root **`CLAUDE.md`**, **`AGENTS.md`**, **`.cursorrules`**. Add **`.claude/settings.json`** or **`.codex/config.toml`** only when the team wants shared tool permissions or defaults.

---

## Workflow (non-trivial tasks)

1. **Orient** — `ai/project/overview.md`, `ai/project/architecture.md` (skim)
2. **Operate** — `ai/operations/commands.md`, `ai/operations/verification.md`
3. **Change** — `ai/prompts/implement.md` or `ai/prompts/debug.md`
4. **Close** — Verify, then `ai/logs/CHANGELOG_AI.md` (and `ai/logs/SESSION_LOG.md` if useful)
5. **Learn** — `ai/knowledge/decisions.md`, `ai/knowledge/known-issues.md` as needed

---

## `ai/project/`

| File | Purpose |
|------|---------|
| `ai/project/overview.md` | Vision, v0 scope, non-goals |
| `ai/project/architecture.md` | Harness-level product vocabulary (facets, Nexus) aligned to implemented runtime |
| `ai/project/conventions.md` | Engineering conventions; **TBD** where not yet chosen |
| `ai/project/ownership-map.md` | Owner template for teams |

---

## `ai/apps/`

| File | Purpose |
|------|---------|
| `ai/apps/runtime-cli.md` | v0 Python runtime + CLI surface (**TBD** entrypoints until implemented) |

Add more `ai/apps/*.md` when new surfaces exist.

---

## `ai/operations/`

| File | Purpose |
|------|---------|
| `ai/operations/env-vars.md` | Env vars and local config (**TBD** until defined) |
| `ai/operations/deployment.md` | Where and how it runs |
| `ai/operations/verification.md` | Definition of “done” |
| `ai/operations/commands.md` | Install, run, test, lint (**TBD** until toolchain exists) |
| `ai/operations/database.md` | SQLite notes (**TBD** schema) |
| `ai/operations/auth.md` | Auth story when known (**TBD**) |
| `ai/operations/payments.md` | N/A for v0 unless product changes |

---

## `ai/knowledge/`

| File | Purpose |
|------|---------|
| `ai/knowledge/tools-and-packages.md` | Verified toolchain (**TBD** until pinned) |
| `ai/knowledge/design-system.md` | UI placeholder until a UI exists |
| `ai/knowledge/known-issues.md` | Operational issues and workarounds |
| `ai/knowledge/decisions.md` | ADR-style log |
| `ai/knowledge/glossary.md` | Shared terms (facet, conductor, …) |

---

## `ai/logs/` and `ai/retros/`

| File | Purpose |
|------|---------|
| `ai/logs/SESSION_LOG.md` | Short handoff for the next session |
| `ai/logs/CHANGELOG_AI.md` | AI-facing change log |
| `ai/retros/TEMPLATE.md` | Retro template |

---

## `ai/prompts/`

| File | Use when |
|------|----------|
| `ai/prompts/explore.md` | Mapping codebase or behavior |
| `ai/prompts/debug.md` | Defects and “why does it…?” |
| `ai/prompts/implement.md` | Features or focused refactors |
| `ai/prompts/review.md` | PR or diff review |
| `ai/prompts/retro.md` | After a milestone or messy task |
