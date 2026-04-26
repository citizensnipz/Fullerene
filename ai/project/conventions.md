# Engineering conventions - Fullerene

Match whatever the repo already uses once Python code exists. These defaults now reflect the first runtime slice.

## Language and runtime

- **Python** for the v0 runtime.
- Minimum Python version is still **TBD** at the harness level.

## Project layout

| Path | Role |
|------|------|
| `fullerene/` | Library / runtime |
| `tests/` | Automated tests |
| `ai/` | Shared AI harness |
| `state/` | Gitignored; default CLI state, tests, and smoke output (see `fullerene.workspace_state`) |

## Style and quality

| Topic | Status |
|-------|--------|
| Formatter / linter | **TBD** |
| Type hints | Prefer on public facet-facing APIs |
| Module naming | `snake_case`; facet names align with `ai/knowledge/glossary.md` |

## Facets and side effects

- Each facet should have clear **inputs**, **outputs**, and **persistence** behavior.
- Side effects stay out of Nexus v0. Future action execution should remain explicit and policy-gated.

## Git

- Small commits; commit message explains **why** when non-obvious.
- No secrets or local runtime artifacts in VCS - see `ai/operations/env-vars.md`. Runtime and test output goes under `state/`, not ad-hoc folders or dot-directories at the repo root.

## AI agents

- Read `ai/MEMORY.md` before large edits.
- After meaningful changes: update `ai/logs/CHANGELOG_AI.md`.
- Do not reshape architecture in code or harness without explicit product direction.
