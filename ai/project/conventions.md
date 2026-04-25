# Engineering conventions — Fullerene

Match whatever the repo already uses once Python code exists. Until then, these are **defaults**, not requirements.

## Language and runtime

- **Python** for the v0 runtime (per product description).
- When `pyproject.toml` or `README.md` pins a version, record it here (**TBD** until then).

## Suggested layout (adjust if the repo differs)

| Path | Role |
|------|------|
| `src/fullerene/` or `fullerene/` | Library / runtime — **TBD** |
| `tests/` | Automated tests — **TBD** |
| `ai/` | This harness (markdown; tooling **TBD**) |

## Style and quality

| Topic | Status |
|-------|--------|
| Formatter / linter | **TBD** (e.g. Ruff, Black) |
| Type hints | Prefer on public facet-facing APIs when code exists |
| Module naming | `snake_case`; facet names align with `ai/knowledge/glossary.md` |

## Facets and side effects

- Each facet should have clear **inputs**, **outputs**, and **persistence** (yes/no; which tables **TBD**).
- Side effects only through **Executor** (or exceptions documented in code and harness).

## Git

- Small commits; commit message explains **why** when non-obvious.
- No secrets or local DB artifacts in VCS — see `ai/operations/env-vars.md`.

## AI agents

- Read `ai/MEMORY.md` before large edits.
- After meaningful changes: `ai/logs/CHANGELOG_AI.md`.
- Do not reshape architecture in code or harness without explicit product direction.
