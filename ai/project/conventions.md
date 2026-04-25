# Engineering conventions — Fullerene

Align with whatever the repo already uses once Python code exists. Until then, prefer these defaults.

## Language & runtime

- **Python** for v0 runtime (per product description).
- Pin **minimum Python** in `pyproject.toml` or `README.md` when added; record here.

## Project layout (suggested — change if repo chooses otherwise)

- `src/fullerene/` or `fullerene/` — library/runtime
- `tests/` — pytest (or chosen framework)
- `ai/` — **this harness** (markdown only unless tooling added later)

## Style

- Formatter/linter: _TBD_ (e.g. Ruff, Black) — document actual choice.
- Types: prefer type hints on public facet interfaces.
- Naming: `snake_case` modules; facet names match glossary in `knowledge/glossary.md`.

## Facets

- Each facet: clear **inputs**, **outputs**, **persistence** (yes/no + which tables).
- Side effects only through **Executor** (or explicitly documented exceptions).

## Git

- Small commits; messages describe *why* when non-obvious.
- No generated secrets or local DB files in VCS — see `operations/env-vars.md`.

## AI agents editing the repo

- Read `ai/MEMORY.md` before large edits.
- After meaningful changes: `ai/logs/CHANGELOG_AI.md`.
- Do not rewrite architecture in code without product direction; update *this* harness when architecture changes.
