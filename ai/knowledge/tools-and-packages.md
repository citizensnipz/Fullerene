# Tools and packages

Record **verified** choices from the repo. Remove or update rows when reality changes.

## Runtime (v0 intent from product description)

| Tool | Role | Notes |
|------|------|-------|
| Python | Language | Version **TBD** (pin in project when known) |
| SQLite | Persistence | Access pattern **TBD** |
| Ollama | Local LLM host | HTTP API; model ids **TBD** |

## Development

| Tool | Role | Status |
|------|------|--------|
| Test runner | e.g. pytest | **TBD** |
| Linter / formatter / types | e.g. Ruff, Black, mypy | **TBD** |

## Fullerene-specific (when code exists)

- Facet packaging and import graph — document in `ai/project/architecture.md` once stable.
- CLI framework (if any) — **TBD** (e.g. Click, Typer, argparse).

## Version pinning

When versions live in `pyproject.toml`, `requirements*.txt`, or similar, name that file here in one sentence (**TBD** until files exist).
