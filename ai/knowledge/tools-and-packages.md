# Tools & packages

Record **verified** toolchain choices. Remove or update rows when the repo changes.

## Runtime

| Tool | Role | Notes |
|------|------|-------|
| Python | v0 language | version TBD |
| SQLite | persistence | via stdlib or wrapper TBD |
| Ollama | local LLM host | HTTP API; model names TBD |

## Development (placeholders)

| Tool | Role | Status |
|------|------|--------|
| pytest | tests | TBD |
| Ruff / Black / mypy | quality | TBD |

## Fullerene-specific

- **Facets**: pure Python modules (planned); list actual import graph in `architecture.md` when stable.
- **CLI**: document framework if any (Click, Typer, argparse).

## Version pinning

- Where versions are pinned (`pyproject.toml`, `requirements*.txt`), mention the file here in one sentence.
