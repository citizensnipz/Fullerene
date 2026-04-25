# Environment variables and local config

Document **only** variables that exist or are explicitly planned. Do not invent secret names.

## Principles

- **Local-first** — Sensible defaults on a dev machine without cloud-only secrets.
- **Secrets** — Never commit real values. Use `.env` (gitignored) or the OS secret store; list **names** here only.

## Variable table (fill when defined)

| Variable | Required | Used by | Description |
|----------|----------|---------|-------------|
| **TBD** | | | e.g. SQLite path |
| **TBD** | | | e.g. Ollama base URL |
| **TBD** | | | e.g. default model id |

## Files (when introduced)

| File | Purpose |
|------|---------|
| `.env.example` | Safe template for developers |
| `.env` | Local overrides (not in git) |

## Notes

- If the runtime uses **XDG** or app-data directories, document resolution order here.
- When v0 auth is confirmed absent, state “no auth env vars” explicitly in this section.
