# Environment variables & local config

Document **only variables that exist or are planned**. Do not invent secret names.

## Principles

- **Local-first**: defaults should work on a dev machine without cloud secrets.
- **Never commit** real secrets; use `.env` (gitignored) or OS secret store; reference variable *names* here.

## Template table

| Variable | Required | Used by | Description |
|----------|----------|---------|-------------|
| _TBD_ | | | e.g. SQLite path |
| _TBD_ | | | e.g. Ollama base URL |
| _TBD_ | | | e.g. default model name |

## Files (when introduced)

| File | Purpose |
|------|---------|
| `.env.example` | Safe template for developers |
| `.env` | Local overrides (not in git) |

## Notes

- If the runtime uses **XDG** or app data dirs, document the resolution order here.
- If no auth in v0, state “no auth env vars” explicitly once confirmed.
