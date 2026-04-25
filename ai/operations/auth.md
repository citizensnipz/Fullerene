# Authentication and authorization

## v0 expectation

- **Local-first** CLI / runtime — no web product auth implied for v0.
- **Skills / executor** — The important “auth” story is **what tools may run** with which arguments. Document in code and here when implemented.

## Document actual behavior

| Surface | Model | Notes |
|---------|--------|-------|
| CLI | **TBD** | e.g. OS user only |
| SQLite file | **TBD** | Filesystem permissions |
| Ollama | **TBD** | Local bind, optional API key |

## If auth is added later

- Threat model (who can invoke what).
- Token storage and rotation.
- How policy maps principals to allowed skills.
