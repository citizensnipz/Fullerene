# Authentication & authorization

## v0 expectation

- Product description emphasizes **local-first** CLI/runtime; **no web product auth** is implied for v0.
- **Skills / Executor**: authorization is the real “auth” story — what tools may run, with which args (document in code + here when implemented).

## Document actual behavior

| Surface | Auth model | Notes |
|---------|------------|-------|
| CLI | _TBD_ | e.g. OS user only |
| SQLite | _TBD_ | file permissions |
| Ollama | _TBD_ | local bind / optional API key |

## When auth is added later

- Threat model (who can call what).
- Token storage and rotation.
- How Fullerene’s **Policy** facet maps principals to allowed skills.
