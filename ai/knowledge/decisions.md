# Architecture and process decisions (ADR-style)

Record decisions that matter later, not every small edit.

## Entry format

```markdown
## YYYY-MM-DD - Short title

- **Status:** proposed | accepted | superseded
- **Context:**
- **Decision:**
- **Consequences:**
- **Supersedes:** (if any)
```

## Decisions

## 2026-04-25 - Nexus v0 is a small dataclass-based runtime with explicit local persistence

- **Status:** accepted
- **Context:** Fullerene needed its first real runtime slice without provider coupling, database setup, or autonomous side effects.
- **Decision:** Implement the central loop as `Nexus` / `NexusRuntime` with stdlib dataclasses for events, facet results, decisions, records, and state. Use a pluggable state store with in-memory and file-backed JSON/JSONL implementations under an explicit local state directory.
- **Consequences:** The runtime is easy to test and inspect, but intentionally simple. SQLite, model backends, and real action execution stay out of scope until later slices need them.

## Suggested future entries

- SQLite migration strategy
- Model-port abstraction if provider integration is added
- Event shape changes between Nexus and richer facets
