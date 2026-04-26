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

## 2026-04-26 - Policy v0 allows internal state CRUD and requires approval for external side effects

- **Status:** accepted
- **Context:** Fullerene already had explicit memory, goals, world model, and behavior layers, but it still needed a deterministic, inspectable permission boundary before future executor/tool work lands.
- **Decision:** Implement `PolicyFacet` plus `fullerene/policy/` SQLite-backed explicit rule storage. Treat explicit policy rows as the canonical store for user/system rules, and enforce two built-in sandbox defaults in the runtime: internal CRUD inside the configured state directory is allowed by default, while modeled external side effects require approval by default unless an explicit allow rule matches. Explicit `deny` rules still override everything, and policy can downgrade `ACT` to `ASK` or `RECORD`.
- **Consequences:** Fullerene can manage its own memories, goals, beliefs, policy rows, and local runtime files inside `state-dir` without unnecessary approval prompts, while shell/network/message/git/tool/file-side-effect actions stay blocked behind approval unless the user explicitly allows them. Future executor work now has a deterministic policy gate to consult instead of relying on behavior heuristics alone.

## 2026-04-25 - Behavior v0 is deterministic, model-free, and inspectable

- **Status:** accepted
- **Context:** After Nexus and Memory v1, Fullerene needed its first explicit decision-policy layer for whether an event should `WAIT`, `RECORD`, `ASK`, or `ACT`, but the v0 scope still excludes LLM planning, graph reasoning, tool execution, and autonomous risky behavior.
- **Decision:** Implement `BehaviorFacet` as a deterministic rules layer over event type/content, explicit metadata, deterministic tags, deterministic salience, and optional passed-through memory metadata. The facet emits an inspectable proposal with reasons and confidence metadata. `ACT` remains only a typed proposal for future execution; the runtime still performs no tool execution.
- **Consequences:** Behavior decisions stay testable, debuggable, and source-visible in v0. Future model-based planning or confidence estimation can be layered on later, but they should not replace the canonical deterministic behavior path without another explicit architecture decision.

## 2026-04-25 - SQLite is the canonical Fullerene memory store

- **Status:** accepted
- **Context:** Memory v0 needed persistent, inspectable storage without loading one giant text file, and without introducing embeddings, vector infrastructure, or model-based summarization.
- **Decision:** Store canonical memory records in SQLite under the local state directory. Treat SQLite rows as source of truth for episodic and semantic memory data, with working memory derived from bounded recent retrieval. Any future embeddings, vector search, or compressed machine representations are retrieval indexes or caches, not the authoritative memory store.
- **Consequences:** Memory stays deterministic, queryable, and testable in v0. Future retrieval layers can be added without changing what counts as remembered data, but richer indexing and schema evolution will need migration discipline.

## 2026-04-25 - Nexus v0 is a small dataclass-based runtime with explicit local persistence

- **Status:** accepted
- **Context:** Fullerene needed its first real runtime slice without provider coupling, database setup, or autonomous side effects.
- **Decision:** Implement the central loop as `Nexus` / `NexusRuntime` with stdlib dataclasses for events, facet results, decisions, records, and state. Use a pluggable state store with in-memory and file-backed JSON/JSONL implementations under an explicit local state directory.
- **Consequences:** The runtime is easy to test and inspect, but intentionally simple. SQLite, model backends, and real action execution stay out of scope until later slices need them.

## Suggested future entries

- SQLite migration strategy
- Model-port abstraction if provider integration is added
- Event shape changes between Nexus and richer facets
