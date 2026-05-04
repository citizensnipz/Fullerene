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

## 2026-05-04 - Goal intent creation and Context v1 both deduplicate active goals deterministically

- **Status:** accepted
- **Context:** After Context v1 landed, reused state directories could still surface multiple active goals that meant the same thing but came from repeated intent phrasing such as `I should remember to finish Fullerene` and `remember to finish Fullerene`. That polluted working context before Planner v1.
- **Decision:** Add deterministic goal normalization and duplicate handling before planning. Goal-intent creation now normalizes descriptions by lowercasing, trimming, removing punctuation, collapsing whitespace, and stripping common leading intent phrases; when an active goal with the same normalized description already exists, the runtime updates that goal instead of creating a new row, merging tags and keeping the higher priority. Context v1 also deduplicates already-persisted active goals before exposing them in the working packet, preferring highest priority, then most recent `updated_at`, then newest `created_at`, with a conservative high-overlap keyword fallback only for near-duplicate comparison.
- **Consequences:** Active-goal context is cleaner and more stable across repeated runs on the same state directory, and Planner can consume a single deterministic goal packet instead of competing duplicate goals. The tradeoff is that exact normalized matching remains the canonical rule, so broader semantic duplicates still require future deliberate design rather than ad-hoc fuzzy matching.

## 2026-05-04 - Context v1 is a deterministic bounded working packet assembled from active state

- **Status:** accepted
- **Context:** Fullerene could already persist memories, goals, world beliefs, policies, and later facet outputs, but the old Context layer exposed only a static recent-memory slice. That left later responses under-grounded even when active goals or beliefs already existed in persistent stores.
- **Decision:** Implement `ContextAssemblyConfig`, `DynamicContextAssembler`, and upgraded `ContextFacet` support for `dynamic_active_facets_v1`. Context v1 always includes the current event, then bounded active goals, relevant/recent memories, active beliefs, a compact policy summary, and optional compact planner / executor / attention / affect / learning summaries when available. The assembly is deterministic, read-only, store-bounded, deduplicated, and visible both to later facet state and to the CLI model prompt builder through a concise working-context summary. It does not use embeddings, RAG, LLM summarization, graph traversal, pressure-based compression, or self-editing context mutation.
- **Consequences:** Fullerene now has a real working-context layer that can ground later behavior, planning, and response generation in persisted state without architecture rewrites or opaque prompt stuffing. Future context work can improve deterministic ranking and selection, but the canonical v1 mechanism remains a bounded assembly packet, not a retrieval or summarization subsystem.

## 2026-04-28 - Affect v0 is a deterministic internal VAD + novelty observer with no downstream influence

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, planning, execution, learning, and attention signals, but it still lacked a narrow place to summarize its own internal state from those signals without collapsing affect into user-emotion detection or behavior modulation.
- **Decision:** Implement `fullerene/affect/` plus `AffectFacet` as a deterministic internal observation layer. Affect v0 derives `valence`, `arousal`, `dominance`, and `novelty` from existing runtime signals only, records an inspectable `AffectState` and `AffectResult`, and may keep a short bounded history in Nexus facet state. It never proposes `ACT`, never mutates other stores, and does not modulate memory, attention, planning, execution, policy, or expression yet.
- **Consequences:** Fullerene now has an explicit affect boundary and a traceable data-collection layer for future work. Later salience modulation, affect-tagged memories, appraisal logic, or pressure integration can reuse the same inspectable artifacts, but v0 remains observational: no emotion recognition, no sentiment model, no prosody, and no learned affect inference.

## 2026-04-28 - Attention v0 is a deterministic metadata-only focus scorer with no broadcast

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, planner, executor, and learning signals, but it still lacked a narrow place to score what should receive foreground focus before any future broadcast mechanism exists.
- **Decision:** Implement `fullerene/attention/` plus `AttentionFacet` as a fixed-weight, inspectable scoring layer. Attention v0 always considers the current event, can score additional memory / goal / belief / execution candidates when those signals are already available, emits `AttentionItem` and `AttentionResult` metadata, selects top-N focus items, and never proposes `ACT`. It does not broadcast a winner, mutate Context, or own a learned model.
- **Consequences:** Fullerene now has an explicit spotlight boundary that is separate from behavior, planning, execution, and context assembly. Future attention broadcast, ignition, and learned weighting can build on the same inspectable artifacts without widening v0 into a decision-maker.

## 2026-04-27 - Learning v0 is a stateless feedback bus with apply-or-propose adjustment records

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, behavior, policy, planner, executor, and verifier layers, but it still lacked a narrow place to close the feedback loop after outcomes were observed.
- **Decision:** Implement `fullerene/learning/` plus `LearningFacet` as a stateless signal processor and feedback bus. Learning v0 classifies explicit user feedback, executor outcomes, and goal lifecycle metadata through deterministic rules only; it emits `LearningSignal`, `AdjustmentRecord`, and `LearningResult` payloads; it may apply only minor safe nudges to goal priority or memory salience when an existing store already supports that change cleanly; and it emits proposals instead of silently applying larger or unsupported changes. Learning owns no canonical persistent state of its own.
- **Consequences:** Fullerene now has an explicit post-outcome adjustment boundary without collapsing learning into memory, goals, behavior, or policy ownership. Future richer learning can build on the same traceable artifacts, but v0 remains conservative: no self-modification, no policy mutation, no executor permission changes, and no model calls.

## 2026-04-27 - Executor v0 is an internal-only execution layer with dry-run default and no partial execution

- **Status:** accepted
- **Context:** Fullerene already had deterministic planning, policy, and verification layers, but it still lacked a controlled place to carry out approved internal actions without collapsing execution into planner logic or opening external side effects.
- **Decision:** Implement `fullerene/executor/` plus `ExecutorFacet` as a deterministic execution boundary. Executor v0 accepts inspectable plans, validates every step before mutation, defaults to dry-run, executes only supported internal actions, and halts on the first blocked, approval-gated, high-risk, unsupported, or malformed step. It does not run shell commands, network calls, git actions, arbitrary file operations, tool execution, or LLM-driven skills.
- **Consequences:** Fullerene now has a concrete "hands" layer that remains narrow, inspectable, and policy-constrained. Planner still proposes, Policy still decides allowed vs approval-required vs denied, and Verifier still validates safety. Future executor versions can widen capability behind explicit roadmap and trust-boundary decisions instead of silently expanding v0.

## 2026-04-27 - Planner v0 is a deterministic plan-proposal facet with policy-filtered steps and no execution

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, behavior, policy, verifier, and context layers, but it still lacked a first-class place to propose ordered next steps without collapsing planning into behavior or introducing tool execution.
- **Decision:** Implement `fullerene/planner/` plus `PlannerFacet` as a deterministic, model-free plan proposal layer. Planner v0 triggers only on explicit plan requests or when a high-priority active goal is present and the current event explicitly asks for next steps. It emits inspectable `Plan` / `PlanStep` objects with deterministic confidence, simple pressure handling, step-level risk labels, and policy-filtered approval/blocking metadata. It does not execute steps or call tools.
- **Consequences:** Fullerene now has a distinct planning boundary that remains transparent and testable in v0. Future richer planning can build on the same plan objects, but execution stays separate and must still pass policy plus verifier checks.

## 2026-04-27 - Verifier v0 runs deterministic post-decision checks and may downgrade unsafe ACT decisions

- **Status:** accepted
- **Context:** Fullerene already had deterministic Behavior and Policy layers, but it still needed a final inspectable safeguard that could validate the aggregated decision trace itself before records were persisted.
- **Decision:** Implement `fullerene/verifier/` plus `VerifierFacet` as a deterministic post-decision validation pass. Nexus now runs normal facets, aggregates an initial decision, then runs verifier checks against the event, facet results, initial decision, and configured state-dir metadata. If verifier finds an unsafe or structurally invalid `ACT`, it may downgrade that decision to `ASK` or `RECORD` before persistence.
- **Consequences:** Fullerene gains a small internal test runner for its own decision process without adding model calls or autonomous execution. Verifier metadata is persisted as a normal `FacetResult`, so callers can inspect which checks failed and why. Future executor work can rely on both Policy and Verifier guardrails instead of behavior heuristics alone.

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
