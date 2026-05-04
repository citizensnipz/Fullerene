# Glossary - Fullerene

Terms for harness and design discussions. Definitions follow the current repo where code exists.

| Term | Meaning |
|------|---------|
| **Facet** | One of twelve modular components (Memory through Learning); the current v0 terminology uses **Behavior** in place of the earlier confidence/decision placeholder. |
| **Nexus** | Central interpreter/integrator loop that accepts events, asks facets for results, integrates a decision, and persists runtime state/logs. |
| **Conductor** | Deprecated harness-era name for the runtime loop; use **Nexus** for implemented runtime behavior. |
| **Event** | Typed runtime input such as a user message, system tick, or system note. |
| **Behavior Facet** | Deterministic decision-policy layer that proposes whether Fullerene should `WAIT`, `RECORD`, `ASK`, or `ACT`, along with inspectable reasons, tags, salience, and confidence. `ACT` is still only a typed proposal in v0. |
| **Memory** | Structured, selective persistence for what Fullerene remembers; in v0 the canonical store is SQLite rather than a monolithic prompt file. |
| **Working memory** | Bounded packet of currently useful state assembled for the current cycle. In Context v1 this is not just a recent-memory slice; it is dynamically assembled from the current event plus active goals, memories, beliefs, policy summary, and compact signal summaries under deterministic limits. |
| **Episodic memory** | Append-only record of what happened; the primary source-of-truth event/history memory in v0. |
| **Semantic memory** | Persistent facts or beliefs distilled from experience; schema-supported in v0, but richer creation logic is deferred. |
| **Tag inference** | Deterministic, lowercase-rule mapping from event content to memory tags (Memory v1). No model calls. |
| **Salience** | Deterministic importance score in `[0.0, 1.0]` attached to a memory record. Memory v1 sums transparent signal boosts (user message, hard-rule-candidate, urgent, correction, authority, communication) over a base score and clamps. |
| **Hard-rule-candidate** | Tag for content that uses absolute language (`never`, `always`, `must`, `don't ever`); a flag for content that may eventually become a stored hard rule. |
| **Affect** | Fullerene's internal affective state vector, derived from its own cognitive activity rather than the user's emotions. Affect v0 is deterministic, inspectable, observation-only, and not sentiment analysis, emotion recognition, or a personality layer. |
| **VAD** | Common affect-space shorthand for **valence**, **arousal**, and **dominance**. Fullerene Affect v0 uses VAD plus novelty. |
| **Valence** | Internal positive-to-negative tone in `[-1.0, 1.0]`, derived in Affect v0 from goal progress, goal failure/staleness, feedback, and executor outcomes. |
| **Arousal** | Internal calm-to-activated level in `[0.0, 1.0]`, derived in Affect v0 from pressure, urgency/salience, and attention spikes when available. |
| **Dominance** | Internal sense of control vs overwhelm in `[0.0, 1.0]`, derived in Affect v0 from executor success/failure and world-model confidence signals when available. |
| **Novelty** | Internal familiar-to-novel score in `[0.0, 1.0]`, derived in Affect v0 from explicit novelty metadata or inverse memory retrieval hit rate when available. |
| **Attention** | Deterministic focus selection for what deserves foregrounding right now. Attention v0 is a fixed-weight, metadata-only spotlight selector; it does not broadcast yet. |
| **AttentionItem** | Inspectable scored focus candidate with `id`, `source`, optional `source_id`, `content`, weighted `components`, total `score`, a `dominant_component`, and metadata. |
| **AttentionResult** | Inspectable Attention v0 output containing the selected top-N `focus_items`, per-candidate `scores`, optional `dominant_source`, strategy string, and metadata. |
| **Focus item** | A candidate that survives Attention v0 competition and lands in the selected top-N list for the current cycle. |
| **Broadcast** | Future Attention v1+ mechanism where the winning focus item is pushed back into the rest of the system. Not implemented in Attention v0. |
| **Context** | Current working packet of information available to Nexus, later facets, and CLI/model prompt grounding. Context v1 is deterministic, inspectable, bounded, and assembled from active state rather than static prompt text. |
| **Working context** | The concrete packet assembled for one Nexus cycle. It always includes the current event and may include active goals, relevant/recent memories, active beliefs, a compact policy summary, and compact signal summaries. |
| **Dynamic context assembly** | Deterministic bounded collection of context items from active stores/facet state using explicit limits, deduplication, simple ranking, and optional salience thresholds. It does not use embeddings, RAG, graph traversal, LLM summarization, or self-editing compression. |
| **ContextWindow** | Serialized context packet with `id`, `created_at`, `items`, `max_items`, `strategy`, and metadata. The current primary strategy is `dynamic_active_facets_v1`; `static_recent_episodic_v0` remains available for explicit compatibility. |
| **ContextItem** | One inspectable item inside a `ContextWindow`, with `id`, `item_type`, `content`, optional `source_id` / `created_at`, and metadata. Context v1 items can represent the current event, goals, memories, beliefs, policy summaries, or compact signal summaries. |
| **Belief** | Explicit persistent world-model record with `claim`, `confidence`, `status`, `tags`, timestamps, source metadata, and optional source links. |
| **World Model** | Structured belief store for what Fullerene believes about reality or tasks, separate from the event/memory log. World Model v0 is explicit and deterministic: SQLite-backed, inspectable, and does not implement automatic belief inference, graph reasoning, or Bayesian updates. |
| **Goal** | Explicit persistent record stored in SQLite with description, priority, status, tags, timestamps, source, and metadata. |
| **Goals** | Deterministic set of explicit goals that bias behavior and later planning. Goals v0 are inspectable and persistent; automatic goal inference is not implemented. |
| **Policy** | Deterministic permission and approval layer that constrains what Fullerene is allowed to do, forbidden from doing, or required to ask approval for. In v0 it evaluates explicit rules plus built-in sandbox defaults; it does not plan, infer rules, or execute tools. |
| **PolicyRule** | Explicit persistent policy row with `rule_type` (`allow`, `deny`, `require_approval`, `prefer`), `target_type`, `target`, `conditions`, priority, enabled/source flags, timestamps, and metadata. |
| **Sandbox** | The boundary around what Fullerene may manage directly without approval. In v0 the configured **state-dir** is the safe internal sandbox for self-managed runtime state. |
| **Repository state** | Gitignored `state/` at the repository root. Holds the default CLI `--state-dir` subtree, per-facet test trees (`mem_storage/`, `goals_storage/`, `world_model_storage/`), and other process-local files; see `fullerene.workspace_state`. Do not add new project-root dot-directories for runtime output. |
| **State-dir** | Explicit local directory that holds `state.json`, `runtime-log.jsonl`, and SQLite stores such as `memory.sqlite3`, `goals.sqlite3`, `world.sqlite3`, and `policy.sqlite3`. Internal CRUD inside this directory is allowed by default in Policy v0. The CLI default is `state/.fullerene-state` when `--state-dir` is omitted. |
| **Planner** | Deterministic plan-proposal layer that can emit inspectable next-step plans without executing them. Planner v0 is model-free, pressure-aware, and policy-filtered. |
| **Plan** | Inspectable proposed plan record with `id`, timestamps, optional source event / goal linkage, ordered `steps`, deterministic `confidence`, `pressure`, `status`, `reasons`, and metadata. |
| **PlanStep** | Inspectable ordered step inside a `Plan`, with description, `target_type`, `risk_level`, approval metadata, and step status such as `proposed`, `blocked`, or `requires_approval`. |
| **RiskLevel** | Deterministic planner/verifier risk label for a plan step: `low`, `medium`, or `high`. High-risk steps require approval before any future execution layer could act on them. |
| **Executor** | Deterministic action-execution layer that carries out approved internal actions only. Executor v0 uses explicit action handlers, does not parse step prose into behavior, is dry-run by default, halts before partial mutation, and refuses shell, network, git, arbitrary file, or other external side effects. |
| **ExecutionRecord** | Inspectable per-step execution artifact with optional `action_type`, optional plan linkage, `status`, `dry_run`, a human-readable message, and structured metadata about why the step ran, failed, or was skipped. Failure metadata carries explicit reason codes such as `blocked_by_policy`, `requires_approval`, `unsupported_action_type`, and `unsupported_target_type`. |
| **Dry-run** | Executor mode where Fullerene simulates supported actions and records what would happen without mutating state stores. Dry-run is the default for Executor v0, and switching to live mode does not broaden permissions. |
| **Helmet Rule** | Trust ladder for execution scope: v0 self-state only, v1 files and skills, v2 network/git reads, v3 real-world action. Trust is accumulated rather than assumed. |
| **Verifier** | Deterministic post-decision inspection layer that validates Fullerene's own runtime artifacts before persistence. In v0 it checks decision shape, facet-result shape, policy compliance, and conservative `ACT` safety requirements; it is not an LLM judge, planner, executor, truth-checker, or hallucination detector. |
| **VerificationResult** | Structured output from one deterministic verifier check: `check_name`, `status` (`passed`, `warning`, `failed`), `severity`, `message`, and metadata. |
| **Policy compliance** | Whether the final Nexus decision is structurally consistent with explicit policy outcomes such as `denied`, `approval_required`, or explicit `allow`, especially for side-effectful `ACT` decisions. |
| **Confidence** | In Behavior v0, confidence is a deterministic inspection score (`confidence` and `confidence_breakdown`) attached to policy output; it is not ML probability or model uncertainty. |
| **Learning** | Stateless feedback processor that observes explicit feedback and deterministic runtime outcomes, then emits traceable adjustment records or proposals. Learning v0 owns no persistent store of its own and does not perform neural training. |
| **LearningSignal** | Inspectable classified feedback artifact with `signal_type`, `source`, bounded `magnitude`, optional event/record linkage, metadata, and explicit reasons. |
| **AdjustmentRecord** | Inspectable adjustment artifact that names a target facet/field, records old/new values and delta when known, marks whether the change was applied, proposed, or skipped, and always points back to a `source_signal_id`. |
| **EMA** | Exponential moving average update rule. Learning v0 uses a conservative `alpha = 0.1` to compute a desired direction of movement before minor-nudge caps and proposal thresholds are applied. |
| **Proposal** | A non-applied adjustment suggestion emitted when the store/config surface is unavailable, the current value is unknown, or the requested change is too large to apply safely in v0. |
| **Skill** | Injectable capability (files, git, inbox, and so on) under policy control. |
| **Bob** | Example agent name from product vision; not a harness term. |
| **Stage** | Theater-model shorthand for Context. |
| **Spotlight** | Theater-model shorthand for Attention. |
| **Audience** | Theater-model shorthand for the facets that may eventually receive attention broadcast. |
| **Director** | Theater-model shorthand for Nexus as the global integration loop. |
| **Script** | Theater-model shorthand for Goals. |
| **Improvisation** | Theater-model shorthand for bottom-up salience and novelty signals that compete with top-down structure. |
