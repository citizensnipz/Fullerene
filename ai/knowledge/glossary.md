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
| **Working memory** | Small, recent memory slice derived from stored records for the current context window. In Context v0 this is a static bounded slice of recent episodic memory. |
| **Episodic memory** | Append-only record of what happened; the primary source-of-truth event/history memory in v0. |
| **Semantic memory** | Persistent facts or beliefs distilled from experience; schema-supported in v0, but richer creation logic is deferred. |
| **Tag inference** | Deterministic, lowercase-rule mapping from event content to memory tags (Memory v1). No model calls. |
| **Salience** | Deterministic importance score in `[0.0, 1.0]` attached to a memory record. Memory v1 sums transparent signal boosts (user message, hard-rule-candidate, urgent, correction, authority, communication) over a base score and clamps. |
| **Hard-rule-candidate** | Tag for content that uses absolute language (`never`, `always`, `must`, `don't ever`); a flag for content that may eventually become a stored hard rule. |
| **Affect** | Signal layer: intensity, urgency, confidence-like cues from text (voice later); not only "mood." |
| **Attention** | What the system foregrounds when building context. |
| **Context** | Current working packet of information available to Nexus and future reasoning systems. Context v0 is deliberately small, deterministic, inspectable, and assembled only from recent episodic memory. |
| **ContextWindow** | Serialized Context v0 packet with `id`, `created_at`, `items`, `max_items`, `strategy`, and metadata. The current strategy is `static_recent_episodic_v0`. |
| **ContextItem** | One inspectable item inside a `ContextWindow`, with `id`, `item_type`, `content`, optional `source_id` / `created_at`, and metadata. Context v0 currently emits memory-backed items only. |
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
| **Executor** | Runs actions or skills under sandbox and permissions. Not implemented in v0 Nexus. |
| **Verifier** | Deterministic post-decision inspection layer that validates Fullerene's own runtime artifacts before persistence. In v0 it checks decision shape, facet-result shape, policy compliance, and conservative `ACT` safety requirements; it is not an LLM judge, planner, executor, truth-checker, or hallucination detector. |
| **VerificationResult** | Structured output from one deterministic verifier check: `check_name`, `status` (`passed`, `warning`, `failed`), `severity`, `message`, and metadata. |
| **Policy compliance** | Whether the final Nexus decision is structurally consistent with explicit policy outcomes such as `denied`, `approval_required`, or explicit `allow`, especially for side-effectful `ACT` decisions. |
| **Confidence** | In Behavior v0, confidence is a deterministic inspection score (`confidence` and `confidence_breakdown`) attached to policy output; it is not ML probability or model uncertainty. |
| **Learning** | Structured updates to policy, goals, or world model from feedback - not neural training. |
| **Skill** | Injectable capability (files, git, inbox, and so on) under policy control. |
| **Bob** | Example agent name from product vision; not a harness term. |
