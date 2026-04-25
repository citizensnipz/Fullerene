# Glossary - Fullerene

Terms for harness and design discussions. Definitions follow the current repo where code exists.

| Term | Meaning |
|------|---------|
| **Facet** | One of twelve modular components (Memory through Learning). |
| **Nexus** | Central interpreter/integrator loop that accepts events, asks facets for results, integrates a decision, and persists runtime state/logs. |
| **Conductor** | Earlier harness placeholder for the central loop; superseded in code by **Nexus**. |
| **Event** | Typed runtime input such as a user message, system tick, or system note. |
| **Memory** | Structured, selective persistence for what Fullerene remembers; in v0 the canonical store is SQLite rather than a monolithic prompt file. |
| **Working memory** | Small, recent memory slice derived from stored records for the current context window. |
| **Episodic memory** | Append-only record of what happened; the primary source-of-truth event/history memory in v0. |
| **Semantic memory** | Persistent facts or beliefs distilled from experience; schema-supported in v0, but richer creation logic is deferred. |
| **Tag inference** | Deterministic, lowercase-rule mapping from event content to memory tags (Memory v1). No model calls. |
| **Salience** | Deterministic importance score in `[0.0, 1.0]` attached to a memory record. Memory v1 sums transparent signal boosts (instruction, emphasis, hard-rule-candidate, urgent, correction) over a base score and clamps. |
| **Hard-rule-candidate** | Tag for content that uses absolute language (`never`, `always`, `must`, `don't ever`); a flag for content that may eventually become a stored hard rule. |
| **Affect** | Signal layer: intensity, urgency, confidence-like cues from text (voice later); not only "mood." |
| **Attention** | What the system foregrounds when building context. |
| **Context** | Assembled view for reasoning or tool steps. |
| **World Model** | Structured beliefs about the environment or tasks (not raw logs). |
| **Goals** | Persistent objectives that drive planning. |
| **Policy** | Skills, rules, permission boundaries. |
| **Planner** | Produces plans or next steps; may invoke an LLM later. |
| **Executor** | Runs actions or skills under sandbox and permissions. Not implemented in v0 Nexus. |
| **Verifier** | Checks plans or outputs against constraints before commit. |
| **Confidence** | Meta-level certainty or gating for actions or model use. |
| **Learning** | Structured updates to policy, goals, or world model from feedback - not neural training. |
| **Skill** | Injectable capability (files, git, inbox, and so on) under policy control. |
| **Bob** | Example agent name from product vision; not a harness term. |
