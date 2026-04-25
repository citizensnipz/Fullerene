# Glossary - Fullerene

Terms for harness and design discussions. Definitions follow the current repo where code exists.

| Term | Meaning |
|------|---------|
| **Facet** | One of twelve modular components (Memory through Learning). |
| **Nexus** | Central interpreter/integrator loop that accepts events, asks facets for results, integrates a decision, and persists runtime state/logs. |
| **Conductor** | Earlier harness placeholder for the central loop; superseded in code by **Nexus**. |
| **Event** | Typed runtime input such as a user message, system tick, or system note. |
| **Memory** | Structured store; selective persistence by importance, affect, repetition. |
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
