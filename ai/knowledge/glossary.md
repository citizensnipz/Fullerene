# Glossary — Fullerene

Terms used across harness and code. Definitions align with the **product description** unless code narrows them.

| Term | Meaning |
|------|---------|
| **Facet** | One of twelve modular components (Memory, Affect, … Learning). |
| **Conductor** | Central orchestration loop observing state/events and coordinating facets. |
| **Memory** | Structured short/long-term store; selective persistence by importance, affect, repetition. |
| **Affect** | Signal layer: intensity, urgency, confidence-like cues from text (voice later); not “mood” only. |
| **Attention** | What the system foregrounds in context building. |
| **Context** | Assembled view fed to reasoning/tool steps. |
| **World Model** | Structured beliefs about environment/tasks (not raw logs). |
| **Goals** | Persistent objectives driving planning. |
| **Policy** | Skills, rules, permission boundaries. |
| **Planner** | Produces plans / next steps; may invoke LLM. |
| **Executor** | Runs actions/skills (sandboxed, permission-controlled). |
| **Verifier** | Checks outputs/plans against constraints before commit. |
| **Confidence** | Meta-level certainty / gating for actions or model use. |
| **Learning** | Updates to policy/goals/world model from feedback—not neural training. |
| **Skill** | Injectable capability (files, git, inbox, …) invoked by Executor under Policy. |
| **Bob** | Example persistent agent instance (product vision); not a harness term. |
