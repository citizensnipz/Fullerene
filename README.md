# Fullerene

Fullerene is an experimental local-first AI runtime built around persistent state, modular decision systems, and inspectable execution.

The project explores what an AI system looks like when the LLM is not treated as the whole agent. Instead of putting everything into a prompt and hoping the model handles it, Fullerene separates memory, goals, attention, policy, planning, verification, learning, and execution into distinct components coordinated by a central runtime.

The core idea:

> The model can reason and speak, but the system should own its state, decisions, constraints, and history.

## Architecture

Fullerene is organized around a central orchestration loop called the **Nexus**.

The Nexus receives an event, loads current state, runs the relevant components, integrates their outputs, chooses a decision, verifies the result, and persists the updated state.

    Event
      ↓
    Nexus
      ↓
    Facets / signal systems
      ↓
    Decision
      ↓
    Verification
      ↓
    Persistence

The runtime currently uses a small decision set:

- `WAIT`
- `RECORD`
- `ASK`
- `ACT`

Keeping the action space small makes the system easier to inspect and test while the architecture develops.

## The 12 facets

Fullerene uses twelve main facets as architectural boundaries:

1. **Memory**
2. **Affect**
3. **Attention**
4. **Context**
5. **World Model**
6. **Goals**
7. **Policy**
8. **Planner**
9. **Executor**
10. **Verifier**
11. **Behavior**
12. **Learning**

Each facet has a specific job.

### Memory

Memory stores and retrieves persistent records across runs.

It can store events, facts, preferences, tasks, feedback, outcomes, and other useful records. Current memory work includes role/domain classification, hybrid retrieval, salience scoring, optional embeddings, and inspectable memory edges.

### Affect

Affect tracks internal signal state such as valence, arousal, dominance, and novelty.

In the current runtime, Affect is an internal signal layer. It contributes runtime state; it is not used as a user emotion detector.

### Attention

Attention scores what should be foregrounded in the current cycle.

It can consider the current event, relevant memories, active goals, beliefs, execution results, and pressure signals. It emits a bounded attention broadcast instead of directly mutating other systems.

### Context

Context builds the working context for a cycle.

It assembles a bounded packet from the current event, memory, goals, beliefs, policy summaries, attention broadcasts, and recent system signals. The goal is to avoid dumping the entire state into a prompt.

### World Model

The World Model stores explicit beliefs separately from memory.

Memory records what happened or what was said. The World Model stores what the system currently believes, with confidence, status, tags, and provenance.

### Goals

Goals stores active goals and priorities.

Goals are explicit, inspectable records that can influence planning, behavior, attention, and context.

### Policy

Policy evaluates whether proposed actions are allowed, denied, preferred, or require approval.

It is deterministic and rule-based. It handles explicit rules, sandbox defaults, approval requirements, target scopes, and plan-level permission checks.

### Planner

Planner builds structured plans from goals, context, policy, and current pressure.

Planner proposes steps. It does not execute them.

### Executor

Executor executes approved internal actions.

It is conservative by default. It supports dry-run behavior, explicit action handlers, refusal reasons, and no partial execution when preflight checks fail.

### Verifier

Verifier checks structured runtime artifacts before they are trusted.

It validates decisions, behavior traces, Nexus cycle maps, planner output, executor results, learning adjustments, policy metadata, and other artifacts. It can recommend retries, escalation, or downgrade unsafe `ACT` decisions.

### Behavior

Behavior chooses the system’s next decision.

It scores candidate decisions using signals like pressure, salience, memory relevance, goal relevance, policy constraints, belief confidence, context load, ambiguity, and latent pressure.

### Learning

Learning routes feedback and outcome signals into conservative adjustments.

It can strengthen memory edges, update belief confidence, validate salience, and emit cross-facet proposals. It does not own its own canonical store and does not rewrite policy or behavior directly.

## Supporting systems

Not every subsystem is a facet.

Some systems are supporting infrastructure used by the Nexus and facets.

### Latent Pressure Buffer

The Latent Pressure Buffer tracks unresolved internal signals across cycles.

Examples include:

- unresolved contradictions
- policy blocks
- approval requirements
- context overload
- verifier failures
- planner conflicts
- repeated attention conflicts
- interrupt recommendations

LPB turns these into persistent pressure entries that can decay, escalate, resolve, or re-enter attention later.

It is supporting signal infrastructure, not a new facet.

## Nexus cycle

A typical Nexus cycle looks like this:

    1. Receive event
    2. Load current state
    3. Assemble context
    4. Run state and signal components
    5. Run behavior and planning components
    6. Aggregate pressure and cycle signals
    7. Apply policy and verifier constraints
    8. Produce final decision
    9. Collect learning events
    10. Persist state and runtime trace

The Nexus records cycle metadata such as:

- pressure components
- signal map
- facet order
- behavior decision trace
- verifier results
- learning events
- internal events queued or processed
- final decision

This makes the runtime easier to debug because decisions are not just returned as text. They leave behind a structured trace.

## State and persistence

Fullerene is local-first.

Runtime state is persisted locally. Current state includes:

- recent events
- memory records
- active goals
- beliefs
- policy rules
- attention history
- learning records
- Nexus cycle traces
- latent pressure state

The goal is for the system to preserve useful state across runs without relying on a giant prompt as the source of truth.

## Design principles

### State belongs to the system

The LLM should not be the only place where memory, goals, constraints, and decisions exist.

### Deterministic first

If a decision can be made with explicit code, it should not require an LLM call.

### Inspectable over magical

Important decisions should produce traces that explain what happened and why.

### Local-first

The runtime should be able to run locally and keep its state locally.

### Safety before autonomy

Planning, execution, policy, and verification are separated so the system can inspect actions before anything is applied.

### Small pieces, clear boundaries

Each component should have a narrow job. Fullerene should be easier to test because the runtime is broken into inspectable parts.

## Long-term direction

Fullerene is moving toward a runtime that can eventually stay active between direct user prompts.

That does not mean constantly generating text. The goal is for the system to support internal state transitions such as:

- decaying stale pressure
- escalating repeated unresolved signals
- updating attention
- maintaining active goals
- tracking unresolved contradictions
- deciding whether something is worth surfacing later

The long-term idea is not a chatbot loop. It is a stateful runtime where the model is one tool inside a larger system.

## Project status

Fullerene is early and experimental.

The current focus is on building the runtime architecture, keeping each component inspectable, and making sure decisions leave useful traces. The project will continue to change as the pieces become more capable and the system becomes easier to test.