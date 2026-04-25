# Fullerene — project overview

## What this is

- **Local-first cognitive architecture**: persistent state, modular “facets,” central orchestration (Conductor), not a single chat turn.
- **v0 intent** (from product description): Python runtime, SQLite storage, Conductor loop, facet interfaces, Ollama for local models, basic memory + policy, CLI. No voice, camera, robotics, or autonomous risky actions.

## What this is not

- Not a generic chatbot wrapper.
- Not “one LLM is the whole system” — the LLM is a reasoning tool inside a constrained loop.

## Current repo state

- **Placeholder**: `README.md` describes vision; implementation may not exist yet.
- When code lands, add: entrypoints, facet module layout, and link them here in one line each.

## Goals (engineering)

- **Persistence**: state survives restarts (SQLite).
- **Modularity**: Memory, Affect, Attention, Context, World Model, Goals, Policy, Planner, Executor, Verifier, Confidence, Learning — each isolatable.
- **Safety by default**: policy + verification before irreversible effects (expand as executor gains tools).

## Non-goals (near term)

- Cloud-only operation, multimodal I/O, unconstrained autonomous actions.

## Placeholders (fill when known)

- **Primary CLI command**: _TBD_
- **Minimum Python version**: _TBD_
- **License**: see repo `LICENSE`
