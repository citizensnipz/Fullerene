# Fullerene — project overview

## What this is

- **Local-first cognitive architecture** — persistent state, modular facets, central orchestration (Conductor), not a single chat turn.
- **v0 intent** (from the product description): Python runtime, SQLite, Conductor loop, facet interfaces, Ollama for local models, basic memory + policy, CLI. No voice, camera, robotics, or autonomous risky actions.

## What this is not

- Not a generic chatbot wrapper.
- Not “one LLM is the whole system” — the LLM is a reasoning tool inside a constrained loop.

## Current repo state

- **Early stage:** root `README.md` states vision; implementation may be missing or partial.
- When code exists, add one line each here: entrypoints, facet layout (paths **TBD** until known).

## Engineering goals

- **Persistence** — state survives restarts (SQLite).
- **Modularity** — facets (Memory, Affect, Attention, Context, World Model, Goals, Policy, Planner, Executor, Verifier, Confidence, Learning) stay isolatable.
- **Safety by default** — policy + verification before irreversible effects (expand as the executor gains tools).

## Non-goals (near term)

- Cloud-only operation, multimodal I/O, unconstrained autonomous actions.

## Placeholders

| Item | Status |
|------|--------|
| Primary CLI command | **TBD** |
| Minimum Python version | **TBD** |
| License | See repo `LICENSE` |
