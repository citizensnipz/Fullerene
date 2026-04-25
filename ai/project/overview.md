# Fullerene - project overview

## What this is

- **Local-first cognitive architecture** - persistent state, modular facets, central integration loop (Nexus), not a single chat turn.
- **Current v0 implementation** - Python runtime, typed events and decisions, Nexus loop, facet interfaces, local snapshot/log persistence, CLI. No LLM provider integration, voice, camera, robotics, or autonomous risky actions yet.

## What this is not

- Not a generic chatbot wrapper.
- Not "one LLM is the whole system" - the LLM is a reasoning tool inside a constrained loop.

## Current repo state

- **Early stage:** root `README.md` states vision; implementation is now beginning with the Nexus runtime slice.
- Verified entrypoints and paths live in `ai/project/architecture.md` and `ai/apps/runtime-cli.md`.

## Engineering goals

- **Persistence** - state survives restarts (currently local snapshot/log files; SQLite can come later if needed).
- **Modularity** - facets (Memory, Affect, Attention, Context, World Model, Goals, Policy, Planner, Executor, Verifier, Behavior, Learning) stay isolatable.
- **Safety by default** - behavior + policy + verification before irreversible effects (expand as the executor gains tools).

## Non-goals (near term)

- Cloud-only operation, multimodal I/O, unconstrained autonomous actions.

## Placeholders

| Item | Status |
|------|--------|
| Primary CLI command | `python -m fullerene` |
| Minimum Python version | **TBD** |
| License | See repo `LICENSE` |
