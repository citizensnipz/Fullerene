# Session log

Cheap handoff between AI sessions or humans: what happened, what is next.

## Instructions

- Append newest entries at the top (below this section).
- Keep entries short (bullets; optional PR or commit links).
- Do not paste secrets or full environment dumps.

## Entry format

```markdown
## YYYY-MM-DD - topic - owner optional

- **Context:**
- **Done:**
- **Verified:** (commands or "N/A")
- **Next:**
- **Blockers:**
```

## Log

### 2026-04-25 - Nexus runtime v0

- **Context:** First real runtime implementation for Fullerene; keep it local, model-agnostic, and testable.
- **Done:** Added `fullerene/` package with Nexus runtime, typed event/decision/state models, facet protocol, `EchoFacet`, in-memory and file-backed state stores, `python -m fullerene` CLI, and `tests/test_nexus_runtime.py`.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --event-type user_message --content "hello nexus" --state-dir .fullerene-state-smoke`
- **Next:** Add more real facets, decide whether JSON persistence stays or gives way to SQLite, and define how future planner / policy / executor paths plug into Nexus.
- **Blockers:** None.

### 2026-04-25 - AI harness cleanup

- **Context:** Tighten harness readability, adapters, and truth rules; no product implementation work.
- **Done:** Reformatted `ai/` docs; shortened root and tool adapters; added `ai/README.md` and harness-vs-repo rule in `MEMORY.md`.
- **Verified:** Markdown review (docs only).
- **Next:** When Python layout exists, fill `ai/operations/commands.md` and `ai/project/architecture.md` mapping table.
- **Blockers:** None.

### 2026-04-25 - AI harness scaffold

- **Context:** Initial `ai/` harness and root adapter files for an early-stage repo.
- **Done:** MEMORY index, project / ops / knowledge templates, prompts, logs.
- **Verified:** N/A (docs only).
- **Next:** Add package layout; wire `ai/operations/commands.md`.
- **Blockers:** None.
