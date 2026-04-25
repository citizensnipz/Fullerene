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

### 2026-04-25 - Memory v1 deterministic tags + salience

- **Context:** Improve Memory v0 with deterministic tag extraction and salience scoring without adding embeddings, vector DBs, model calls, RAG, voice, or prosody.
- **Done:** Added `fullerene/memory/inference.py` (tag rules + salience signals + explain helpers); wired `MemoryFacet` to merge metadata + inferred tags and compute salience with breakdown metadata; added `explain_score` to `fullerene/memory/scoring.py`; expanded `tests/test_memory.py`; updated `ai/project/architecture.md` (Memory v1 section), `ai/knowledge/glossary.md`, and `ai/logs/CHANGELOG_AI.md`.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v` (30/30 passing); `python -m fullerene --memory --content "don't ever skip my boss emails" --state-dir .smoke-memory-v1` returned salience `0.7` with tags `["communication", "authority", "hard-rule-candidate"]`.
- **Next:** Decide whether semantic memory creation should be triggered from `hard-rule-candidate` tags before any embedding/vector index work, and consider adding a `--explain` CLI flag that surfaces the salience and retrieval breakdowns.
- **Blockers:** None.

### 2026-04-25 - Memory facet v0

- **Context:** Implement the first real Memory facet without embeddings, RAG, summarization, or a giant prompt file.
- **Done:** Added `fullerene/memory/` with `MemoryRecord`, deterministic scoring helpers, and `SQLiteMemoryStore`; added `fullerene/facets/memory.py`; wired the CLI to create `memory.sqlite3` beside `state.json` and `runtime-log.jsonl`; added `tests/test_memory.py`; updated architecture / glossary / decision docs for Memory v0.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --content "hello memory" --state-dir <temp-state-dir>`; `python -m fullerene --content "hello memory again" --state-dir <same-temp-state-dir>`
- **Next:** Add better deterministic salience and tag rules, then decide how semantic memory creation should happen before any embedding or vector index work.
- **Blockers:** None.

### 2026-04-25 - Nexus facet error isolation

- **Context:** Harden Nexus v0 so one facet failure does not abort event processing, later facets still run, and persistence still occurs.
- **Done:** Updated `fullerene/nexus/runtime.py` to isolate per-facet exceptions and convert them into sanitized `FacetResult` error records with `error_type` and `error_message` metadata. Extended `tests/test_nexus_runtime.py` to cover error isolation, continued execution, and persisted logging/state.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --content "hello nexus"`
- **Next:** If richer failure handling is needed later, decide whether facet errors should also set structured severity or retry hints without widening v0 scope.
- **Blockers:** None.

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
