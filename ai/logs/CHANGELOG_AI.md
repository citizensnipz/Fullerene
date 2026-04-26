# AI-facing changelog

Changes that matter for future AI coding sessions (layout, commands, invariants). Not a substitute for user-facing release notes.

## Instructions

- Newest first.
- One line or bullet per change set; prefer paths over long prose.

## Changelog

### 2026-04-26 (l)

- Added `fullerene/goals/` with `Goal`, `GoalStatus`, `GoalSource`, and `SQLiteGoalStore` as the canonical Goals v0 store in `goals.sqlite3`.
- Added `fullerene/facets/goals.py` and exported `GoalsFacet` so Nexus can emit deterministic goal-relevance signals from tag overlap, keyword overlap, and goal priority.
- Updated `fullerene/facets/behavior.py` so behavior can read inspectable goal signals and apply a small confidence boost without changing the core decision rules.
- Updated `fullerene/nexus/runtime.py` so later facets can observe earlier facet `state_updates` during the same event pass.
- Updated `fullerene/cli.py` with `--goals`, `--metadata`, and explicit metadata-driven `create_goal` support; goal creation remains explicit and deterministic.
- Added `tests/test_goals.py` covering goal model/store behavior, goals facet scoring, CLI goal DB creation, metadata-driven goal creation, and Memory+Goals+Behavior runtime integration.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/logs/SESSION_LOG.md` for Goals v0, including the explicit note that automatic goal inference is not implemented.

### 2026-04-26 (k)

- Updated `fullerene/cli.py` so `--memory` defaults memory DB placement to `<state-dir>/memory.sqlite3` when `--memory-db` is omitted; explicit `--memory-db` override remains supported.
- Added CLI coverage in `tests/test_memory.py` for state-dir default memory DB creation and explicit override path handling.
- Added a behavior-only CLI assertion in `tests/test_behavior.py` that `--behavior` runs independently and does not create `memory.sqlite3` unless `--memory` is enabled.
- Refreshed stale harness wording in `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/MEMORY.md` to emphasize Nexus naming, Behavior Facet v0 scope, deterministic inspectable confidence metadata, and no model/provider/tool execution integration in v0.

### 2026-04-25 (j)

- Added `fullerene/facets/behavior.py` with deterministic `BehaviorFacet` rules for `WAIT` / `RECORD` / `ASK` / `ACT`, including inspectable `selected_decision`, `confidence`, `salience`, `tags_considered`, and `reasons` metadata.
- Wired `BehaviorFacet` into `fullerene/facets/__init__.py`, `fullerene/__init__.py`, and `fullerene/cli.py` behind `--behavior`.
- Adjusted `fullerene/facets/echo.py` so empty user messages no longer force a `RECORD` proposal, letting the behavior layer return `WAIT` for truly empty input.
- Added `tests/test_behavior.py` for rule coverage, runtime integration, and CLI behavior; extended `tests/test_nexus_runtime.py` with an explicit priority-order test; documented the deterministic behavior decision in `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/knowledge/decisions.md`.

### 2026-04-25 (i)

- Tightened `fullerene/memory/inference.py` to infer the requested starter tags from lowercase content, including `correction`, and to score salience from user-message plus tag signals only.
- Updated `fullerene/memory/scoring.py` so retrieval-side event tags merge explicit metadata tags with deterministic content-inferred tags.
- Refactored `fullerene/facets/memory.py` to derive merged tags and salience explicitly before building each `MemoryRecord`.
- Refreshed `tests/test_memory.py` to cover the requested boss-email tags, correction-driven salience, clamp behavior, content-inferred storage, and retrieval preference.
- Updated `ai/project/architecture.md` and `ai/logs/SESSION_LOG.md` to reflect the current Memory v1 behavior and verification commands.

### 2026-04-25 (h)

- Added `fullerene/memory/inference.py` with deterministic tag rules (`infer_tags`, `merge_tags`) and salience scoring (`compute_salience`, `explain_salience`) for Memory v1.
- Wired `fullerene/facets/memory.py` to merge metadata-supplied tags with inferred tags (explicit tags retain priority), compute salience, and persist `metadata_tags` / `inferred_tags` / `salience_breakdown` for inspection.
- Added `explain_score` in `fullerene/memory/scoring.py` so retrieval rankings can be explained by component (keyword/tag/salience/recency); retrieval weights and bounds unchanged.
- Extended `tests/test_memory.py` with tag-inference, salience-scoring, MemoryFacet integration, and tag-favored retrieval tests.
- Updated `ai/project/architecture.md` (Memory v1 section + verified mapping note) and `ai/knowledge/glossary.md` (tag inference, salience, hard-rule-candidate).

### 2026-04-25 (g)

- Polished `fullerene/cli.py` so `MemoryFacet` is opt-in behind `--memory`; default CLI behavior remains `EchoFacet` only.
- Extended `tests/test_memory.py` with CLI-path coverage for `--memory` creating `memory.sqlite3` and the default path not requiring or creating the SQLite memory DB.
- Updated `ai/operations/commands.md` with the memory smoke command.

### 2026-04-25 (f)

- Added `fullerene/memory/` with `MemoryRecord`, `MemoryType`, deterministic scoring helpers, and `SQLiteMemoryStore` as the canonical Memory v0 store.
- Added `fullerene/facets/memory.py` and exported `MemoryFacet` so Nexus and the CLI can persist episodic memories and retrieve a bounded relevant/recent memory set.
- Added `tests/test_memory.py` covering record round-tripping, SQLite schema/init, CRUD and retrieval behavior, bounded facet loading, and Nexus integration with `EchoFacet`.
- Updated `ai/project/architecture.md`, `ai/knowledge/decisions.md`, `ai/knowledge/glossary.md`, and `ai/logs/SESSION_LOG.md` for Memory v0 and the SQLite-canonical-memory decision.

### 2026-04-25 (e)

- Hardened `fullerene/nexus/runtime.py` so facet exceptions are isolated into sanitized `FacetResult` error entries instead of aborting the event loop.
- Extended `tests/test_nexus_runtime.py` to verify failed facets do not crash Nexus, other facets still run, and persistence still happens.

### 2026-04-25 (d)

- Added initial runtime package under `fullerene/` with `nexus/`, `facets/`, `state/`, `cli.py`, and `__main__.py`.
- Added `tests/test_nexus_runtime.py` and documented the runnable commands in `ai/operations/commands.md`.
- Updated harness terminology from Conductor to Nexus in `ai/project/overview.md`, `ai/project/architecture.md`, `ai/apps/runtime-cli.md`, `ai/operations/verification.md`, and `ai/knowledge/glossary.md`.
- Recorded the v0 runtime persistence decision in `ai/knowledge/decisions.md` and session handoff notes in `ai/logs/SESSION_LOG.md`.

### 2026-04-25 (c)

- Harness cleanup: reformatted `ai/**/*.md`; shortened `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.cursor/rules/fullerene-harness.mdc`, `.claude/rules/fullerene-harness.md`, `.codex/AGENTS.md`.
- Added harness-vs-repo truth rule and consolidated working agreements in `ai/MEMORY.md`; added `ai/README.md`.
- Clarified `ai/project/architecture.md` as harness vocabulary, not an implementation spec.

### 2026-04-25 (b)

- Tool-native hooks: `.cursor/rules/fullerene-harness.mdc`, `.claude/rules/fullerene-harness.md`, `.codex/AGENTS.md`; documented in `ai/MEMORY.md`; gitignore `.claude/settings.local.json`.

### 2026-04-25

- Initial `ai/` harness and root `CLAUDE.md`, `AGENTS.md`, `.cursorrules`.
