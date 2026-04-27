# AI-facing changelog

Changes that matter for future AI coding sessions (layout, commands, invariants). Not a substitute for user-facing release notes.

## Instructions

- Newest first.
- One line or bullet per change set; prefer paths over long prose.

## Changelog

### 2026-04-27 (learning)

- Added `fullerene/learning/` with `LearningSignal`, `AdjustmentRecord`, `LearningResult`, deterministic feedback/execution/goal signal classifiers, and conservative apply-or-propose adjustment logic (`alpha = 0.1`, minor nudges only, major changes become proposals).
- Added `fullerene/facets/learning.py` and exported `LearningFacet`; it is a stateless feedback processor that never proposes `ACT`, records inspectable learning metadata, and only applies supported minor salience/priority nudges.
- Updated `fullerene/cli.py` with `--learning`, ordered after executor so Learning can observe current-cycle execution results through existing facet state updates, and reused configured memory/goal stores instead of creating a learning-owned store.
- Added `update_memory_salience` to `fullerene/memory/store.py` as the smallest safe mutable memory API needed for Learning v0; goal priority nudges reuse the existing `GoalStore.update_goal` path.
- Added `tests/test_learning.py` for learning models, signal classification, adjustment rules, store integration, full-stack Nexus integration, executor-observed outcomes, and CLI smoke coverage.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, `ai/knowledge/decisions.md`, and `ai/logs/SESSION_LOG.md` for Learning v0, the v1-v3 roadmap, and the stove rule.

### 2026-04-27 (executor hardening)

- Hardened `fullerene/executor/runner.py` so Executor v0 now uses explicit `ActionType -> handler` registration, refuses unsupported or external `target_type` values with distinct reason codes, never infers actions from step descriptions, and no longer backfills `emit_event` payloads from prose.
- Updated `fullerene/executor/models.py` so `ExecutionRecord.action_type` can stay unset when execution fails before a supported handler is resolved, avoiding misleading fallback action labels.
- Tightened `fullerene/cli.py` so `--live` only flips execution mode when `--execute-plan` is also present.
- Expanded `tests/test_executor.py` with coverage for unknown action/target failures, explicit reason-code assertions, no description-based execution inference, no-bypass `--live` behavior, and halt-on-first-failure semantics.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/logs/SESSION_LOG.md` to document explicit handlers, loud unknown-action failure, dry-run default, and the fact that live mode does not broaden Executor v0 permissions.

### 2026-04-27 (executor)

- Added `fullerene/executor/` with `ActionType`, `ExecutionMode`, `ExecutionRecord`, `ExecutionResult`, `ExecutionStatus`, and `InternalActionExecutor` for conservative internal-only plan execution.
- Added `fullerene/facets/executor.py` and exported `ExecutorFacet`; it executes only when `execute_plan` is requested, defaults to dry-run, reads the latest planner payload from facet state, never proposes `ACT`, and records structured execution metadata.
- Updated `fullerene/facets/planner.py` so planner state updates now persist `last_plan`, and updated `fullerene/planner/builder.py` so current planner-generated steps declare explicit `noop` action types for safe Executor v0 handling.
- Updated `fullerene/cli.py` with `--executor`, `--execute-plan`, and `--live`; planner now feeds executor in CLI runs without a broader Nexus rewrite.
- Added `tests/test_executor.py` covering executor models, runner refusal behavior, no-partial-execution guarantees, facet behavior, Nexus integration, and CLI coverage.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, `ai/knowledge/decisions.md`, and `ai/logs/SESSION_LOG.md` for Executor v0, the Helmet Rule, and the explicit v1-v3 execution roadmap.

### 2026-04-27 (planner)

- Added `fullerene/planner/` with `Plan`, `PlanStep`, `PlanStatus`, `PlanStepStatus`, `RiskLevel`, and `DeterministicPlanBuilder` for deterministic, inspectable plan proposals.
- Added `fullerene/facets/planner.py` and exported `PlannerFacet`; it triggers on explicit planning requests or high-priority active goals, derives simple pressure from event metadata / salience / behavior confidence, emits ordered steps, and never executes actions.
- Updated `fullerene/cli.py` with `--planner` and `--pressure`; when planner is enabled it reuses any configured goals/world/policy stores rather than creating separate planner-only storage.
- Added verifier `PlanSafetyCheck` in `fullerene/verifier/checks.py` so high-risk planner steps must require approval and blocked steps cannot appear inside an approved plan payload.
- Added `tests/test_planner.py` for planner models, builder behavior, policy filtering, facet behavior, Nexus integration, verifier safety, and CLI coverage.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, `ai/knowledge/decisions.md`, and `ai/logs/SESSION_LOG.md` for Planner v0 plus explicit v1-v3 future roadmap notes.

### 2026-04-27 (context polish)

- Confirmed `ContextFacet` export remains available from `fullerene.facets` and added explicit coverage in `tests/test_context.py`.
- Polished `ai/project/architecture.md` wording so the current implemented runtime list explicitly includes Memory, Goals, World Model, Behavior, Policy, Verifier, Context, and Echo without implying Context v1-v3 behavior is implemented.

### 2026-04-27 (context)

- Added `fullerene/context/` with `ContextItem`, `ContextItemType`, `ContextWindow`, `STATIC_RECENT_EPISODIC_V0`, and `StaticContextAssembler`.
- Added `fullerene/facets/context.py` and exported `ContextFacet`; it assembles a bounded static context window from `MemoryStore.list_recent(limit, memory_type=episodic)` and returns inspectable window metadata without planning or retrieval heuristics.
- Updated `fullerene/cli.py` with `--context` and `--context-window-size`; `--memory` and `--context` now share one memory-store instance, and `ContextFacet` is ordered before `MemoryFacet` so the context window reflects prior stored episodic records rather than the current event.
- Added `tests/test_context.py` for context model serialization, assembler bounds/filtering, empty-store handling, Nexus integration, and CLI multi-run behavior.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/logs/SESSION_LOG.md` for Context v0, with v1-v3 explicitly marked future-only.

### 2026-04-27 (verifier)

- Added `fullerene/verifier/` with `VerificationStatus`, `VerificationSeverity`, `VerificationResult`, `VerificationSummary`, and deterministic checks for decision shape, facet-result shape, policy compliance, and conservative `ACT` approval requirements.
- Added `fullerene/facets/verifier.py` and exported `VerifierFacet`; it runs after initial Nexus aggregation and persists inspectable verifier metadata as a normal `FacetResult`.
- Updated `fullerene/nexus/runtime.py` so post-decision verifier facets can downgrade unsafe or structurally invalid `ACT` decisions to `ASK` or `RECORD` before persistence.
- Updated `fullerene/cli.py` with `--verify`.
- Added `tests/test_verifier.py` for verifier models, checks, Nexus integration, persisted metadata, and CLI smoke coverage.
- Updated `fullerene/world_model/store.py` to use `PRAGMA locking_mode = EXCLUSIVE`, matching the other SQLite stores on this filesystem and fixing multi-store integration test stability.

### 2026-04-26 (repository state / world model)

- Renamed the gitignored workspace tree from `scratch/` to `state/`; `fullerene/scratch.py` is now `fullerene/workspace_state.py` with `WORKSPACE_STATE_DIR_NAME`, `DEFAULT_STATE_DIR` (`state/.fullerene-state`), and `workspace_state_root()` to avoid clashing with the `fullerene.state` store package.
- `tests/test_world_model.py` now uses `state/world_model_storage/…` (parallel to `mem_storage/` and `goals_storage/`) so world-model test DBs are grouped and the folder can be deleted as a unit.
- **Agent note:** all local ephemera go under `state/`; do not reintroduce `scratch/` or root-level `world_model_storage/`.

### 2026-04-26 (scratch)

- Introduced `fullerene/scratch.py` with `DEFAULT_STATE_DIR` (`scratch/.fullerene-state`) and `scratch_root()`; CLI `--state-dir` now defaults to that path so local runs stay under one gitignored tree.
- Pointed all unit tests that used repo-root or system-temp paths at `scratch/` (`mem_storage/`, `goals_storage/`, `.test-*` prefixes) via `scratch_root()`.
- Replaced the older `.gitignore` list of per-pattern runtime folders with a single `scratch/` entry; updated `ai/project/architecture.md`, `ai/project/conventions.md`, `ai/knowledge/glossary.md`, `ai/operations/commands.md`, `ai/operations/verification.md`, and `ai/apps/runtime-cli.md` accordingly.
- **Agent note:** do not create new dot-directories or parallel storage folders at the project root; add ephemeral paths under `scratch/` and reuse `fullerene.scratch`.

### 2026-04-26 (n)

- Added `fullerene/policy/` with `PolicyRule`, policy enums, and `SQLitePolicyStore` as the canonical explicit Policy v0 store in `policy.sqlite3`.
- Added `fullerene/facets/policy.py` and exported `PolicyFacet` so Nexus can evaluate explicit rules plus built-in sandbox defaults for internal state CRUD and external approval requirements.
- Updated `fullerene/nexus/runtime.py` so policy `denied` results force `RECORD` and policy `approval_required` results force `ASK`, even when another facet proposed `ACT`.
- Updated `fullerene/cli.py` with `--policy`, `--policy-db`, and explicit metadata-driven `create_policy` support; normal messages still do not infer policies automatically.
- Added `tests/test_policy.py` covering policy model/store CRUD, facet evaluation precedence, behavior/runtime integration, CLI DB creation, and metadata-driven policy creation.
- Updated `fullerene/policy/store.py` to use `PRAGMA locking_mode = EXCLUSIVE`, matching the existing workspace-safe SQLite pattern used by the other working stores on this filesystem.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, `ai/knowledge/decisions.md`, and `ai/logs/SESSION_LOG.md` for Policy v0 and the internal-state sandbox decision.

### 2026-04-26 (m)

- Added `fullerene/world_model/` with `Belief`, `BeliefStatus`, `BeliefSource`, and `SQLiteWorldModelStore` as the canonical World Model v0 store in `world.sqlite3`.
- Added `fullerene/facets/world_model.py` and exported `WorldModelFacet` so Nexus can emit deterministic belief-relevance signals from tag overlap, keyword overlap, and belief confidence.
- Updated `fullerene/facets/behavior.py` so behavior can read inspectable world-model signals and apply a small confidence boost without changing the core decision rules.
- Updated `fullerene/cli.py` with `--world`, `--world-db`, and explicit metadata-driven `create_belief` support; belief creation remains explicit and deterministic.
- Added `tests/test_world_model.py` covering belief model/store behavior, world-model facet scoring, CLI DB creation, metadata-driven belief creation, behavior integration, and Memory+Goals+WorldModel+Behavior runtime integration.
- Updated `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/logs/SESSION_LOG.md` for World Model v0, including the explicit note that automatic belief inference is not implemented.

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
