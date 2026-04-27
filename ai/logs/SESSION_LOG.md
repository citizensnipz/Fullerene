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

### 2026-04-27 - Executor v0 hardening pass

- **Context:** Tighten Executor v0 so it stays explicit and conservative without adding any new action capabilities.
- **Done:** Reworked `fullerene/executor/runner.py` to use explicit action-handler maps, removed action inference from `target_type` except explicit `noop`, removed description-based `emit_event` payload fallback, split unsupported failures into clearer reason codes (`unsupported_action_type`, `unsupported_target_type`, `unsupported_live_action`, `execution_failed`), and kept halt-on-first-failure behavior intact; updated `fullerene/executor/models.py` so unresolved failure records can leave `action_type` empty instead of guessing; tightened `fullerene/cli.py` so `--live` only flips execution mode when `--execute-plan` is also set; expanded `tests/test_executor.py` with strict reason-code, no-inference, no-bypass, and no-partial-execution coverage.
- **Verified:** `python -m unittest tests.test_executor -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --planner --executor --execute-plan --content "make a plan for this"`; `python -m fullerene --planner --executor --execute-plan --live --content "make a plan for this"`
- **Next:** If Planner v1 needs richer live execution, add explicit internal action payloads instead of widening Executor v0 inference rules.
- **Blockers:** None

### 2026-04-27 - Executor v0 internal-only execution

- **Context:** Add Executor v0 so planner output can be carried through a controlled execution layer without opening external side effects or blurring planner/policy/verifier boundaries.
- **Done:** Added `fullerene/executor/` with `ExecutionRecord`, `ExecutionResult`, enums, and `InternalActionExecutor`; added `fullerene/facets/executor.py`; updated `PlannerFacet` state updates so later facets can read the current plan payload without a Nexus API rewrite; marked planner-generated steps as explicit `noop` actions for safe v0 execution; wired `ExecutorFacet` into `fullerene/facets/__init__.py`, `fullerene/__init__.py`, and `fullerene/cli.py` behind `--executor`, `--execute-plan`, and `--live`; added `tests/test_executor.py`; updated architecture, glossary, decisions, and changelog docs with Executor v0, the Helmet Rule, and the v1-v3 roadmap.
- **Verified:** `python -m unittest tests.test_executor -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --planner --executor --execute-plan --content "make a plan for this"`; `python -m fullerene --planner --executor --execute-plan --live --content "make a plan for this" --metadata "{\"target_type\": \"shell\"}"`
- **Next:** Decide whether Planner v1 should emit richer explicit internal action payloads beyond `noop`, so Executor can graduate from plan-trace execution to targeted state mutations without overloading v0 heuristics.
- **Blockers:** Full-suite runtime is long enough that the first `unittest discover` run hit the harness timeout even though the suite finished cleanly; rerun with a larger timeout when the harness cap matters.

### 2026-04-27 - Planner v0 deterministic plan proposals

- **Context:** Add Planner v0 as a deterministic, inspectable facet that can propose ordered steps without executing anything or adding a full pressure subsystem.
- **Done:** Added `fullerene/planner/` (`models.py`, `builder.py`, `__init__.py`) plus `fullerene/facets/planner.py`; wired `PlannerFacet` into `fullerene/facets/__init__.py`, `fullerene/__init__.py`, and `fullerene/cli.py` behind `--planner` and `--pressure`; planner can read optional goals/world/policy stores, derive pressure from event metadata or behavior confidence, generate ordered plans, and mark blocked / approval-required steps deterministically; added verifier `PlanSafetyCheck`; added `tests/test_planner.py`; updated architecture, glossary, decisions, and changelog docs for Planner v0 and the v1-v3 roadmap.
- **Verified:** `python -m unittest tests.test_planner -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --planner --content "make a plan for this"`; `python -m fullerene --planner --pressure 0.8 --content "what are the steps?"`; `python -m fullerene --planner --goals --world --policy --content "what are the next steps?"`
- **Next:** If Planner v1 lands, keep the current `Plan` / `PlanStep` data model stable and add richer context or multi-goal selection around it instead of widening Planner v0 into execution.
- **Blockers:** None

### 2026-04-27 - Context v0 static recent episodic window

- **Context:** Add Context v0 as a simple static working-context facet without dynamic assembly, embeddings, salience filtering, or planner behavior.
- **Done:** Added `fullerene/context/` (`models.py`, `assembler.py`, `__init__.py`) plus `fullerene/facets/context.py`; wired `ContextFacet` into `fullerene/cli.py` behind `--context` and `--context-window-size`; reused the same memory SQLite store object when `--memory` and `--context` run together; added `tests/test_context.py`; updated architecture and glossary docs for Context v0 and deferred v1-v3 roadmap notes.
- **Verified:** `python -m unittest tests.test_context -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --memory --context --content "first context memory" --state-dir state/.smoke-context-v0`; `python -m fullerene --memory --context --content "second context memory" --state-dir state/.smoke-context-v0`; `python -m fullerene --memory --context --context-window-size 2 --content "show recent context" --state-dir state/.smoke-context-v0`
- **Next:** If future behavior or planner work needs more than recent episodic memory, add Context v1 as deterministic multi-facet assembly rather than widening v0.
- **Blockers:** None

### 2026-04-27 - Verifier v0 deterministic post-decision checks

- **Context:** Add a deterministic verifier that validates Nexus decisions and facet outputs after aggregation, with special safety checks around `ACT`.
- **Done:** Added `fullerene/verifier/` (`models.py`, `checks.py`, `__init__.py`) plus `fullerene/facets/verifier.py`; wired `NexusRuntime` to run verifier facets after initial decision aggregation and allow them to downgrade unsafe `ACT` decisions before persistence; added CLI `--verify`; added `tests/test_verifier.py`; updated `fullerene/world_model/store.py` to use `PRAGMA locking_mode = EXCLUSIVE` so multi-store integration tests run cleanly on this filesystem.
- **Verified:** `python -m unittest tests.test_verifier -v`; `python -m unittest tests.test_policy.PolicyRuntimeIntegrationTests.test_nexus_runs_with_memory_goals_world_behavior_policy_and_echo_facets -v`
- **Next:** Consider whether verifier should eventually emit a first-class decision-trace object instead of only facet metadata once executor/planner work adds more intermediate artifacts.
- **Blockers:** None

### 2026-04-26 - Rename `scratch/` → `state/`, world model under `state/world_model_storage/`

- **Context:** Align the gitignored tree with the "state" name; keep world model tests from writing at repo root; mirror `mem_storage` / `goals_storage` with `world_model_storage` so it can be deleted wholesale.
- **Done:** `fullerene/workspace_state.py` replaces `fullerene/scratch.py` (`state/`, `DEFAULT_STATE_DIR` = `state/.fullerene-state`, `workspace_state_root()`). `.gitignore` and harness docs use `state/`. Glossary "Scratch" → "Repository state". `test_world_model` temp dirs: `state/world_model_storage/.test-world-model-*`.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`
- **Next:** N/A
- **Blockers:** None

### 2026-04-26 - Centralize runtime and test output under scratch/

- **Context:** Ad-hoc folders and dot-directories (`.test-behavior-*`, `mem_storage/`, `goals_storage/`, `.smoke-*`, system temp for policy/world tests) were polluting the repo root; needed one conventional location and updated generators/readers.
- **Done:** Added `fullerene/scratch.py` (`DEFAULT_STATE_DIR`, `scratch_root()`); CLI default `--state-dir` is `scratch/.fullerene-state`; all tests and docs now target `scratch/`.`.gitignore` uses a single `scratch/` rule. Glossary, architecture, conventions, commands, verification, runtime-cli, and this changelog were updated. Older session log lines may still mention historical `.smoke-*` or `.fullerene-state` at repo root—new work should use the gitignored `state/` tree and `fullerene.workspace_state` (see log entry *Rename `scratch/` → `state/`*).
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`
- **Next:** N/A
- **Blockers:** None

### 2026-04-26 - Policy v0 deterministic permissions

- **Context:** Add a persistent Policy layer so Fullerene can evaluate explicit permission rules and sandbox boundaries separately from memory, goals, world beliefs, and behavior.
- **Done:** Added `fullerene/policy/` with `PolicyRule`, policy enums, and `SQLitePolicyStore`; added `fullerene/facets/policy.py`; wired `PolicyFacet` into `fullerene/cli.py` behind `--policy` with `--policy-db` override and explicit metadata-driven `create_policy`; updated `fullerene/nexus/runtime.py` so policy `denied` / `approval_required` results override an `ACT` proposal safely; added `tests/test_policy.py`; updated architecture, glossary, and decisions docs.
- **Verified:** `python -m unittest tests.test_policy -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; direct `python -m fullerene --policy ...` smoke runs verified through exact `python -m fullerene` argv lists executed via Python `subprocess` because PowerShell mangles inline JSON quoting for `--metadata`.
- **Next:** Add CLI inspection/update flows for policy rows (`list`, `enable`, `disable`, `delete`, `reprioritize`) and decide whether world-model SQLite should also adopt the workspace-safe `PRAGMA locking_mode = EXCLUSIVE` path for consistency.
- **Blockers:** PowerShell inline JSON quoting makes the literal `--metadata '{"..."}'` examples unreliable on this machine; the CLI logic itself is verified.

### 2026-04-26 - World Model v0 explicit beliefs

- **Context:** Add persistent World Model v0 so Fullerene can store explicit beliefs about reality separately from episodic memory, without adding inference, reasoning, embeddings, or planning.
- **Done:** Added `fullerene/world_model/` with `Belief`, `BeliefStatus`, `BeliefSource`, and `SQLiteWorldModelStore`; added `fullerene/facets/world_model.py`; exported world-model types through `fullerene/` and `fullerene/facets/`; updated `fullerene/cli.py` with `--world`, `--world-db`, and explicit metadata-driven `create_belief` support; extended `BehaviorFacet` with inspectable world-alignment confidence signals; added `tests/test_world_model.py`; updated architecture and glossary docs.
- **Verified:** `python -m unittest tests.test_world_model -v`; `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --world --content "SQLite is the canonical memory store" --metadata '{"create_belief": true}' --state-dir .smoke-world-v0`; `python -m fullerene --world --content "Should we change memory storage?" --state-dir .smoke-world-v0`
- **Next:** Add explicit inspection/update commands for beliefs (`list`, `stale`, `contradict`, `retire`, `re-confidence`) so users do not need to open SQLite directly to manage world state.
- **Blockers:** None.

### 2026-04-26 - Goals v0 deterministic store and facet

- **Context:** Add persistent Goals v0 so Fullerene can keep explicit directional state without adding planning, LLM calls, embeddings, or automatic goal generation.
- **Done:** Added `fullerene/goals/` with `Goal`, status/source enums, and `SQLiteGoalStore`; added `fullerene/facets/goals.py`; exported goals through `fullerene/` and `fullerene/facets/`; updated `fullerene/cli.py` with `--goals`, `--metadata`, and explicit `create_goal` handling; adjusted `fullerene/nexus/runtime.py` so later facets can observe earlier facet state updates within the same event; extended `BehaviorFacet` with inspectable goal-alignment confidence signals; added `tests/test_goals.py`.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --goals --content "track my tasks" --metadata '{"create_goal": true}' --state-dir .smoke-goals-v0`; `python -m fullerene --goals --content "work on my tasks" --state-dir .smoke-goals-v0`; `python -m fullerene --goals --behavior --content "work on my tasks" --state-dir .smoke-goals-v0`
- **Next:** Add an inspect/update CLI surface for goals (`list`, `pause`, `complete`, `reprioritize`) so users can manage persistent goals without editing SQLite directly.
- **Blockers:** None.

### 2026-04-26 - Behavior v0 integration polish

- **Context:** Polish Behavior v0 integration and stale harness language, and align CLI memory path behavior with state-dir defaults.
- **Done:** Updated `fullerene/cli.py` so `--memory` defaults SQLite to `<state-dir>/memory.sqlite3` when `--memory-db` is omitted; kept explicit `--memory-db` override behavior. Added CLI coverage in `tests/test_memory.py` for state-dir default and override path, plus a behavior-only CLI assertion in `tests/test_behavior.py` that no memory DB is created. Refreshed `ai/project/architecture.md`, `ai/knowledge/glossary.md`, and `ai/MEMORY.md` to clarify Nexus naming, Behavior Facet v0 scope, deterministic decision policy semantics, inspectable confidence metadata, and no tool execution/model integration in v0.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --memory --behavior --content "don’t ever skip my boss emails"`; `python -m fullerene --behavior --content "what should I do next?"`
- **Next:** Decide whether to keep historical "Conductor" references in docs once external docs are fully migrated to Nexus-only naming.
- **Blockers:** None.

### 2026-04-25 - Behavior Facet v0

- **Context:** Add the first deterministic behavior/decision policy layer after Nexus + Memory v1, without LLM planning, graph reasoning, tool execution, or autonomous side effects.
- **Done:** Added `fullerene/facets/behavior.py` with deterministic `WAIT` / `RECORD` / `ASK` / `ACT` policy rules, inspectable confidence/reason metadata, and optional memory-signal awareness; wired `--behavior` into `fullerene/cli.py`; exported `BehaviorFacet`; made `EchoFacet` ignore empty user messages so behavior-driven `WAIT` can win cleanly; added `tests/test_behavior.py`; added explicit runtime decision-priority constant in `fullerene/nexus/runtime.py`; updated architecture / glossary / decisions docs.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`
- **Next:** Decide whether future executor/planner work should consume `BehaviorFacet` metadata directly or whether a small executor-intent schema should sit between behavior and execution.
- **Blockers:** None.

### 2026-04-25 - Memory v1 deterministic scoring completion

- **Context:** Finish Memory v1 so tag extraction is content-driven and salience is derived from deterministic tag signals only, without embeddings, RAG, LLM calls, or prosody.
- **Done:** Updated `fullerene/memory/inference.py` to add `correction` inference and the requested salience formula; updated retrieval-side event tag extraction in `fullerene/memory/scoring.py`; refactored `fullerene/facets/memory.py` to derive tags/salience explicitly before storing; refreshed `tests/test_memory.py`; updated `ai/project/architecture.md` and `ai/logs/CHANGELOG_AI.md`.
- **Verified:** `python -m unittest discover -s tests -p "test_*.py" -v`; `python -m fullerene --content "don't ever skip my boss emails"`
- **Next:** If the CLI should surface memory inference by default, decide whether `--memory` should remain opt-in or whether a dedicated explain/output flag should be added.
- **Blockers:** None.

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
