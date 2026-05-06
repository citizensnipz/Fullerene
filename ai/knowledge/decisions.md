# Architecture and process decisions (ADR-style)

Record decisions that matter later, not every small edit.

## 2026-05-06 - Behavior v2.2 modularization keeps v2.1 semantics

- **Status:** accepted
- **Context:** Behavior v2.1 concentrated lexical extraction, adapter extraction, scoring, confidence, trace, and learning helpers inside `fullerene/facets/behavior.py`, which made evolution and focused tests harder.
- **Decision:** Introduce `fullerene/behavior/` modules (`models`, `lexical`, `signals`, `scoring`, `confidence`, `trace`, `learning`) and preserve `BehaviorFacet` as the public contract in `fullerene/facets/behavior.py`. Keep decisions constrained to `WAIT/RECORD/ASK/ACT`, deterministic-only behavior, no in-facet LLM calls, and compatibility metadata including `response_template`.
- **Consequences:** Behavior internals are now testable by module boundary while runtime contract remains stable for Nexus/CLI/tests. `response_intent` becomes explicit metadata while `response_template` remains compatibility output.

## 2026-05-06 - Goals v1 remains explicit and non-executing

- **Status:** accepted
- **Context:** Needed richer goal dynamics (reinforcement/lifecycle/progress metadata) without introducing goal hierarchy, decomposition, conflict resolution, or autonomous execution behavior.
- **Decision:** Extend the existing goal model/store/facet pipeline in place with additive schema fields (`reinforcement_score`, activation/completion/evidence/block/stale metadata), deterministic lifecycle helpers (`pause`, `resume`, `complete`, `update status`), and deterministic score formulas for ranking and pressure contribution. Keep goals as bias-only signals for Context/Behavior/Planner/LPB metadata.
- **Consequences:** Goals remain inspectable persistent records that influence attention and planning pressure without executing actions or expanding into Goals v2/v3 scope.

## 2026-05-06 - Executor v1 uses explicit skill manifest + sandbox file ops

- **Status:** accepted
- **Context:** Executor v0 handled only internal actions. v1 needed sandboxed file operations, explicit external skill invocation boundaries, approval gating, and planner/verifier-friendly metadata without introducing shell/network/git/MCP/dynamic loading.
- **Decision:** Keep `ExecutorFacet` as the same facet boundary and upgrade `fullerene/executor/` with explicit registry primitives (`registry.py`, `skills.py`, `models.py` additions), sandbox path resolver (`sandbox.py`), sandboxed file skills (`file_ops.py`), separate bounded file-op audit log, per-step approval pending/timeout statuses, fail-closed live policy checks, and stronger execution record metadata (`skill_name`, `skill_version`, action/target/policy/approval/sandbox fields). External skills are available only when explicitly registered via code/config, never dynamic imports from plan payload.
- **Consequences:** Executor remains deterministic and inspectable while widening capability only to sandbox file operations and explicit registered skills. No shell/network/git/MCP/parallel/rollback/plugin ecosystem expansion was introduced. Planner and verifier now receive richer step feedback and consistency metadata.

## 2026-05-06 - Verifier v1.5 extends deterministic artifact consistency checks

- **Status:** accepted
- **Context:** Verifier v1 already performed deterministic schema checks, but Behavior v2.2 traces, Context v2 packets, World Model v1 lifecycle artifacts, and model/output metadata needed tighter cross-artifact consistency validation and clearer retry/escalation recommendations.
- **Decision:** Keep existing VerifierFacet architecture and add v1.5 deterministic validators in-place: Behavior trace consistency rules, Context v2 packet checks, World Model v1 belief/edge checks, output metadata + generic unsupported capability/source-claim detection, generic skill/executor validator hooks, and policy/planner/executor consistency checks. Keep retry/escalation recommendation-only (no automatic retry loop, no new facet, no LLM judge).
- **Consequences:** Fullerene surfaces more structured warnings/errors for inconsistent or unsafe runtime artifacts while preserving inspectability, deterministic behavior, Nexus integration, and existing Verifier v1 guardrails.

## 2026-05-06 - World Model v1 adds deterministic belief lifecycle and contradiction pressure

- **Context:** World Model v0 only supported explicit belief rows/relevance lookup, which was insufficient for tracking evolving confidence, contradictions, provenance, and unresolved pressure signals from conflicting claims.
- **Decision:** Upgrade to World Model v1 in-place (same facet/runtime boundary): expand `Belief`/SQLite schema with lifecycle/provenance fields (`sources`, normalized key, belief type, support/contradiction counts, last support/contradiction/update event IDs, priority, metadata), add deterministic formation/update logic in `WorldModelFacet`, implement contradiction/redundancy heuristics, emit contradiction/uncertainty pressure signal payloads for LPB ingestion, and persist lightweight write-time belief edges.
- **Consequences:** Fullerene can now represent what it believes, how confident it is, and when claims conflict without deleting historical beliefs. Context and behavior receive richer belief metadata/signals while staying deterministic and inspectable. No new facet, no LLM extraction, no clustering, and no graph traversal were introduced in v1.

## Entry format

```markdown
## YYYY-MM-DD - Short title

- **Status:** proposed | accepted | superseded
- **Context:**
- **Decision:**
- **Consequences:**
- **Supersedes:** (if any)
```

## Decisions

## 2026-05-06 - Context v2.1 adds deterministic working-memory reference anchors

- **Status:** accepted
- **Context:** Context v2 preserved bounded same-session working-memory transcript turns, but generic referential follow-ups (`that`, `it`, `one`, `there`) still lacked compact continuity structure for downstream Behavior/prompt grounding.
- **Decision:** Extend existing Context internals only (no new facet/subsystem) with deterministic bounded continuity extraction from working-memory turns: add JSON-safe `ReferenceAnchor` and `ConversationContinuity` metadata, derive conservative anchors/topic terms/unresolved references without LLM calls, expose continuity fields on `ContextWindow.metadata` + `ContextFacet` state/metadata, add optional bounded `conversation_continuity` context item, and render a concise prompt-grounding section only when useful.
- **Consequences:** Referential continuity is now inspectable and available to Behavior/model grounding while preserving Context/Memory boundaries. No Dialogue subsystem, no Memory v3 graph traversal, and no summarization pipeline were introduced.

## 2026-05-06 - Context v2 uses pressure/relevance deterministic assembly (no Memory v3 scope)

- **Status:** accepted
- **Context:** Context v1/v1.5 provided dynamic bounded packets plus working-memory inclusion, but selection still leaned on broad bounded slices. We needed more deterministic relevance/pressure-based selection while preserving immediate dialogue continuity and avoiding Memory v3/graph/LLM scope.
- **Decision:** Add `pressure_relevance_v2` as a third context strategy in existing Context/Memory wiring (no new facet). Context v2 protects current event + bounded same-session working-memory turns, scores remaining candidates with deterministic weighted components (`relevance`, `pressure`, `salience`, `recency`, `confidence`, `priority`), applies bounded budget selection and cutoffs, integrates compact LPB and attention signals, keeps long-term retrieval on Memory v2 hybrid retrieval, and emits richer inclusion/exclusion/stale trace metadata. CLI adds explicit context strategy/budget/cutoff flags.
- **Consequences:** Context packets become more inspectable and decision-oriented without changing Memory authority boundaries or introducing graph/community/summarization behavior. `dynamic_active_facets_v1` remains the conservative default; operators opt into v2 strategy explicitly.

## 2026-05-06 - Memory v2.5 working memory remains inside Memory/Context

- **Status:** accepted
- **Context:** Memory v2 improved long-term relevance, but immediate pronoun/reference continuity in multi-turn interactive dialogue still needed exact recent turns (for example, resolving "one" to "a name") without introducing Memory v3 graph/compression or a new Dialogue subsystem.
- **Decision:** Extend existing Memory/Context only. Add `memory_layer` (`working` | `long_term`) to memory rows with backward-compatible migration defaulting prior rows to `long_term`; add bounded session-scoped working-turn helpers (`add_working_turn`, `list_working_turns`, `prune_working_memory`) that store exact role/content/turn-index/session metadata; exclude working rows from long-term hybrid retrieval; include bounded working turns in context as `working_memory` items when `session_id` exists; add prompt `Recent conversation` grounding; wire interactive loop to maintain one stable `session_id`, store exact user/assistant visible turns, and prune per-session turns each cycle.
- **Consequences:** Immediate conversational continuity improves deterministically while preserving Memory v2 boundaries and inspectability. No new facet/module, no LLM summarization/compression, no automatic working→long-term promotion, no graph/community retrieval-time work.

## 2026-05-06 - Interactive Loop v0 transcript-first terminal UX

- **Status:** accepted
- **Context:** Interactive Loop v0 originally rendered every tick, which interleaved status redraws with in-progress typing and made line input difficult to use.
- **Decision:** Make transcript output the default for interactive mode: idle ticks stay silent unless explicitly enabled, status is printed after user submissions and stop/warning conditions, and `/status`, `/help`, `/quit` commands provide in-loop control. Add optional idle tick visibility with `--interactive-show-ticks` and cadence control with `--interactive-status-every`.
- **Consequences:** Interactive CLI becomes usable as a line-oriented conversation surface without introducing a full TUI dependency. Clear-screen redraw remains non-default/experimental and future renderer work stays out of scope.

## 2026-05-06 - LPB v1.1 idle-tick stabilization and latent-only pressure stop nuance

- **Status:** accepted
- **Context:** LPB v1 could self-reinforce during idle `SYSTEM_TICK` loops by repeatedly ingesting routine interrupt/context/learning echoes, causing latent pressure saturation and repeated `latent_pressure_total_high` ignition despite no genuinely new danger signal. Shared tick stop logic also treated latent-only saturation the same as critical danger pressure.
- **Decision:** Keep LPB and stop safety rails intact, but add conservative v1.1 controls: tick-aware LPB ingestion gating (`should_ingest_signal_on_tick`) with explicit allow-lists for critical verifier/policy/new signals and suppression for LPB/Nexus/Behavior/Learning echo loops on `SYSTEM_TICK`; faster idle decay for inactive entries; same-key tick reactivation dampening; damped bounded total-pressure aggregation weights; and stricter tick-time total ignition requirements. Update `TickStopTracker` so latent-only high pressure no longer trips the 5-tick danger stop, while verifier/policy/danger pressure still uses the short threshold.
- **Consequences:** Interactive/continuous/manual tick loops remain bounded and safe while avoiding premature 5-tick stops from latent-only idle saturation. Critical verifier/policy danger paths still escalate/stop quickly. LPB remains deterministic, inspectable, and non-autonomous.

## 2026-05-06 - Interactive Loop v0 (foreground conversational loop with bounded ticks)

- **Status:** accepted
- **Context:** Continuous Loop v0 supports repeated internal `SYSTEM_TICK` cycles, but operators also need to converse with Fullerene while the loop remains active in the same foreground CLI session.
- **Decision:** Add `fullerene/interactive/` (`models.py`, `input.py`, `renderer.py`, `runner.py`, `__init__.py`) and CLI flags `--interactive`, `--interactive-interval`, `--interactive-max-ticks`, `--interactive-no-clear`, `--interactive-allow-model`, `--interactive-no-expression`, `--interactive-stop-on-ask-user`. Interactive Loop v0 alternates tick processing and line-oriented user input; non-empty user lines become `USER_MESSAGE` events with `interactive_input` metadata and go through normal Nexus processing. Tick events keep manual-tick metadata, remain quiet by default, and keep model calls disabled.
- **Consequences:** Fullerene now stays active and conversational in a single foreground CLI loop without introducing daemon/background cognition, full TUI complexity, or autonomous tool execution. Model output in interactive mode is opt-in and restricted to user-message realization only; `SYSTEM_TICK` remains deterministic and model-free.

## 2026-05-06 - Continuous Loop v0 (foreground bounded loop, minimal surface)

- **Status:** accepted
- **Context:** Manual Tick Runner and Watch Mode already provide bounded repeated internal cycles, but operators need a straightforward foreground loop for repeated `SYSTEM_TICK` execution with minimal non-scrolling output and explicit stop conditions.
- **Decision:** Add `fullerene/continuous/` (`models.py`, `runner.py`, `renderer.py`) and CLI `--loop`, `--loop-interval`, `--loop-max-ticks`, `--loop-no-clear`, `--loop-allow-expression`, `--loop-stop-on-ask-user`, `--loop-json`. Continuous Loop v0 reuses tick summary + stop-rule plumbing, runs one `SYSTEM_TICK` per interval in foreground only, defaults to bounded runs (`max_ticks=100`), and shows only mode/pressure/text-status. Reject `--model` with `--loop` in v0.
- **Consequences:** Fullerene now has a deterministic, inspectable, bounded foreground loop that preserves state between ticks and avoids trace spam. Daemon/background mode, autonomous tool execution, richer TUI/face rendering, and generated prose remain explicit future work.

## 2026-05-06 - Watch Mode v0 (controlled manual ticks + terminal snapshots)

- **Status:** accepted
- **Context:** Operators need a human-readable, terminal-facing view of how Fullerene's pressure/latent pressure/expression/interrupt/presentation signals evolve across bounded `SYSTEM_TICK` cycles, without adding an always-on daemon, TUI/watch framework, or autonomous expression.
- **Decision:** Implement **`fullerene/watch/`** (`models.py`, `runner.py`, `renderer.py`) with `WatchConfig`, a JSON-serializable `WatchSnapshot`, and `run_watch_mode()` that reuses `fullerene/tick/runner.py` (`run_manual_ticks`) plus Presentation Vector v0. Add CLI flags `--watch`, `--watch-ticks`, `--watch-interval`, `--watch-clear`, `--watch-trace`, `--watch-json`. Render plain stdout text by default and JSON when requested. Never dump LLM/prose and never bypass Expression Gate.
- **Consequences:** Watch Mode v0 provides deterministic, inspectable terminal snapshots suitable for manual debugging and state inspection. Continuous-loop/watch daemon behavior and richer UI renderers remain future work.

## 2026-05-06 - Presentation Vector v0 (read-only UI projection)

- **Status:** accepted
- **Context:** Downstream surfaces need a stable, animation-friendly snapshot of “what Fullerene appears to be doing” without importing cognition into the UI layer or adding a thirteenth facet.
- **Decision:** Implement **`fullerene/presentation/`** (`models.py`, `mapping.py`, `vector.py`) with **`derive_presentation_vector`** deterministic priority/intensity rules, renderer-neutral hints only, JSON round-tripping, optional **`validate_presentation_vector_v0`** in **`fullerene/verifier/artifacts.py`**, Manual Tick summary / CLI **`--presentation`** embedding, and **no** mutations of runtime state or facet wiring.
- **Consequences:** ASCII/Rive/Electron/watch-mode consumers can subscribe to a stable packet later; always-on UI loops and asset rendering remain explicitly out of scope until separately specified.

## 2026-05-06 - Manual Tick Runner v0 (explicit SYSTEM_TICK sequences)

- **Status:** accepted
- **Context:** Operators need to observe whether Nexus/LPB/interrupts/expression/verifier state evolves across cycles without injecting user prompts, without adding an always-on daemon, watch UI, or LLM calls from the runner.
- **Decision:** Implement **`fullerene/tick/runner.py`** with `run_manual_ticks`, compact per-tick **`summarize_tick_record`**, JSON-serializable **`TickRunResult`**, and CLI flags **`--tick`**, **`--ticks N`** (default 1, hard cap **100**), **`--tick-reason`**, **`--tick-summary`**, **`--allow-tick-expression`**. Each tick is **`EventType.SYSTEM_TICK`** with **`content=""`** and metadata **`manual_tick`**, **`tick_index`** (1-based), **`tick_count`**, optional **`tick_reason`**, and **`suppress_expression`** default **true** (opt out with **`--allow-tick-expression`**). Reject **`--model`** with manual ticks so CLI text generation is never implied. Apply conservative **stop early** rules (consecutive very high `system_pressure`, repeated verifier-critical cycles, repeated expression `ask_user` with same source, internal-event overflow, runtime exceptions). **`--json`** emits **`{ "tick_run": … }`**; omit heavy **`records`** from JSON unless **`--debug`**. Watch mode / continuous loop v0 stays out of scope here.
- **Consequences:** Single-message CLI JSON shape unchanged when **not** using manual ticks. Full-facet tick runs may hit verifier-driven stop conditions sooner than requested tick counts; operators tune facets or state for longer runs.

## 2026-05-06 - Expression Gate v0 (recommend outbound expression only)

- **Status:** accepted
- **Context:** Nexus v2 can queue bounded internal interrupts and LPB can signal ignition without user-facing autonomy. The product needs a deterministic boundary between internal pressure and actionable outward messaging without turning Nexus into always-on orchestration or a prose generator.
- **Decision:** Implement **Expression Gate v0** under `fullerene/expression/` (not under `facets/`): deterministic scoring from signal map / LPB / interrupts / Behavior / Verifier / Policy / Context overload / Attention / Learning / event metadata; modes `silent` → `log_only` → `status_only` → `short_utterance` → `ask_user` with conservative defaults; persists `ExpressionBudgetState` (+ bounded history) in `facet_state["nexus"]["expression_gate"]`; merges `expression_*` facets into each cycle record + cycle trace metadata; Nexus appends Verifier **`validate_expression_gate_v0`** rows to the verifier facet’s `artifact_checks` post hoc. No LLM, no network threads, no final natural language, no tool execution from the gate.
- **Consequences:** User-facing autonomy remains off by default unless a downstream layer consumes recommendation metadata. Manual tick runners / watch TUIs remain future hooks; Nexus v2 `allowed_user_expression` stays permanently false because interrupt suppression is orthogonal to Gate recommendations.

## 2026-05-06 - Nexus v2 bounded interrupt arbitration and suppression (single-cycle)

- **Status:** accepted
- **Context:** Behavior v2, LPB v1, Verifier v1, Policy v1, Learning v1, Planner v1, and Attention v1 all emit signals that can recommend interrupts, retries, approvals, or ignition-like pressure relief, but Nexus v1 lacked a unified deterministic gate against internal-event chatter, duplicate interrupts, cooldown storms, or unsafe ACT implications.
- **Decision:** Implement Nexus v2 as a **purely internal**, **deterministic**, **single outer-call** arbitration layer (`fullerene/nexus/interrupts.py` + hooks in `fullerene/nexus/runtime.py`): extract JSON-serializable `InterruptCandidate` records after LPB updates, compute clamped additive priority with inspectable score components, run ordered suppression rules (cooldown keyed by deterministic `cooldown_key`, duplicate equivalents within a cycle, low-priority cutoff, context-overload downgrade for non-safety interrupts, suppression when policy denies the ACT execution path, user-expression permanently false, at most one internal winner, bounded queue depth), optionally queue one `INTERNAL`/`nexus_interrupt` event when no explicit facet `internal_events` are already present, persist cooldown and audit metadata under `facet_state["nexus"]` (no new DB), extend `cycle_trace` / record metadata with compact interrupt audit fields, and add Verifier v1 prior-cycle audit rows (`validate_nexus_interrupt_v2_audit`) without LLM judging. LPB ignition remains **candidate-only** (no speech, no execution).
- **Consequences:** Operators gain inspectable interrupt/suppression traces; internal routing stays bounded and policy-safe; always-on daemons, Expression Gate / user-visible speech, dynamic facet reorder, and autonomous tool use remain explicitly out of scope for this ADR.

## 2026-05-06 - Latent Pressure Buffer v1 is Nexus-owned signal infrastructure, not a facet

- **Status:** accepted
- **Context:** Fullerene already had a `latent_pressure` slot in pressure aggregation, but it was not a persistent inspectable subsystem and could not track unresolved pressure recurrence/decay across cycles.
- **Decision:** Implement LPB v1 under `fullerene/signals/latent_pressure/` as reusable support infrastructure (`LatentPressureEntry`, `LatentPressureResult`, `update_latent_pressure`) and integrate it directly in Nexus runtime after facet results and before final persistence. Persist LPB state under `facet_state["signals"]["latent_pressure"]`; expose LPB result metadata on `NexusRecord`; feed LPB `latent_pressure_total` into `CycleSignalMap` pressure components. LPB remains non-facet infrastructure and does not execute actions, emit user-facing speech, run autonomous loops, or perform LLM calls.
- **Consequences:** Latent pressure is now deterministic, bounded, JSON-inspectable, and persistent across cycles. Behavior/Nexus can consume LPB totals without adding a 13th facet. Deeper Attention/Context reinsertion behavior remains future work.

## 2026-05-06 - Verifier v1 is deterministic artifact/schema validation with retry hints only

- **Status:** accepted
- **Context:** Behavior v2 traces, Nexus v1 `CycleSignalMap` / cycle context, Planner v1 grounding metadata, Executor v0 results, and Learning v1 adjustment metadata needed a single deterministic validation layer beyond v0 structural/policy/act checks—without introducing LLM-as-judge, external fact lookup, benchmarks, fuzzy scoring, autonomous repair, policy invention, executor permission widening, or hidden retries.
- **Decision:** Promote Verifier to **v1** by adding `fullerene/verifier/artifacts.py` validators, `ArtifactSchemaCheck`, richer `FacetResult`/`VerificationSummary` metadata (`verifier_version`, `artifact_checks`, retry/escalation reason lists, `validation_codes`, `validated_artifact_kinds`, optional `downgraded_decision` hints), and a minimal Nexus `verifier_cycle_context` injection ahead of verifier facets; keep all checks deterministic and JSON-serializable; failures that make `ACT`/execution ambiguous downgrade conservatively (`ASK` or `RECORD` per artifact rules).
- **Consequences:** Operators and tests can introspect normalized validation rows per artifact kind; future runners may honor retry/escalation metadata without implying automatic retries inside Verifier. Verifier v2+ remains separate (eval harnesses / deeper policy still require explicit ADRs).

## 2026-05-06 - Learning v1 is a deterministic cross-facet feedback router without a Learning-owned store

- **Status:** accepted
- **Context:** Nexus v1 already emits `CycleSignalMap`, `cycle_trace`, and collects `cycle_learning_events`; Behavior v2 emits `behavior_decision_trace_v2`. Learning v0 only classified explicit feedback/execution/goal signals and did not consume those cross-facet artifacts.
- **Decision:** Implement Learning v1 as a deterministic, JSON-inspectable router: read Nexus preview keys (`current_cycle_signal_map`, `current_cycle_learning_events` injected at the start of the learning_signal phase), Behavior/Context/Memory facet state, and v0 signals; emit `cross_facet_routes` and extended `LearningResult.metadata` (`learning_version: v1`, consumed events, signal_sources, adjustment lists, reasons). Apply mutations only through explicit store APIs (`strengthen_memory_edge`, `update_belief_confidence` / `update_belief`, existing salience/goal updates). Behavior threshold and policy/executor permission changes remain proposal-only. No TD learning, meta-learning, LLM calls, background loops, unbounded graph scans, or new Learning SQLite DB.
- **Consequences:** Memory owns edge-weight/SQLite graph writes; World Model owns belief confidence and status flags; Learning coordinates signals and proposals. Nexus must expose current-cycle learning artifacts to facets in the same outer `process_event` call before Learning runs.
- **Supersedes:** n/a (v0 rules remain embedded in `build_learning_result` for compatibility)

## 2026-05-06 - Policy v1 deterministic permission evaluation (approval tokens + plan aggregation)

- **Status:** accepted
- **Context:** Policy needed to remain explicit and deterministic while becoming more expressive and inspectable: approval gating needed explicit, locally validated approval metadata; evaluation results needed a structured, JSON-serializable decision model including precedence trace; and planner plans/steps needed plan-level policy aggregation so Nexus could reliably downgrade to `ASK`/`RECORD` without bypassing Verifier.
- **Decision:** Upgrade `PolicyFacet` to Policy v1 by emitting a comprehensive `policy_evaluation` payload (status/effective_action/risk/target fields, approval token validity, fallbacks, reasons/warnings, and `rule_precedence_trace`), adding richer structured matching over explicit action/plan-step context (wildcard targets and deterministic metadata equality/conditions matching), implementing deterministic approval-token semantics for `require_approval` conversions, and adding `execute_plan` aggregation to produce `plan_policy_evaluation` plus per-step lists (`denied_step_ids`, `approval_required_step_ids`, `allowed_step_ids`, `preferred_step_ids`). Preserve v0 deny-wins and do not add LLM policy inference, learned rule acceptance, policy self-modification, external authorization, or Learning-driven automatic rule updates.
- **Consequences:** Operators and tests can inspect why a decision happened (including which rule was decisive), approval requirements are deterministic and locally validated, and plan-level safety decisions become uniform across Nexus/Behavior/Executor integration. Policy v2+ remains separate for deeper learning/eval harnesses or any future broadened capability work behind explicit ADRs.

## 2026-05-05 - Nexus v1 deeper pass keeps single-cycle scope and makes Nexus the canonical signal aggregator

- **Status:** accepted
- **Context:** Behavior v2, Attention v1, Context v1, Planner v1, Verifier v0, and Learning v0 already emitted useful metadata, but cross-facet signals were spread across facet payloads and pressure aggregation was implicit/averaged rather than a single canonical inspectable map owned by Nexus.
- **Decision:** Keep Nexus v1 as a bounded single-cycle runtime (no always-on daemon loop, no sleep/wake cycle, no dynamic suppression/reordering, no autonomous expression), and add a canonical per-cycle `CycleSignalMap` in Nexus. Compute `system_pressure` from explicit components (`event_pressure + attention_pressure + latent_pressure + contradiction_pressure + context_overload_pressure + interrupt_pressure`, clamped to `[0.0, 1.0]`) and persist `pressure_components`. Collect facet `learning_event` payloads into cycle metadata/trace only. Allow Behavior interrupt recommendations to queue at most one inspectable internal event candidate per outer `process_event` call, with no same-call recursive expansion. Persist compact `cycle_trace` and last-cycle nexus state artifacts for inspection/debugging.
- **Consequences:** Nexus now owns canonical cross-facet signal aggregation and exposes deterministic inspectable cycle artifacts without widening authority boundaries. Behavior v2 traces are captured but not directly consumed by Learning yet. Nexus v2 can later build continuous orchestration on top of these inspectable artifacts without rewriting this bounded v1 contract.

## 2026-05-05 - Memory v2 adds optional embedding index, role/domain classification, hybrid retrieval, and bounded write-time edges

- **Status:** accepted
- **Context:** Memory v1 retrieval scored by keyword overlap, tag overlap, salience, and recency, which let prior repeated questions outrank useful preference/fact context (for example, "What kind of book should I read next?" outranking "I like to read sci-fi novels and non-fiction autobiographies"). Fullerene needed better grounding without collapsing memory into an embedding monolith, breaking offline tests, or leaping into a Memory v3 linked-graph subsystem.
- **Decision:** Implement Memory v2 as additive layers over Memory v1. Add deterministic role classification (`preference`, `fact`, `question`, `task`, `feedback`, `outcome`, `unknown`) and domain inference (for example, `reading_books`, `outdoors_water`, `project_software`, `task_work`) at write time, persisted on `MemoryRecord.role` / `MemoryRecord.domain` and on new `role` / `domain` SQLite columns. Add an optional `EmbeddingProvider` protocol with a `DeterministicHashEmbeddingProvider` for offline tests/fallback and an opt-in `OllamaEmbeddingProvider`; persist vectors in a non-authoritative `memory_embeddings` table. Add `hybrid_retrieve_relevant` with the score `0.35 * semantic + 0.20 * tag + 0.15 * salience + 0.10 * recency + 0.10 * domain_match + 0.10 * role_bonus - role_penalty`, deterministic query-intent detection (`recommendation`, `planning`, `factual`, `unknown`), and inspectable `explain_hybrid_score` breakdowns. Compute bounded write-time edges (`same_goal`, `tag_overlap`, `temporal_proximity`, `keyword_similarity`, `semantic_similarity`, `same_domain`, `role_related`) into a new `memory_edges` table over a candidate set capped at recent + high-salience + same-domain memories; do not traverse edges at retrieval time. Update Context v1 dynamic assembly and the CLI prompt builder to consume hybrid retrieval, surface `retrieval_strategy`, `query_intent`, `included_memory_roles`, `included_memory_domains`, and per-memory `score_breakdown`, and annotate prompt-grounded memories with `role` and `domain`.
- **Consequences:** Recommendation/advice queries now retrieve preference memories ahead of repeated prior questions, with inspectable reasons. SQLite remains the source of truth, embeddings are an optional index, and missing or failing providers degrade to deterministic v1 retrieval. Bounded write-time edges give v3 a starting point without paying graph-traversal cost in v2. Out of scope for v2: LLM summarization, retrieval-time graph traversal, Leiden / community detection, learned weights, mandatory external services, and any hardcoded per-topic special cases.

## 2026-05-05 - Attention v1 broadcasts the winning focus item as metadata/state only

- **Status:** accepted
- **Context:** Attention v0 could score and rank focus candidates, but it stopped at top-N metadata. Context and later facets had no stable, minimal way to observe the winning attention item, repeated focus pressure, or close-score competition without a broader Nexus rewrite.
- **Decision:** Upgrade Attention to v1 by keeping the existing deterministic fixed-weight scorer, then adding deterministic bottom-up vs top-down classification, winner broadcast, close-score conflict signaling, and bounded winner history. The winning item is stored as `last_attention_broadcast` plus related ids/mode/history on attention facet state and is mirrored in `AttentionResult` metadata. Context may expose that broadcast as an `attention` context item, and Behavior may read prior-cycle top-down broadcasts for a small confidence-only bias. Attention v1 does not mutate Memory, Goals, World Model, Policy, Planner, or Executor stores; it does not trigger another Nexus cycle; and it does not add ignition, refractory, predictive, learned, or cluster-based attention mechanics.
- **Consequences:** Fullerene now has a minimal broadcast contract for the attention winner without widening authority boundaries. Later work can build on the existing inspectable broadcast/conflict/history packet for ignition thresholds, pressure integration, or richer routing, but v1 remains deterministic, bounded, and state-first rather than mutation-first.

## 2026-05-04 - Goal intent creation and Context v1 both deduplicate active goals deterministically

- **Status:** accepted
- **Context:** After Context v1 landed, reused state directories could still surface multiple active goals that meant the same thing but came from repeated intent phrasing such as `I should remember to finish Fullerene` and `remember to finish Fullerene`. That polluted working context before Planner v1.
- **Decision:** Add deterministic goal normalization and duplicate handling before planning. Goal-intent creation now normalizes descriptions by lowercasing, trimming, removing punctuation, collapsing whitespace, and stripping common leading intent phrases; when an active goal with the same normalized description already exists, the runtime updates that goal instead of creating a new row, merging tags and keeping the higher priority. Context v1 also deduplicates already-persisted active goals before exposing them in the working packet, preferring highest priority, then most recent `updated_at`, then newest `created_at`, with a conservative high-overlap keyword fallback only for near-duplicate comparison.
- **Consequences:** Active-goal context is cleaner and more stable across repeated runs on the same state directory, and Planner can consume a single deterministic goal packet instead of competing duplicate goals. The tradeoff is that exact normalized matching remains the canonical rule, so broader semantic duplicates still require future deliberate design rather than ad-hoc fuzzy matching.

## 2026-05-04 - Context v1 is a deterministic bounded working packet assembled from active state

- **Status:** accepted
- **Context:** Fullerene could already persist memories, goals, world beliefs, policies, and later facet outputs, but the old Context layer exposed only a static recent-memory slice. That left later responses under-grounded even when active goals or beliefs already existed in persistent stores.
- **Decision:** Implement `ContextAssemblyConfig`, `DynamicContextAssembler`, and upgraded `ContextFacet` support for `dynamic_active_facets_v1`. Context v1 always includes the current event, then bounded active goals, relevant/recent memories, active beliefs, a compact policy summary, and optional compact planner / executor / attention / affect / learning summaries when available. The assembly is deterministic, read-only, store-bounded, deduplicated, and visible both to later facet state and to the CLI model prompt builder through a concise working-context summary. It does not use embeddings, RAG, LLM summarization, graph traversal, pressure-based compression, or self-editing context mutation.
- **Consequences:** Fullerene now has a real working-context layer that can ground later behavior, planning, and response generation in persisted state without architecture rewrites or opaque prompt stuffing. Future context work can improve deterministic ranking and selection, but the canonical v1 mechanism remains a bounded assembly packet, not a retrieval or summarization subsystem.

## 2026-04-28 - Affect v0 is a deterministic internal VAD + novelty observer with no downstream influence

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, planning, execution, learning, and attention signals, but it still lacked a narrow place to summarize its own internal state from those signals without collapsing affect into user-emotion detection or behavior modulation.
- **Decision:** Implement `fullerene/affect/` plus `AffectFacet` as a deterministic internal observation layer. Affect v0 derives `valence`, `arousal`, `dominance`, and `novelty` from existing runtime signals only, records an inspectable `AffectState` and `AffectResult`, and may keep a short bounded history in Nexus facet state. It never proposes `ACT`, never mutates other stores, and does not modulate memory, attention, planning, execution, policy, or expression yet.
- **Consequences:** Fullerene now has an explicit affect boundary and a traceable data-collection layer for future work. Later salience modulation, affect-tagged memories, appraisal logic, or pressure integration can reuse the same inspectable artifacts, but v0 remains observational: no emotion recognition, no sentiment model, no prosody, and no learned affect inference.

## 2026-04-28 - Attention v0 is a deterministic metadata-only focus scorer with no broadcast

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, planner, executor, and learning signals, but it still lacked a narrow place to score what should receive foreground focus before any future broadcast mechanism exists.
- **Decision:** Implement `fullerene/attention/` plus `AttentionFacet` as a fixed-weight, inspectable scoring layer. Attention v0 always considers the current event, can score additional memory / goal / belief / execution candidates when those signals are already available, emits `AttentionItem` and `AttentionResult` metadata, selects top-N focus items, and never proposes `ACT`. It does not broadcast a winner, mutate Context, or own a learned model.
- **Consequences:** Fullerene now has an explicit spotlight boundary that is separate from behavior, planning, execution, and context assembly. Future attention broadcast, ignition, and learned weighting can build on the same inspectable artifacts without widening v0 into a decision-maker.

## 2026-04-27 - Learning v0 is a stateless feedback bus with apply-or-propose adjustment records

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, behavior, policy, planner, executor, and verifier layers, but it still lacked a narrow place to close the feedback loop after outcomes were observed.
- **Decision:** Implement `fullerene/learning/` plus `LearningFacet` as a stateless signal processor and feedback bus. Learning v0 classifies explicit user feedback, executor outcomes, and goal lifecycle metadata through deterministic rules only; it emits `LearningSignal`, `AdjustmentRecord`, and `LearningResult` payloads; it may apply only minor safe nudges to goal priority or memory salience when an existing store already supports that change cleanly; and it emits proposals instead of silently applying larger or unsupported changes. Learning owns no canonical persistent state of its own.
- **Consequences:** Fullerene now has an explicit post-outcome adjustment boundary without collapsing learning into memory, goals, behavior, or policy ownership. Future richer learning can build on the same traceable artifacts, but v0 remains conservative: no self-modification, no policy mutation, no executor permission changes, and no model calls.

## 2026-04-27 - Executor v0 is an internal-only execution layer with dry-run default and no partial execution

- **Status:** accepted
- **Context:** Fullerene already had deterministic planning, policy, and verification layers, but it still lacked a controlled place to carry out approved internal actions without collapsing execution into planner logic or opening external side effects.
- **Decision:** Implement `fullerene/executor/` plus `ExecutorFacet` as a deterministic execution boundary. Executor v0 accepts inspectable plans, validates every step before mutation, defaults to dry-run, executes only supported internal actions, and halts on the first blocked, approval-gated, high-risk, unsupported, or malformed step. It does not run shell commands, network calls, git actions, arbitrary file operations, tool execution, or LLM-driven skills.
- **Consequences:** Fullerene now has a concrete "hands" layer that remains narrow, inspectable, and policy-constrained. Planner still proposes, Policy still decides allowed vs approval-required vs denied, and Verifier still validates safety. Future executor versions can widen capability behind explicit roadmap and trust-boundary decisions instead of silently expanding v0.

## 2026-04-27 - Planner v0 is a deterministic plan-proposal facet with policy-filtered steps and no execution

- **Status:** accepted
- **Context:** Fullerene already had deterministic memory, goals, world model, behavior, policy, verifier, and context layers, but it still lacked a first-class place to propose ordered next steps without collapsing planning into behavior or introducing tool execution.
- **Decision:** Implement `fullerene/planner/` plus `PlannerFacet` as a deterministic, model-free plan proposal layer. Planner v0 triggers only on explicit plan requests or when a high-priority active goal is present and the current event explicitly asks for next steps. It emits inspectable `Plan` / `PlanStep` objects with deterministic confidence, simple pressure handling, step-level risk labels, and policy-filtered approval/blocking metadata. It does not execute steps or call tools.
- **Consequences:** Fullerene now has a distinct planning boundary that remains transparent and testable in v0. Future richer planning can build on the same plan objects, but execution stays separate and must still pass policy plus verifier checks.

## 2026-04-27 - Verifier v0 runs deterministic post-decision checks and may downgrade unsafe ACT decisions

- **Status:** accepted
- **Context:** Fullerene already had deterministic Behavior and Policy layers, but it still needed a final inspectable safeguard that could validate the aggregated decision trace itself before records were persisted.
- **Decision:** Implement `fullerene/verifier/` plus `VerifierFacet` as a deterministic post-decision validation pass. Nexus now runs normal facets, aggregates an initial decision, then runs verifier checks against the event, facet results, initial decision, and configured state-dir metadata. If verifier finds an unsafe or structurally invalid `ACT`, it may downgrade that decision to `ASK` or `RECORD` before persistence.
- **Consequences:** Fullerene gains a small internal test runner for its own decision process without adding model calls or autonomous execution. Verifier metadata is persisted as a normal `FacetResult`, so callers can inspect which checks failed and why. Future executor work can rely on both Policy and Verifier guardrails instead of behavior heuristics alone.

## 2026-04-26 - Policy v0 allows internal state CRUD and requires approval for external side effects

- **Status:** accepted
- **Context:** Fullerene already had explicit memory, goals, world model, and behavior layers, but it still needed a deterministic, inspectable permission boundary before future executor/tool work lands.
- **Decision:** Implement `PolicyFacet` plus `fullerene/policy/` SQLite-backed explicit rule storage. Treat explicit policy rows as the canonical store for user/system rules, and enforce two built-in sandbox defaults in the runtime: internal CRUD inside the configured state directory is allowed by default, while modeled external side effects require approval by default unless an explicit allow rule matches. Explicit `deny` rules still override everything, and policy can downgrade `ACT` to `ASK` or `RECORD`.
- **Consequences:** Fullerene can manage its own memories, goals, beliefs, policy rows, and local runtime files inside `state-dir` without unnecessary approval prompts, while shell/network/message/git/tool/file-side-effect actions stay blocked behind approval unless the user explicitly allows them. Future executor work now has a deterministic policy gate to consult instead of relying on behavior heuristics alone.

## 2026-04-25 - Behavior v0 is deterministic, model-free, and inspectable

- **Status:** accepted
- **Context:** After Nexus and Memory v1, Fullerene needed its first explicit decision-policy layer for whether an event should `WAIT`, `RECORD`, `ASK`, or `ACT`, but the v0 scope still excludes LLM planning, graph reasoning, tool execution, and autonomous risky behavior.
- **Decision:** Implement `BehaviorFacet` as a deterministic rules layer over event type/content, explicit metadata, deterministic tags, deterministic salience, and optional passed-through memory metadata. The facet emits an inspectable proposal with reasons and confidence metadata. `ACT` remains only a typed proposal for future execution; the runtime still performs no tool execution.
- **Consequences:** Behavior decisions stay testable, debuggable, and source-visible in v0. Future model-based planning or confidence estimation can be layered on later, but they should not replace the canonical deterministic behavior path without another explicit architecture decision.

## 2026-04-25 - SQLite is the canonical Fullerene memory store

- **Status:** accepted
- **Context:** Memory v0 needed persistent, inspectable storage without loading one giant text file, and without introducing embeddings, vector infrastructure, or model-based summarization.
- **Decision:** Store canonical memory records in SQLite under the local state directory. Treat SQLite rows as source of truth for episodic and semantic memory data, with working memory derived from bounded recent retrieval. Any future embeddings, vector search, or compressed machine representations are retrieval indexes or caches, not the authoritative memory store.
- **Consequences:** Memory stays deterministic, queryable, and testable in v0. Future retrieval layers can be added without changing what counts as remembered data, but richer indexing and schema evolution will need migration discipline.

## 2026-04-25 - Nexus v0 is a small dataclass-based runtime with explicit local persistence

- **Status:** accepted
- **Context:** Fullerene needed its first real runtime slice without provider coupling, database setup, or autonomous side effects.
- **Decision:** Implement the central loop as `Nexus` / `NexusRuntime` with stdlib dataclasses for events, facet results, decisions, records, and state. Use a pluggable state store with in-memory and file-backed JSON/JSONL implementations under an explicit local state directory.
- **Consequences:** The runtime is easy to test and inspect, but intentionally simple. SQLite, model backends, and real action execution stay out of scope until later slices need them.

## Suggested future entries

- SQLite migration strategy
- Model-port abstraction if provider integration is added
- Event shape changes between Nexus and richer facets
