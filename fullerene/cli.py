"""Minimal CLI for exercising the Nexus runtime."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from fullerene.continuous import ContinuousLoopConfig, run_continuous_loop
from fullerene.context import ContextAssemblyConfig
from fullerene.interactive import InteractiveLoopConfig, run_interactive_loop
from fullerene.facets import (
    AffectFacet,
    AttentionFacet,
    BehaviorFacet,
    ContextFacet,
    EchoFacet,
    ExecutorFacet,
    GoalsFacet,
    LearningFacet,
    MemoryFacet,
    PlannerFacet,
    PolicyFacet,
    VerifierFacet,
    WorldModelFacet,
)
from fullerene.goals import (
    Goal,
    GoalSource,
    GoalStatus,
    SQLiteGoalStore,
    find_matching_active_goal,
    normalize_goal_description,
)
from fullerene.memory import (
    SQLiteMemoryStore,
    build_embedding_provider,
    infer_tags,
    merge_tags,
    normalize_tags,
)
from fullerene.models import ModelAdapter, ModelAdapterError, OllamaAdapter
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.policy import (
    PolicyRule,
    PolicySource,
    SQLitePolicyStore,
    coerce_policy_rule_type,
    coerce_policy_source,
    coerce_policy_target_type,
)
from fullerene.presentation import derive_presentation_vector
from fullerene.tick.runner import TICK_HARD_CAP, TickRunResult, build_tick_event_metadata, run_manual_ticks
from fullerene.workspace_state import DEFAULT_STATE_DIR
from fullerene.state import FileStateStore
from fullerene.world_model import Belief, BeliefSource, SQLiteWorldModelStore
from fullerene.watch import WatchConfig, run_watch_mode


FULL_PRESET_FLAGS = (
    "memory",
    "context",
    "goals",
    "world",
    "behavior",
    "policy",
    "planner",
    "executor",
    "learning",
    "attention",
    "affect",
    "verify",
)
TEXT_RESPONSE_TEMPLATES = {
    "status_report": (
        "I'm running a local runtime cycle: recording the event, updating state, "
        "checking relevant facets, and deciding whether anything needs action."
    ),
    "clarification_needed": "I need a bit more context before I can act on that.",
    "clarify_recommendation_preferences": (
        "What preferences should I use, and what purpose should the recommendation serve?"
    ),
    "grounded_response_available": (
        "I found relevant internal context and can answer from that grounding."
    ),
    "next_steps_available": (
        "I can propose next steps from the current goal and planner state."
    ),
}
GOAL_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (
        re.compile(
            r"^\s*i\s+need\s+to\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        re.compile(
            r"^\s*i\s+should\s+remember\s+to\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        re.compile(
            r"^\s*remember\s+to\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        re.compile(
            r"^\s*remember\s+that\s+(?P<description>.+?)\s+is\s+important[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        re.compile(
            r"^\s*(?P<description>.+?)\s+is\s+(?:important|needed|necessary)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        re.compile(
            r"^\s*we\s+should\s+focus\s+on\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        re.compile(
            r"^\s*my\s+goal\s+is\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        re.compile(
            r"^\s*i\s+want\s+to\s+(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        re.compile(
            r"^\s*we\s+(?:need|must)\s+(?:to\s+)?(?P<description>.+?)[.!?\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a single event through the Fullerene Nexus runtime."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional prompt content. Equivalent to --content when --content is omitted.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Enable all implemented runtime facets for this run.",
    )
    parser.add_argument(
        "--latent-pressure",
        action="store_true",
        help=(
            "Enable latent pressure support metadata in runtime output. "
            "LPB infrastructure is integrated into Nexus and persists regardless."
        ),
    )
    parser.add_argument(
        "--expression-gate",
        action="store_true",
        help=(
            "Append compact Expression Gate v0 recommendation line to concise CLI "
            "output (--json/--debug already include full NexusRecord metadata)."
        ),
    )
    parser.add_argument(
        "--tick",
        action="store_true",
        help=(
            "Run SYSTEM_TICK event(s) instead of a user message. Explicit manual "
            "internal cycles only (not a daemon or scheduler)."
        ),
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1,
        metavar="N",
        help=f"With --tick, process N SYSTEM_TICK events in sequence (default 1, max {TICK_HARD_CAP}).",
    )
    parser.add_argument(
        "--tick-reason",
        default=None,
        help="Optional reason stored on each tick in metadata['tick_reason'].",
    )
    parser.add_argument(
        "--tick-summary",
        action="store_true",
        help="After a multi-tick run, print one compact summary line per tick.",
    )
    parser.add_argument(
        "--allow-tick-expression",
        action="store_true",
        help=(
            "Do not set metadata suppress_expression on manual ticks; Expression "
            "Gate may surface user-facing recommendation flags when scoring allows."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run Interactive Loop v0 (foreground SYSTEM_TICK + USER_MESSAGE loop).",
    )
    parser.add_argument(
        "--interactive-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="With --interactive, tick interval (default 1.0; clamped).",
    )
    parser.add_argument(
        "--interactive-max-ticks",
        type=int,
        default=1000,
        metavar="N",
        help="With --interactive, max SYSTEM_TICK cycles (default 1000; clamped).",
    )
    parser.add_argument(
        "--interactive-no-clear",
        action="store_true",
        help=(
            "Compatibility flag. Interactive mode defaults to transcript output "
            "(no screen redraw)."
        ),
    )
    parser.add_argument(
        "--interactive-clear",
        action="store_true",
        help="With --interactive, enable experimental clear-screen redraw mode.",
    )
    parser.add_argument(
        "--interactive-allow-model",
        action="store_true",
        help="With --interactive, allow --model for USER_MESSAGE output only.",
    )
    parser.add_argument(
        "--interactive-no-expression",
        action="store_true",
        help="With --interactive, suppress expression recommendations on SYSTEM_TICK.",
    )
    parser.add_argument(
        "--interactive-stop-on-ask-user",
        action="store_true",
        help="With --interactive, stop when expression mode asks user.",
    )
    parser.add_argument(
        "--interactive-show-ticks",
        action="store_true",
        help="With --interactive, print compact tick status lines while idle.",
    )
    parser.add_argument(
        "--interactive-status-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "With --interactive-show-ticks, print idle tick status every N ticks "
            "(0 means every tick)."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session id used for interactive working-memory continuity.",
    )
    parser.add_argument(
        "--working-memory-context-turns",
        type=int,
        default=8,
        metavar="N",
        help="Recent working-memory turns included in context (default 8).",
    )
    parser.add_argument(
        "--working-memory-turns",
        type=int,
        default=20,
        metavar="N",
        help="Working-memory turns retained per session after pruning (default 20).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run Continuous Loop v0 (foreground, bounded SYSTEM_TICK loop).",
    )
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="With --loop, sleep interval between ticks (default 1.0; clamped).",
    )
    parser.add_argument(
        "--loop-max-ticks",
        type=int,
        default=100,
        metavar="N",
        help="With --loop, maximum ticks before stopping (default 100; clamped).",
    )
    parser.add_argument(
        "--loop-no-clear",
        action="store_true",
        help="With --loop, print one compact line per tick (no screen clear).",
    )
    parser.add_argument(
        "--loop-allow-expression",
        action="store_true",
        help="With --loop, allow Expression Gate recommendations to be surfaced.",
    )
    parser.add_argument(
        "--loop-stop-on-ask-user",
        action="store_true",
        help="With --loop, stop early when expression mode requests ask_user.",
    )
    parser.add_argument(
        "--loop-json",
        action="store_true",
        help="With --loop, emit JSON result instead of live text rendering.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run bounded manual SYSTEM_TICK cycles and render compact watch snapshots.",
    )
    parser.add_argument(
        "--watch-ticks",
        type=int,
        default=10,
        metavar="N",
        help="With --watch, process N SYSTEM_TICK cycles (default 10, clamped).",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="With --watch, sleep SECONDS between rendered ticks (0 disables).",
    )
    parser.add_argument(
        "--watch-clear",
        action="store_true",
        help="With --watch, clear screen between rendered ticks.",
    )
    parser.add_argument(
        "--watch-trace",
        action="store_true",
        help="With --watch, include compact trace fragments when available.",
    )
    parser.add_argument(
        "--watch-json",
        action="store_true",
        help="With --watch, emit JSON watch_run output instead of text.",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Emit Presentation Vector v0 (compact lines and/or summaries; read-only projection)."
        ),
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable the SQLite-backed MemoryFacet for this run.",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help=(
            "Enable the ContextFacet for this run. The default strategy is the "
            "dynamic working-context assembler; static recent-memory mode remains "
            "available with --context-strategy static."
        ),
    )
    parser.add_argument(
        "--context-strategy",
        choices=("static", "dynamic", "pressure_relevance_v2"),
        default="dynamic",
        help="Context strategy used when --context is enabled.",
    )
    parser.add_argument(
        "--context-max-items",
        type=int,
        default=16,
        help="Maximum total context items for pressure_relevance_v2.",
    )
    parser.add_argument(
        "--context-max-goals",
        type=int,
        default=3,
        help="Maximum number of active goals included by dynamic context assembly.",
    )
    parser.add_argument(
        "--context-max-memories",
        type=int,
        default=None,
        help=(
            "Maximum number of memories included by dynamic context assembly. "
            "Defaults to --context-window-size when omitted."
        ),
    )
    parser.add_argument(
        "--context-max-beliefs",
        type=int,
        default=5,
        help="Maximum number of beliefs included by dynamic context assembly.",
    )
    parser.add_argument(
        "--context-salience-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum salience required for memories included by dynamic context "
            "assembly."
        ),
    )
    parser.add_argument(
        "--context-min-relevance",
        type=float,
        default=0.15,
        help="Minimum relevance/final score for pressure_relevance_v2 inclusion.",
    )
    parser.add_argument(
        "--context-min-pressure",
        type=float,
        default=0.20,
        help="Minimum pressure score for LPB pressure-based inclusion.",
    )
    parser.add_argument(
        "--affect",
        action="store_true",
        help="Enable deterministic internal affect-state observation for this run.",
    )
    parser.add_argument(
        "--affect-history-size",
        type=int,
        default=20,
        help="Maximum number of recent affect states retained by --affect.",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Enable the deterministic AttentionFacet for this run.",
    )
    parser.add_argument(
        "--attention-top-n",
        type=int,
        default=3,
        help="Maximum number of focus items emitted by --attention.",
    )
    parser.add_argument(
        "--attention-history-size",
        type=int,
        default=20,
        help="Maximum number of recent attention broadcasts retained by --attention.",
    )
    parser.add_argument(
        "--behavior",
        action="store_true",
        help="Enable the deterministic BehaviorFacet for this run.",
    )
    parser.add_argument(
        "--goals",
        action="store_true",
        help="Enable the SQLite-backed GoalsFacet for this run.",
    )
    parser.add_argument(
        "--world",
        action="store_true",
        help="Enable the SQLite-backed WorldModelFacet for this run.",
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Enable the SQLite-backed PolicyFacet for this run.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable deterministic post-decision verification for this run.",
    )
    parser.add_argument(
        "--verifier",
        action="store_true",
        dest="verify",
        help="Alias for --verify.",
    )
    parser.add_argument(
        "--planner",
        action="store_true",
        help="Enable the deterministic PlannerFacet for this run.",
    )
    parser.add_argument(
        "--executor",
        action="store_true",
        help="Enable the deterministic ExecutorFacet for this run.",
    )
    parser.add_argument(
        "--learning",
        action="store_true",
        help="Enable the stateless LearningFacet for this run.",
    )
    parser.add_argument(
        "--execute-plan",
        action="store_true",
        help="Request plan execution through ExecutorFacet for this run.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute plans live instead of the default dry-run simulation.",
    )
    parser.add_argument(
        "--executor-live",
        action="store_true",
        help="Alias for --live when using executor.",
    )
    parser.add_argument(
        "--executor-sandbox-dir",
        default=None,
        help="Override executor sandbox directory (still sandbox-constrained).",
    )
    parser.add_argument(
        "--executor-list-skills",
        action="store_true",
        help="List registered executor skills and exit.",
    )
    parser.add_argument(
        "--event-type",
        choices=[event_type.value for event_type in EventType],
        default=EventType.USER_MESSAGE.value,
        help="The kind of event to process.",
    )
    parser.add_argument(
        "--content",
        default=None,
        help="Event content for user message or system note events.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full NexusRecord JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional text model adapter, for example ollama:gemma3:4b.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the full NexusRecord JSON for debugging.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON object attached to the event metadata.",
    )
    parser.add_argument(
        "--feedback",
        choices=("positive", "negative"),
        default=None,
        help="Convenience shortcut for event.metadata['feedback'].",
    )
    parser.add_argument(
        "--target-memory-id",
        default=None,
        help="Convenience shortcut for event.metadata['target_memory_id'].",
    )
    parser.add_argument(
        "--target-goal-id",
        default=None,
        help="Convenience shortcut for event.metadata['target_goal_id'].",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help=(
            "Optional deterministic pressure override clamped to 0.0-1.0 and "
            "shared by planner, attention, and affect."
        ),
    )
    parser.add_argument(
        "--novelty",
        type=float,
        default=None,
        help=(
            "Optional deterministic novelty override clamped to 0.0-1.0 and "
            "shared by attention and affect."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help="Local directory for the runtime snapshot and append-only log.",
    )
    parser.add_argument(
        "--memory-db",
        default=None,
        help=(
            "SQLite path used by --memory and --context runs. "
            "Defaults to <state-dir>/memory.sqlite3 when omitted."
        ),
    )
    parser.add_argument(
        "--context-window-size",
        type=int,
        default=5,
        help=(
            "Compatibility window size for static context and the default memory "
            "cap for dynamic context."
        ),
    )
    parser.add_argument(
        "--memory-embeddings",
        action="store_true",
        help=(
            "Enable Memory v2 embedding-index storage and hybrid retrieval. "
            "Defaults to the deterministic in-process provider unless "
            "--embedding-model overrides it. Embeddings remain optional; "
            "missing or failing providers fall back to deterministic v1 retrieval."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Optional embedding model spec, for example 'deterministic' or "
            "'ollama:nomic-embed-text'. Implies --memory-embeddings when set."
        ),
    )
    parser.add_argument(
        "--goals-db",
        default=None,
        help=(
            "SQLite path used by --goals runs. "
            "Defaults to <state-dir>/goals.sqlite3 when omitted."
        ),
    )
    parser.add_argument(
        "--world-db",
        default=None,
        help=(
            "SQLite path used by --world runs. "
            "Defaults to <state-dir>/world.sqlite3 when omitted."
        ),
    )
    parser.add_argument(
        "--policy-db",
        default=None,
        help=(
            "SQLite path used by --policy runs. "
            "Defaults to <state-dir>/policy.sqlite3 when omitted."
        ),
    )
    return parser


def _cli_build_nexus_runtime(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    event: Event,
) -> NexusRuntime:
    """Construct ``NexusRuntime`` from CLI flags; ``event`` seeds goal/world/policy hooks."""
    state_dir = Path(args.state_dir)
    store = FileStateStore(state_dir)
    content = event.content
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    facets: list[Any] = []
    memory_store: SQLiteMemoryStore | None = None
    goal_store: SQLiteGoalStore | None = None
    world_store: SQLiteWorldModelStore | None = None
    policy_store: SQLitePolicyStore | None = None
    if args.memory or args.context:
        memory_db_path = (
            Path(args.memory_db) if args.memory_db else state_dir / "memory.sqlite3"
        )
        memory_store = SQLiteMemoryStore(memory_db_path)
    if args.goals:
        goals_db_path = (
            Path(args.goals_db) if args.goals_db else state_dir / "goals.sqlite3"
        )
        goal_store = SQLiteGoalStore(goals_db_path)
        _create_goal_from_metadata(goal_store, content=content, metadata=metadata)
        _create_goal_from_intent(goal_store, content=content, metadata=metadata)
    if args.world:
        world_db_path = (
            Path(args.world_db) if args.world_db else state_dir / "world.sqlite3"
        )
        world_store = SQLiteWorldModelStore(world_db_path)
        _create_belief_from_metadata(world_store, event=event)
    if args.policy:
        policy_db_path = (
            Path(args.policy_db) if args.policy_db else state_dir / "policy.sqlite3"
        )
        policy_store = SQLitePolicyStore(policy_db_path)
        try:
            _create_policy_from_metadata(
                policy_store,
                content=content,
                metadata=metadata,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.context:
        context_max_memories = (
            args.context_max_memories
            if args.context_max_memories is not None
            else args.context_window_size
        )
        context_config = ContextAssemblyConfig(
            max_goals=args.context_max_goals,
            max_items_total=max(1, int(args.context_max_items)),
            max_long_term_memories=context_max_memories,
            max_memories=context_max_memories,
            max_beliefs=args.context_max_beliefs,
            min_relevance_score=_clamp_unit(args.context_min_relevance),
            min_pressure_score=_clamp_unit(args.context_min_pressure),
            include_working_memory=True,
            include_lpb=True,
            include_attention=True,
            include_policy=True,
            include_world_model=True,
            include_goals=True,
            include_recent_signals=True,
            max_working_turns=max(0, int(args.working_memory_context_turns)),
            max_working_memory_turns=max(0, int(args.working_memory_context_turns)),
            salience_threshold=_clamp_unit(args.context_salience_threshold),
            strategy=args.context_strategy,
        )
        facets.append(
            ContextFacet(
                memory_store,
                goal_store=goal_store,
                world_model_store=world_store,
                policy_store=policy_store,
                window_size=args.context_window_size,
                strategy=args.context_strategy,
                config=context_config,
            )
        )
    if args.memory:
        embedding_provider = None
        if args.memory_embeddings or args.embedding_model:
            spec = args.embedding_model or "deterministic"
            embedding_provider = build_embedding_provider(spec)
        facets.append(
            MemoryFacet(memory_store, embedding_provider=embedding_provider)
        )
    if args.goals:
        facets.append(GoalsFacet(goal_store))
    if args.world:
        facets.append(WorldModelFacet(world_store))
    if args.behavior:
        facets.append(BehaviorFacet())
    if args.policy:
        facets.append(PolicyFacet(policy_store, state_dir=state_dir))
    if args.planner:
        facets.append(
            PlannerFacet(
                goal_store=goal_store,
                world_model_store=world_store,
                policy_store=policy_store,
                state_dir=state_dir,
            )
        )
    if args.executor:
        facets.append(
            ExecutorFacet(
                goal_store=goal_store,
                world_model_store=world_store,
                memory_store=memory_store,
                state_dir=state_dir,
                sandbox_dir=Path(args.executor_sandbox_dir) if args.executor_sandbox_dir else None,
            )
        )
    if args.learning:
        facets.append(
            LearningFacet(
                memory_store=memory_store,
                goal_store=goal_store,
                world_model_store=world_store,
            )
        )
    if args.attention:
        facets.append(
            AttentionFacet(
                memory_store=memory_store,
                top_n=args.attention_top_n,
                history_size=args.attention_history_size,
            )
        )
    if args.affect:
        facets.append(AffectFacet(history_size=args.affect_history_size))
    facets.append(EchoFacet())
    if args.verify:
        facets.append(VerifierFacet(state_dir=state_dir))

    return NexusRuntime(facets=facets, store=store)


def _format_presentation_compact(vector: dict[str, Any]) -> str:
    return (
        f"mode={vector.get('mode')} "
        f"intensity={float(vector.get('intensity') or 0):.2f} "
        f"motion={vector.get('motion')} "
        f"channel={vector.get('channel')}"
    )


def _format_tick_summary_line(summary: dict[str, Any], *, presentation: bool = False) -> str:
    idx = summary.get("tick_index")
    base = (
        f"tick {idx}: decision={summary.get('decision')} "
        f"system_pressure={float(summary.get('system_pressure') or 0):.3f} "
        f"latent_pressure={float(summary.get('latent_pressure') or 0):.3f} "
        f"expression_mode={summary.get('expression_mode')} "
        f"expression_suppressed={summary.get('expression_suppressed')} "
        f"interrupt_candidates={summary.get('interrupt_candidates_count')} "
        f"suppressed_interrupts={summary.get('suppressed_interrupt_count')} "
        f"internal_processed={summary.get('internal_event_processed')}"
    )
    if presentation:
        pv = summary.get("presentation_vector")
        if isinstance(pv, dict):
            base = f"tick {idx}: {_format_presentation_compact(pv)}"
    return base


def _print_tick_run_cli(args: argparse.Namespace, result: TickRunResult) -> None:
    if args.json or args.debug:
        blob = result.to_dict()
        if args.json and not args.debug:
            blob = dict(blob)
            blob.pop("records", None)
        print(json.dumps({"tick_run": blob}, indent=2))
        return
    if args.tick_summary or args.presentation:
        for row in result.summaries:
            if "error" in row:
                print(f"tick {row.get('tick_index')}: error={row.get('error')}")
            else:
                print(_format_tick_summary_line(row, presentation=bool(args.presentation)))
    parts = [
        f"tick_run: completed={result.tick_count}",
        f"stopped_early={str(result.stopped_early).lower()}",
    ]
    if result.stop_reason:
        parts.append(f"stop_reason={result.stop_reason}")
    if result.stop_tick_index is not None:
        parts.append(f"stop_tick_index={result.stop_tick_index}")
    fs = result.final_state_summary
    parts.append(f"event_count={fs.get('event_count')}")
    parts.append(f"system_pressure={float(fs.get('system_pressure') or 0):.3f}")
    print(" ".join(parts))


def _interactive_user_text_output(
    record,
    *,
    model_adapter: ModelAdapter | None = None,
    debug: bool = False,
) -> str:
    output = _derive_response_output(
        record,
        model_adapter=model_adapter,
        debug=debug,
    )
    response = output.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    if response is not None:
        return str(response)
    if output.get("recorded"):
        return "(recorded)"
    return "(silent)"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.full:
        _apply_full_preset(args)
    content = args.content if args.content is not None else args.prompt or ""
    metadata = _parse_metadata(parser, args.metadata)
    if args.feedback is not None:
        metadata["feedback"] = args.feedback
    if args.target_memory_id is not None:
        metadata["target_memory_id"] = args.target_memory_id
    if args.target_goal_id is not None:
        metadata["target_goal_id"] = args.target_goal_id
    if args.pressure is not None:
        metadata["pressure"] = _clamp_unit(args.pressure)
    if args.novelty is not None:
        metadata["novelty"] = _clamp_unit(args.novelty)
    if args.execute_plan:
        metadata["execute_plan"] = True
    if args.executor_live:
        args.live = True
    if args.live and args.execute_plan:
        metadata["dry_run"] = False
    if args.executor_sandbox_dir:
        metadata["executor_sandbox_dir"] = str(Path(args.executor_sandbox_dir))
    if args.ticks < 1:
        parser.error("--ticks must be at least 1.")
    if args.ticks > TICK_HARD_CAP:
        parser.error(f"--ticks must be at most {TICK_HARD_CAP}.")

    watch_mode_run = bool(getattr(args, "watch", False))
    loop_mode_run = bool(getattr(args, "loop", False))
    interactive_mode_run = bool(getattr(args, "interactive", False))
    manual_tick_run = bool(args.tick) or args.ticks > 1
    if interactive_mode_run and loop_mode_run:
        parser.error("--interactive is not compatible with --loop.")
    if interactive_mode_run and watch_mode_run:
        parser.error("--interactive is not compatible with --watch.")
    if interactive_mode_run and manual_tick_run:
        parser.error("--interactive is not compatible with --tick / --ticks.")
    if loop_mode_run and watch_mode_run:
        parser.error("--loop is not compatible with --watch.")
    if loop_mode_run and manual_tick_run:
        parser.error("--loop is not compatible with --tick / --ticks.")
    if watch_mode_run and (args.tick or args.ticks > 1):
        parser.error("--watch is not compatible with --tick / --ticks; use --watch-ticks instead.")
    if watch_mode_run and args.event_type != EventType.USER_MESSAGE.value:
        parser.error("--watch requires user_message event-type; omit --event-type for watch runs.")
    if watch_mode_run and args.model:
        parser.error("--model is not used with watch mode; remove --model for watch runs.")
    if loop_mode_run and args.model:
        parser.error("Continuous loop does not support --model in v0.")
    if interactive_mode_run and args.json:
        parser.error("--json is not supported with --interactive in v0.")
    if interactive_mode_run and args.model and not args.interactive_allow_model:
        parser.error(
            "Interactive loop does not support --model in v0 unless --interactive-allow-model is set."
        )

    if manual_tick_run and args.event_type != EventType.USER_MESSAGE.value:
        parser.error("--tick / --ticks runs SYSTEM_TICK; omit --event-type or keep user_message.")
    if manual_tick_run and args.model:
        parser.error("--model is not used with manual ticks; remove --model for tick runs.")

    if args.attention_top_n < 1:
        parser.error("--attention-top-n must be at least 1.")
    if args.attention_history_size < 1:
        parser.error("--attention-history-size must be at least 1.")
    if args.affect_history_size < 1:
        parser.error("--affect-history-size must be at least 1.")
    if args.context_max_goals < 0:
        parser.error("--context-max-goals must be at least 0.")
    if args.context_max_beliefs < 0:
        parser.error("--context-max-beliefs must be at least 0.")
    if args.context_max_memories is not None and args.context_max_memories < 0:
        parser.error("--context-max-memories must be at least 0.")
    if args.context_max_items < 1:
        parser.error("--context-max-items must be at least 1.")
    if args.context_window_size < 1:
        parser.error("--context-window-size must be at least 1.")
    if args.interactive_status_every < 0:
        parser.error("--interactive-status-every must be at least 0.")
    if args.working_memory_context_turns < 0:
        parser.error("--working-memory-context-turns must be at least 0.")
    if args.working_memory_turns < 1:
        parser.error("--working-memory-turns must be at least 1.")

    model_adapter = _build_model_adapter(parser, args.model)

    if args.executor_list_skills:
        facet = ExecutorFacet(
            state_dir=Path(args.state_dir),
            sandbox_dir=Path(args.executor_sandbox_dir) if args.executor_sandbox_dir else None,
        )
        payload = [entry.to_dict() for entry in facet.executor.registry.list_skills()]
        print(json.dumps({"executor_skills": payload}, indent=2))
        return 0

    if interactive_mode_run:
        if args.event_type != EventType.USER_MESSAGE.value:
            parser.error("--interactive manages SYSTEM_TICK and USER_MESSAGE events; omit --event-type.")
        cfg = InteractiveLoopConfig(
            interval_seconds=args.interactive_interval,
            max_ticks=args.interactive_max_ticks,
            clear_screen=bool(args.interactive_clear and not args.interactive_no_clear),
            allow_expression=not bool(args.interactive_no_expression),
            allow_model=bool(args.interactive_allow_model),
            stop_on_ask_user=bool(args.interactive_stop_on_ask_user),
            show_ticks=bool(args.interactive_show_ticks),
            status_every=max(0, int(args.interactive_status_every or 0)),
            session_id=args.session_id or "",
            working_memory_context_turns=max(1, int(args.working_memory_context_turns or 8)),
            working_memory_retain_turns=max(1, int(args.working_memory_turns or 20)),
        ).clamped()
        if not cfg.allow_model:
            model_adapter = None
        hook_meta = build_tick_event_metadata(
            tick_index=1,
            tick_count=cfg.max_ticks,
            tick_reason="interactive_loop_v0",
            suppress_expression=not cfg.allow_expression,
            extra={**metadata, "interactive_loop": True},
        )
        hook_event = Event(
            event_type=EventType.SYSTEM_TICK,
            content="",
            metadata=hook_meta,
        )
        runtime = _cli_build_nexus_runtime(parser, args, event=hook_event)
        run_interactive_loop(
            runtime,
            cfg,
            output_writer=sys.stdout,
            extra_metadata=metadata,
            user_output_builder=lambda record: _interactive_user_text_output(
                record,
                model_adapter=model_adapter if cfg.allow_model else None,
                debug=args.debug,
            ),
        )
        return 0

    if loop_mode_run:
        if args.event_type != EventType.USER_MESSAGE.value:
            parser.error("--loop runs SYSTEM_TICK; omit --event-type or keep user_message.")
        cfg = ContinuousLoopConfig(
            interval_seconds=args.loop_interval,
            max_ticks=args.loop_max_ticks,
            clear_screen=not bool(args.loop_no_clear),
            allow_tick_expression=bool(args.loop_allow_expression),
            stop_on_ask_user=bool(args.loop_stop_on_ask_user),
        ).clamped()
        hook_meta = build_tick_event_metadata(
            tick_index=1,
            tick_count=cfg.max_ticks,
            tick_reason="continuous_loop_v0",
            suppress_expression=not cfg.allow_tick_expression,
            extra=metadata,
        )
        hook_event = Event(
            event_type=EventType.SYSTEM_TICK,
            content="",
            metadata=hook_meta,
        )
        runtime = _cli_build_nexus_runtime(parser, args, event=hook_event)
        if args.loop_json:
            result = run_continuous_loop(
                runtime,
                cfg,
                output_writer=None,
                extra_metadata=metadata,
            )
            print(json.dumps({"continuous_loop": result.to_dict()}, indent=2))
            return 0
        result = run_continuous_loop(
            runtime,
            cfg,
            output_writer=sys.stdout,
            extra_metadata=metadata,
        )
        return 0

    if watch_mode_run:
        cfg = WatchConfig(
            ticks=args.watch_ticks,
            interval_seconds=args.watch_interval,
            clear_screen=args.watch_clear,
            show_trace=args.watch_trace,
            show_json=args.watch_json,
        ).clamped()
        hook_meta = build_tick_event_metadata(
            tick_index=1,
            tick_count=cfg.ticks,
            tick_reason=None,
            suppress_expression=True,
            extra=metadata,
        )
        hook_event = Event(
            event_type=EventType.SYSTEM_TICK,
            content="",
            metadata=hook_meta,
        )
        runtime = _cli_build_nexus_runtime(parser, args, event=hook_event)
        run_watch_mode(
            runtime,
            cfg,
            tick_run_extra_metadata=metadata,
        )
        return 0

    if manual_tick_run:
        hook_meta = build_tick_event_metadata(
            tick_index=1,
            tick_count=args.ticks,
            tick_reason=args.tick_reason,
            suppress_expression=not args.allow_tick_expression,
            extra=metadata,
        )
        hook_event = Event(
            event_type=EventType.SYSTEM_TICK,
            content="",
            metadata=hook_meta,
        )
        runtime = _cli_build_nexus_runtime(parser, args, event=hook_event)
        tick_result = run_manual_ticks(
            runtime,
            total_ticks=args.ticks,
            tick_reason=args.tick_reason,
            suppress_expression=not args.allow_tick_expression,
            extra_metadata=metadata,
            include_full_records=bool(args.json or args.debug),
            include_presentation=bool(args.presentation or args.tick_summary),
        )
        _print_tick_run_cli(args, tick_result)
        return 0

    state_dir = Path(args.state_dir)
    event = Event(
        event_type=EventType(args.event_type),
        content=content,
        metadata=metadata,
    )
    runtime = _cli_build_nexus_runtime(parser, args, event=event)
    record = runtime.process_event(event)

    if args.json or args.debug:
        out_rec = record.to_dict()
        if args.presentation:
            out_rec = dict(out_rec)
            out_rec["presentation_vector"] = derive_presentation_vector(
                record,
                runtime.state,
            ).to_dict()
        print(json.dumps(out_rec, indent=2))
    else:
        text = format_record_output(
            record,
            model_adapter=model_adapter,
            debug=args.debug,
            expression_gate=args.expression_gate,
        )
        lines = [text]
        if args.presentation:
            lines.append(
                "presentation: "
                + _format_presentation_compact(
                    derive_presentation_vector(record, runtime.state).to_dict(),
                )
            )
        print("\n".join(lines))
    return 0


def _apply_full_preset(args: argparse.Namespace) -> None:
    for flag_name in FULL_PRESET_FLAGS:
        setattr(args, flag_name, True)


def format_record_output(
    record,
    *,
    model_adapter: ModelAdapter | None = None,
    debug: bool = False,
    expression_gate: bool = False,
) -> str:
    """Return deterministic, concise CLI output for a processed record."""
    decision = record.decision
    lines = [f"decision: {decision.action.value.upper()}"]
    output = _derive_response_output(
        record,
        model_adapter=model_adapter,
        debug=debug,
    )

    if output.get("tool") is not None:
        lines.append(f"tool: {output['tool']}")
    response = output.get("response")
    lines.append(f"response: {json.dumps(response)}")
    if output.get("recorded") is not None:
        lines.append(f"recorded: {str(output['recorded']).lower()}")
    lines.append(f"reason: {decision.reason}")
    if expression_gate:
        er = (
            record.metadata.get("expression_recommendation")
            if isinstance(record.metadata, dict)
            else None
        )
        if isinstance(er, dict):
            lines.append(
                "expression: "
                f"mode={er.get('mode')} "
                f"score={round(float(er.get('expression_score') or 0), 4)} "
                f"suppressed={str(bool(er.get('suppressed'))).lower()} "
                f"intent={er.get('suggested_intent')}"
            )
    return "\n".join(lines)


def _derive_response_output(
    record,
    *,
    model_adapter: ModelAdapter | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    action = record.decision.action
    if action.value == "wait":
        return {"response": None}
    if action.value == "record":
        return {"response": None, "recorded": True}
    if action.value == "ask":
        output_metadata = _text_output_metadata(record)
        response = _generate_or_render_text_response(
            record,
            output_metadata,
            "clarification_needed",
            model_adapter=model_adapter,
            debug=debug,
        )
        return {
            "tool": output_metadata.get("tool"),
            "response": response,
        }
    if action.value == "act":
        output_metadata = _text_output_metadata(record)
        if not _is_text_output(output_metadata):
            return {"response": None}
        return {
            "tool": output_metadata.get("tool"),
            "response": _generate_or_render_text_response(
                record,
                output_metadata,
                None,
                model_adapter=model_adapter,
                debug=debug,
            ),
        }
    return {"response": None}


def _text_output_metadata(record) -> dict[str, Any]:
    source_facets = list(getattr(record.decision, "source_facets", []) or [])
    for facet_name in source_facets:
        metadata = _facet_metadata(record, facet_name=facet_name)
        if _is_text_output(metadata):
            return metadata

    for result in record.facet_results:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if result.proposed_decision == record.decision.action and _is_text_output(metadata):
            return metadata
    return {}


def _facet_metadata(record, *, facet_name: str) -> dict[str, Any]:
    for result in record.facet_results:
        if result.facet_name != facet_name:
            continue
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        return metadata
    return {}


def _is_text_output(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("output_type") == "text"
        or metadata.get("tool") == "text"
        or metadata.get("response_template") in TEXT_RESPONSE_TEMPLATES
    )


def _render_text_response(
    metadata: dict[str, Any],
    default_template: str | None,
    *,
    record=None,
) -> str | None:
    template_name = metadata.get("response_template") or default_template
    if not isinstance(template_name, str):
        return None
    if template_name == "next_steps_available" and record is not None:
        next_steps_response = _render_next_steps_response(record)
        if next_steps_response is not None:
            return next_steps_response
    return TEXT_RESPONSE_TEMPLATES.get(template_name)


def _generate_or_render_text_response(
    record,
    metadata: dict[str, Any],
    default_template: str | None,
    *,
    model_adapter: ModelAdapter | None,
    debug: bool,
) -> str | None:
    if not _metadata_flag(metadata, "response_needed"):
        return None

    if model_adapter is not None:
        prompt = _build_model_prompt(record, metadata)
        try:
            response = model_adapter.generate(prompt).strip()
            if response:
                return response
        except ModelAdapterError as exc:
            if debug:
                print(f"warning: model generation failed: {exc}", file=sys.stderr)

    return _render_text_response(metadata, default_template, record=record)


def _build_model_prompt(record, metadata: dict[str, Any]) -> str:
    decision = record.decision.action.value.upper()
    lines = [
        "You are a local AI system.",
        "Only generate text; do not decide actions, modify state, or call tools.",
        f"User said: {_prompt_line(record.event.content)}",
        f"System decision: {decision}",
        "Current working context:",
        *_working_context_prompt_lines(record),
        "Recent conversation:",
        *_recent_conversation_prompt_lines(record),
        "Response grounding:",
        f"- query intent: {metadata.get('query_intent') or 'none'}",
        f"- planner summary: {_recent_planner_summary(record)}",
        f"- missing context: {_missing_context_summary(metadata)}",
        f"- response template: {metadata.get('response_template') or 'none'}",
        "Respond concisely.",
    ]
    return "\n".join(lines)


def _working_context_prompt_lines(record) -> list[str]:
    context_window = _context_window_payload(record)
    if not isinstance(context_window, dict):
        return _fallback_working_context_lines(record)

    raw_items = context_window.get("items", [])
    if not isinstance(raw_items, list):
        return _fallback_working_context_lines(record)
    items = [item for item in raw_items if isinstance(item, dict)]
    if not items:
        return _fallback_working_context_lines(record)

    lines: list[str] = []
    current_event = next(
        (
            _prompt_line(str(item.get("content", "")))
            for item in items
            if item.get("item_type") == "event"
            and str(item.get("content", "")).strip()
        ),
        _prompt_line(record.event.content) if record.event.content.strip() else "none",
    )
    lines.append(f"- current event: {current_event}")
    lines.append(f"- active unresolved signals: {_context_signal_summary(items) or 'none'}")
    lines.append(f"- attention broadcast: {_context_attention_summary(items) or 'none'}")
    lines.append(f"- active goals: {_context_goal_summary(items) or 'none'}")
    lines.append(f"- active beliefs: {_context_belief_summary(items) or 'none'}")
    lines.append(
        f"- relevant memories: {_context_memory_summary(items, context_source='relevant') or 'none'}"
    )
    lines.append(
        f"- recent memories: {_context_memory_summary(items, context_source='recent') or 'none'}"
    )
    lines.append(f"- policy: {_context_policy_summary(items) or 'none'}")
    lines.append(f"- system status signals: {_context_signal_summary(items) or 'none'}")
    return lines


def _recent_conversation_prompt_lines(record) -> list[str]:
    context_window = _context_window_payload(record)
    if not isinstance(context_window, dict):
        return ["- none"]
    raw_items = context_window.get("items", [])
    if not isinstance(raw_items, list):
        return ["- none"]
    recent_items = [
        item
        for item in raw_items
        if isinstance(item, dict) and item.get("item_type") == "working_memory"
    ]
    if not recent_items:
        return ["- none"]
    lines: list[str] = []
    for item in recent_items:
        metadata = item.get("metadata")
        role = "turn"
        if isinstance(metadata, dict):
            role = str(metadata.get("dialogue_role") or "turn")
        content = _coerce_prompt_string(item.get("content")) or ""
        if not content:
            continue
        lines.append(f"- {role.title()}: {content}")
    return lines or ["- none"]


def _fallback_working_context_lines(record) -> list[str]:
    return [
        f"- current event: {_prompt_line(record.event.content) or 'none'}",
        f"- active goals: {_active_goals_summary(record)}",
        f"- relevant memories: {_relevant_memories_summary(record)}",
        f"- recent memories: {_relevant_memories_summary(record)}",
        f"- active beliefs: {_relevant_beliefs_summary(record)}",
        "- policy: none",
        f"- signals: {_recent_planner_summary(record)}",
    ]


def _context_window_payload(record) -> dict[str, Any] | None:
    context_metadata = _facet_metadata(record, facet_name="context")
    payload = context_metadata.get("context_window")
    if isinstance(payload, dict):
        return payload
    return None


def _context_goal_summary(items: list[dict[str, Any]]) -> str | None:
    summaries: list[str] = []
    seen_goal_keys: set[str] = set()
    for item in items:
        if item.get("item_type") != "goal":
            continue
        content = _coerce_prompt_string(item.get("content"))
        if content is None:
            continue
        goal_key = normalize_goal_description(content)
        if goal_key in seen_goal_keys:
            continue
        seen_goal_keys.add(goal_key)
        metadata = item.get("metadata", {})
        priority = metadata.get("priority") if isinstance(metadata, dict) else None
        if isinstance(priority, (int, float)):
            summaries.append(f"{content} (priority {float(priority):.1f})")
        else:
            summaries.append(content)
        if len(summaries) >= 3:
            break
    if summaries:
        return "; ".join(summaries)
    return None


def _context_attention_summary(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        if item.get("item_type") != "attention":
            continue
        content = _coerce_prompt_string(item.get("content"))
        if content is None:
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            return content
        source = _coerce_prompt_string(metadata.get("attention_source"))
        mode = _coerce_prompt_string(metadata.get("attention_mode"))
        score = metadata.get("attention_score")
        if isinstance(score, (int, float)):
            return (
                f"{content} (source {source or 'unknown'}, "
                f"mode {mode or 'unknown'}, score {float(score):.2f})"
            )
        return content
    return None


def _context_memory_summary(
    items: list[dict[str, Any]],
    *,
    context_source: str,
) -> str | None:
    summaries: list[str] = []
    for item in items:
        if item.get("item_type") != "memory":
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if metadata.get("context_source") != context_source:
            continue
        content = _coerce_prompt_string(item.get("content"))
        if content is None:
            continue
        annotations: list[str] = []
        role = metadata.get("role")
        if isinstance(role, str) and role and role != "unknown":
            annotations.append(f"role={role}")
        domain = metadata.get("domain")
        if isinstance(domain, str) and domain:
            annotations.append(f"domain={domain}")
        if annotations:
            summaries.append(f"{content} ({', '.join(annotations)})")
        else:
            summaries.append(content)
        if len(summaries) >= 3:
            break
    if summaries:
        return "; ".join(summaries)
    return None


def _context_belief_summary(items: list[dict[str, Any]]) -> str | None:
    summaries: list[str] = []
    for item in items:
        if item.get("item_type") != "belief":
            continue
        content = _coerce_prompt_string(item.get("content"))
        if content is None:
            continue
        metadata = item.get("metadata", {})
        confidence = metadata.get("confidence") if isinstance(metadata, dict) else None
        if isinstance(confidence, (int, float)):
            summaries.append(f"{content} (confidence {float(confidence):.1f})")
        else:
            summaries.append(content)
        if len(summaries) >= 3:
            break
    if summaries:
        return "; ".join(summaries)
    return None


def _context_policy_summary(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        if item.get("item_type") != "policy":
            continue
        return _coerce_prompt_string(item.get("content"))
    return None


def _context_signal_summary(items: list[dict[str, Any]]) -> str | None:
    summaries: list[str] = []
    for item in items:
        if item.get("item_type") != "signal":
            continue
        content = _coerce_prompt_string(item.get("content"))
        if content is None:
            continue
        summaries.append(content)
    if summaries:
        return "; ".join(summaries[:5])
    return None


def _active_goals_summary(record) -> str:
    goals_metadata = _facet_metadata(record, facet_name="goals")
    return _goals_summary(goals_metadata.get("active_goals"))


def _relevant_goals_summary(record) -> str:
    goals_metadata = _facet_metadata(record, facet_name="goals")
    return _goals_summary(goals_metadata.get("relevant_goals"))


def _relevant_memories_summary(record) -> str:
    memory_metadata = _facet_metadata(record, facet_name="memory")
    raw_memories = memory_metadata.get("relevant_memories")
    if isinstance(raw_memories, list) and raw_memories:
        summaries: list[str] = []
        for memory in raw_memories[:3]:
            if not isinstance(memory, dict):
                continue
            preview = _coerce_prompt_string(memory.get("content_preview"))
            if preview is not None:
                summaries.append(preview)
        if summaries:
            return "; ".join(summaries)
    return "none"


def _relevant_beliefs_summary(record) -> str:
    world_metadata = _facet_metadata(record, facet_name="world_model")
    raw_beliefs = world_metadata.get("relevant_beliefs")
    if isinstance(raw_beliefs, list) and raw_beliefs:
        summaries: list[str] = []
        for belief in raw_beliefs[:3]:
            if not isinstance(belief, dict):
                continue
            claim = _coerce_prompt_string(belief.get("claim"))
            if claim is None:
                continue
            confidence = belief.get("confidence")
            if isinstance(confidence, (int, float)):
                summaries.append(f"{claim} (confidence {float(confidence):.1f})")
            else:
                summaries.append(claim)
        if summaries:
            return "; ".join(summaries)
    return "none"


def _missing_context_summary(metadata: dict[str, Any]) -> str:
    raw_missing = metadata.get("missing_context")
    if isinstance(raw_missing, list) and raw_missing:
        return ", ".join(str(item) for item in raw_missing[:5])
    return "none"


def _goals_summary(raw_goals: Any) -> str:
    if isinstance(raw_goals, list) and raw_goals:
        summaries: list[str] = []
        seen_goal_keys: set[str] = set()
        for goal in raw_goals:
            if not isinstance(goal, dict):
                continue
            description = _coerce_prompt_string(goal.get("description"))
            if description is None:
                continue
            goal_key = normalize_goal_description(description)
            if goal_key in seen_goal_keys:
                continue
            seen_goal_keys.add(goal_key)
            priority = goal.get("priority")
            if isinstance(priority, (int, float)):
                summaries.append(f"{description} (priority {float(priority):.1f})")
            else:
                summaries.append(description)
            if len(summaries) >= 3:
                break
        if summaries:
            return "; ".join(summaries)
    return "none"


def _top_goal_description(record) -> str | None:
    goals_metadata = _facet_metadata(record, facet_name="goals")
    for key in ("relevant_goals", "active_goals"):
        goals = goals_metadata.get(key)
        if not isinstance(goals, list):
            continue
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            description = _coerce_prompt_string(goal.get("description"))
            if description is not None:
                return description
    return None


def _recent_planner_summary(record) -> str:
    planner_metadata = _facet_metadata(record, facet_name="planner")
    plan = planner_metadata.get("plan")
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            step = steps[-1]
            if isinstance(step, dict):
                description = _coerce_prompt_string(step.get("description"))
                if description is not None:
                    return description
        title = _coerce_prompt_string(plan.get("title"))
        if title is not None:
            return title
    return _recent_action_summary(record)


def _render_next_steps_response(record) -> str | None:
    goal_description = _top_goal_description(record)
    planner_summary = _recent_planner_summary(record)
    if goal_description is None:
        return None
    if planner_summary != "none":
        return f"Focus next on {goal_description}. Suggested next step: {planner_summary}"
    return f"Focus next on {goal_description}."


def _recent_action_summary(record) -> str:
    for facet_name in ("executor", "planner"):
        metadata = _facet_metadata(record, facet_name=facet_name)
        summary = _coerce_prompt_string(metadata.get("reasons"))
        if summary is not None:
            return f"{facet_name}: {summary}"

    for result in reversed(record.facet_results):
        if result.facet_name in {"executor", "planner"}:
            return f"{result.facet_name}: {_prompt_line(result.summary)}"
    return "none"


def _coerce_prompt_string(value: Any) -> str | None:
    if isinstance(value, str):
        return _prompt_line(value)
    if isinstance(value, list) and value:
        cleaned_values = [
            _prompt_line(str(item))
            for item in value[:3]
            if isinstance(item, (str, int, float, bool))
        ]
        if cleaned_values:
            return ", ".join(cleaned_values)
    return None


def _prompt_line(value: str, *, limit: int = 240) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


def _build_model_adapter(
    parser: argparse.ArgumentParser,
    raw_model: str | None,
) -> ModelAdapter | None:
    if raw_model is None:
        return None
    if not raw_model.startswith("ollama:"):
        parser.error("--model currently supports only ollama:<model-name>.")

    model_name = raw_model.removeprefix("ollama:").strip()
    if not model_name:
        parser.error("--model ollama:<model-name> requires a model name.")
    return OllamaAdapter(model_name)


def _parse_metadata(
    parser: argparse.ArgumentParser,
    raw_metadata: str | None,
) -> dict[str, Any]:
    if raw_metadata is None:
        return {}

    try:
        payload = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        parser.error(f"--metadata must be valid JSON: {exc}")

    if not isinstance(payload, dict):
        parser.error("--metadata must decode to a JSON object.")
    return payload


def _create_goal_from_metadata(
    store: SQLiteGoalStore,
    *,
    content: str,
    metadata: dict[str, Any],
) -> Goal | None:
    if not _metadata_flag(metadata, "create_goal"):
        return None
    if not content.strip():
        return None

    explicit_tags: list[str] = []
    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        explicit_tags = normalize_tags(raw_tags)

    goal = Goal(
        description=content,
        priority=0.5,
        tags=merge_tags(explicit_tags, infer_tags(content)),
        source=GoalSource.USER,
        metadata={
            key: value
            for key, value in metadata.items()
            if key != "create_goal"
        },
    )
    store.add_goal(goal)
    return goal


def _create_goal_from_intent(
    store: SQLiteGoalStore,
    *,
    content: str,
    metadata: dict[str, Any],
) -> Goal | None:
    if _metadata_flag(metadata, "create_goal"):
        return None
    detected_goal = _detect_goal_intent(content)
    if detected_goal is None:
        return None
    description, priority = detected_goal

    explicit_tags: list[str] = []
    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        explicit_tags = normalize_tags(raw_tags)
    tags = merge_tags(explicit_tags, infer_tags(description), infer_tags(content))

    matching_goal = find_matching_active_goal(
        store.list_active_goals(limit=50),
        description,
    )
    if matching_goal is not None:
        matching_goal.priority = max(matching_goal.priority, priority)
        matching_goal.tags = merge_tags(matching_goal.tags, tags)
        matching_goal.status = GoalStatus.ACTIVE
        matching_goal.source = GoalSource.USER
        matching_goal.metadata = {
            **matching_goal.metadata,
            "merged_from_duplicate_intent": True,
            "merged_goal_normalized_key": normalize_goal_description(description),
            "last_intent_phrase": content.strip(),
        }
        store.update_goal(matching_goal)
        return matching_goal

    goal = Goal(
        description=description,
        priority=priority,
        tags=tags,
        source=GoalSource.USER,
        status=GoalStatus.ACTIVE,
        metadata={"intent_phrase": content.strip()},
    )
    store.add_goal(goal)
    return goal


def _detect_goal_intent(content: str) -> tuple[str, float] | None:
    for pattern, priority in GOAL_INTENT_PATTERNS:
        match = pattern.match(content)
        if match is None:
            continue
        description = _clean_goal_description(match.group("description"))
        if description:
            return description, priority
    return None


def _clean_goal_description(description: str) -> str:
    cleaned = " ".join(description.strip().strip("\"'`").split())
    cleaned = re.sub(r"^(?:to|that)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .!?")


def _metadata_flag(metadata: dict[str, Any], key: str) -> bool:
    raw_value = metadata.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _create_belief_from_metadata(
    store: SQLiteWorldModelStore,
    *,
    event: Event,
) -> Belief | None:
    if not _metadata_flag(event.metadata, "create_belief"):
        return None
    if not event.content.strip():
        return None

    explicit_tags: list[str] = []
    raw_tags = event.metadata.get("tags", [])
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        explicit_tags = normalize_tags(raw_tags)

    belief = Belief(
        claim=event.content,
        confidence=0.7,
        tags=merge_tags(explicit_tags, infer_tags(event.content)),
        source=_belief_source_for_event(event.event_type),
        source_event_id=event.event_id,
        metadata={
            key: value
            for key, value in event.metadata.items()
            if key != "create_belief"
        },
    )
    store.add_belief(belief)
    return belief


def _belief_source_for_event(event_type: EventType) -> BeliefSource:
    if event_type == EventType.USER_MESSAGE:
        return BeliefSource.USER
    return BeliefSource.SYSTEM


def _create_policy_from_metadata(
    store: SQLitePolicyStore,
    *,
    content: str,
    metadata: dict[str, Any],
) -> PolicyRule | None:
    if not _metadata_flag(metadata, "create_policy"):
        return None

    rule_type = coerce_policy_rule_type(metadata.get("rule_type"))
    if rule_type is None:
        raise ValueError(
            "create_policy metadata requires a valid rule_type "
            "(allow, deny, require_approval, prefer)."
        )

    target_type = coerce_policy_target_type(metadata.get("target_type"))
    if target_type is None:
        raise ValueError(
            "create_policy metadata requires a valid target_type "
            "(internal_state, file_write, file_delete, shell, network, "
            "message, git, tool, decision, tag, general)."
        )

    conditions = metadata.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError("create_policy metadata field 'conditions' must be a JSON object.")

    priority = metadata.get("priority", 0.0)
    if not isinstance(priority, (int, float)):
        raise ValueError("create_policy metadata field 'priority' must be numeric.")

    rule = PolicyRule(
        name=_policy_name_from_metadata(content=content, metadata=metadata),
        description=_coerce_metadata_string(metadata, "description") or content.strip(),
        rule_type=rule_type,
        target_type=target_type,
        target=_coerce_metadata_string(metadata, "target") or "*",
        conditions=conditions,
        priority=float(priority),
        enabled=_metadata_enabled(metadata),
        source=coerce_policy_source(metadata.get("source")) or PolicySource.USER,
        metadata=_policy_metadata_payload(metadata),
    )
    store.add_policy(rule)
    return rule


def _policy_name_from_metadata(
    *,
    content: str,
    metadata: dict[str, Any],
) -> str:
    for key in ("policy_name", "name"):
        value = _coerce_metadata_string(metadata, key)
        if value:
            return value
    if content.strip():
        return content.strip()
    return "policy-rule"


def _coerce_metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    raw_value = metadata.get(key)
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip()
    return cleaned or None


def _metadata_enabled(metadata: dict[str, Any]) -> bool:
    if "enabled" not in metadata:
        return True
    return _metadata_flag(metadata, "enabled")


def _policy_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw_policy_metadata = metadata.get("policy_metadata")
    if isinstance(raw_policy_metadata, dict):
        payload.update(raw_policy_metadata)

    control_keys = {
        "create_policy",
        "rule_type",
        "target_type",
        "target",
        "priority",
        "enabled",
        "source",
        "conditions",
        "policy_name",
        "name",
        "description",
        "policy_metadata",
    }
    for key, value in metadata.items():
        if key in control_keys:
            continue
        payload[key] = value
    return payload


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
