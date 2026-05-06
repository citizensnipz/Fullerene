from __future__ import annotations

import time
from typing import Any, Callable, IO

from fullerene.interactive.input import InputProvider, ThreadedStdinInputProvider
from fullerene.interactive.models import InteractiveLoopConfig, InteractiveLoopResult
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.tick import TickStopTracker, build_tick_event_metadata, summarize_tick_record


def _is_verifier_critical(record_dict: dict[str, Any]) -> bool:
    for fr in record_dict.get("facet_results", []):
        if not isinstance(fr, dict) or fr.get("facet_name") != "verifier":
            continue
        md = fr.get("metadata")
        if not isinstance(md, dict):
            continue
        for key in ("artifact_checks", "schema_checks"):
            rows = md.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("status") or "").lower() != "failed":
                    continue
                if str(row.get("severity") or "").lower() in {"critical", "error"}:
                    return True
    return False


def run_interactive_loop(
    runtime: NexusRuntime,
    config: InteractiveLoopConfig,
    *,
    output_writer: IO[str] | None = None,
    input_provider: InputProvider | None = None,
    extra_metadata: dict[str, Any] | None = None,
    user_output_builder: Callable[[Any], str] | None = None,
) -> InteractiveLoopResult:
    cfg = config.clamped()
    writer = output_writer
    provider = input_provider or ThreadedStdinInputProvider()
    created_provider = input_provider is None
    stop_tracker = TickStopTracker()
    tick_index = 0
    input_index = 0
    stop_reason: str | None = None
    stopped_early = False
    final_mode = "unknown"
    final_pressure = 0.0
    final_latent_pressure = 0.0
    final_expression_mode = "silent"
    final_decision = "wait"
    last_status_tick = 0
    final_text = "(silent)"
    last_tick_at = 0.0
    next_tick_at = time.monotonic()
    working_turn_index = 0
    working_turns_stored = 0
    memory_store = _resolve_memory_store(runtime)
    session_id = str(cfg.session_id or "interactive")

    exit_commands = {cmd.strip().casefold() for cmd in cfg.exit_commands}
    if writer is not None and cfg.show_help_on_start:
        writer.write(
            "Interactive loop started. Enter text and press Enter. "
            f"Exit with: {', '.join(cfg.exit_commands)}\n"
        )
        writer.write("[status] mode=unknown pressure=0.00\n\n")
        writer.write(cfg.input_prompt)
        writer.flush()

    try:
        while tick_index < cfg.max_ticks:
            now = time.monotonic()
            if now >= next_tick_at:
                tick_index += 1
                tick_meta = build_tick_event_metadata(
                    tick_index=tick_index,
                    tick_count=cfg.max_ticks,
                    tick_reason="interactive_loop_v0",
                    suppress_expression=not cfg.allow_expression,
                    extra={
                        **(extra_metadata or {}),
                        "interactive_loop": True,
                        "session_id": session_id,
                    },
                )
                tick_event = Event(
                    event_type=EventType.SYSTEM_TICK,
                    content="",
                    metadata=tick_meta,
                )
                try:
                    tick_record = runtime.process_event(tick_event)
                except KeyboardInterrupt:
                    stopped_early = True
                    stop_reason = "keyboard_interrupt"
                    break
                except Exception as exc:
                    stopped_early = True
                    stop_reason = f"runtime_exception:{exc.__class__.__name__}"
                    break

                tick_summary = summarize_tick_record(
                    tick_record,
                    state=runtime.state,
                    include_presentation=True,
                )
                tick_record_dict = tick_record.to_dict()
                pv = tick_summary.get("presentation_vector") or {}
                final_mode = str(pv.get("mode") or "unknown")
                final_pressure = float(tick_summary.get("system_pressure") or 0.0)
                final_latent_pressure = float(tick_summary.get("latent_pressure") or 0.0)
                final_expression_mode = str(tick_summary.get("expression_mode") or "silent")
                final_decision = str(tick_summary.get("decision") or "wait")
                final_text = "(silent)"
                should_print_tick = cfg.show_ticks and (
                    cfg.status_every <= 0 or tick_index % cfg.status_every == 0
                )
                if should_print_tick and writer is not None:
                    writer.write(
                        f"\n[tick {tick_index}] mode={final_mode} "
                        f"pressure={final_pressure:.2f} text={final_text}\n"
                    )
                    writer.write(cfg.input_prompt)
                    writer.flush()
                    last_status_tick = tick_index

                if cfg.stop_on_ask_user:
                    expr = tick_record_dict.get("metadata", {}).get("expression_recommendation")
                    if isinstance(expr, dict) and str(expr.get("mode")) == "ask_user":
                        stopped_early = True
                        stop_reason = "expression_ask_user"
                        if writer is not None:
                            writer.write(
                                "\n[warning] Expression Gate requested user attention.\n"
                            )
                        break
                if cfg.stop_on_verifier_critical and _is_verifier_critical(tick_record_dict):
                    stopped_early = True
                    stop_reason = "verifier_escalation"
                    if writer is not None:
                        writer.write("\n[warning] Verifier escalation detected.\n")
                    break
                shared_reason = stop_tracker.evaluate(tick_record, tick_summary)
                if shared_reason:
                    stopped_early = True
                    stop_reason = shared_reason
                    if writer is not None:
                        writer.write(
                            f"\n[warning] Tick safety stop triggered: {shared_reason}\n"
                        )
                    break

                last_tick_at = now
                next_tick_at = max(next_tick_at + cfg.interval_seconds, now)

            wait_for_input = max(0.0, min(0.05, next_tick_at - time.monotonic()))
            line = provider.poll_line(wait_for_input)
            if line is None:
                continue
            raw = str(line)
            if not raw.strip():
                continue
            command = raw.strip().casefold()

            if command == "/help":
                if writer is not None:
                    writer.write(
                        "\nCommands: /status, /help, /quit, /ticks on, /ticks off\n"
                    )
                    writer.write(
                        f"Exit commands: {', '.join(cfg.exit_commands)}\n\n{cfg.input_prompt}"
                    )
                    writer.flush()
                continue
            if command == "/status":
                if writer is not None:
                    writer.write(
                        "\n"
                        f"[status] tick={tick_index} inputs={input_index} "
                        f"mode={final_mode} pressure={final_pressure:.2f} "
                        f"latent={final_latent_pressure:.2f} "
                        f"expression={final_expression_mode} decision={final_decision}\n\n"
                        f"{cfg.input_prompt}"
                    )
                    writer.flush()
                continue
            if command == "/ticks on":
                cfg.show_ticks = True
                if writer is not None:
                    writer.write("\n[status] tick output enabled\n\n")
                    writer.write(cfg.input_prompt)
                    writer.flush()
                continue
            if command == "/ticks off":
                cfg.show_ticks = False
                if writer is not None:
                    writer.write("\n[status] tick output disabled\n\n")
                    writer.write(cfg.input_prompt)
                    writer.flush()
                continue
            if command == "/quit":
                stopped_early = True
                stop_reason = "exit_command"
                break

            if command in exit_commands:
                stopped_early = True
                stop_reason = "exit_command"
                break

            input_index += 1
            user_event = Event(
                event_type=EventType.USER_MESSAGE,
                content=raw,
                metadata={
                    **(extra_metadata or {}),
                    "interactive_input": True,
                    "input_index": input_index,
                    "suppress_expression": False,
                    "session_id": session_id,
                },
            )
            working_turn_index += 1
            user_working_id = _store_working_turn(
                memory_store=memory_store,
                content=raw,
                session_id=session_id,
                turn_index=working_turn_index,
                dialogue_role="user",
                source_event_id=user_event.event_id,
                created_from="interactive",
                metadata={"event_id": user_event.event_id},
            )
            if user_working_id is not None:
                working_turns_stored += 1
            try:
                user_record = runtime.process_event(user_event)
            except KeyboardInterrupt:
                stopped_early = True
                stop_reason = "keyboard_interrupt"
                break
            except Exception as exc:
                stopped_early = True
                stop_reason = f"runtime_exception:{exc.__class__.__name__}"
                break

            user_summary = summarize_tick_record(
                user_record,
                state=runtime.state,
                include_presentation=True,
            )
            user_pv = user_summary.get("presentation_vector") or {}
            final_mode = str(user_pv.get("mode") or "unknown")
            final_pressure = float(user_summary.get("system_pressure") or 0.0)
            final_latent_pressure = float(user_summary.get("latent_pressure") or 0.0)
            final_expression_mode = str(user_summary.get("expression_mode") or "silent")
            final_decision = str(user_summary.get("decision") or "wait")
            if callable(user_output_builder):
                final_text = user_output_builder(user_record) or "(silent)"
            else:
                final_text = "(silent)"
            if final_text.strip():
                working_turn_index += 1
                assistant_working_id = _store_working_turn(
                    memory_store=memory_store,
                    content=final_text,
                    session_id=session_id,
                    turn_index=working_turn_index,
                    dialogue_role="assistant",
                    source_event_id=user_event.event_id,
                    created_from="interactive",
                    metadata={
                        "response_to_event_id": user_event.event_id,
                        "paired_turn_id": user_working_id,
                    },
                )
                if assistant_working_id is not None:
                    working_turns_stored += 1
            if memory_store is not None:
                _ = memory_store.prune_working_memory(
                    session_id=session_id,
                    keep_last=cfg.working_memory_retain_turns,
                )

            if writer is not None:
                writer.write(
                    "\n"
                    f"You: {raw.strip()}\n"
                    f"Fullerene: {final_text}\n"
                    f"[status] mode={final_mode} pressure={final_pressure:.2f}\n\n"
                )
                writer.write(cfg.input_prompt)
                writer.flush()
                last_status_tick = tick_index

            if cfg.interval_seconds > 0:
                next_tick_at = max(next_tick_at, last_tick_at + cfg.interval_seconds)

        if tick_index >= cfg.max_ticks and stop_reason is None:
            stop_reason = "max_ticks_reached"
    finally:
        if created_provider:
            provider.close()

    if writer is not None:
        if tick_index > last_status_tick and stop_reason not in {"exit_command", None}:
            writer.write(
                f"[status] mode={final_mode} pressure={final_pressure:.2f}\n"
            )
        writer.write(f"Stopped: {stop_reason or 'completed'}\n")
        writer.flush()

    return InteractiveLoopResult(
        ticks_processed=tick_index,
        inputs_processed=input_index,
        stopped_early=bool(stopped_early),
        stop_reason=stop_reason,
        final_mode=final_mode,
        final_pressure=final_pressure,
        final_text_output=final_text,
        metadata={
            "interval_seconds": cfg.interval_seconds,
            "max_ticks": cfg.max_ticks,
            "allow_expression": cfg.allow_expression,
            "allow_model": cfg.allow_model,
            "last_tick": tick_index,
            "last_mode": final_mode,
            "last_pressure": final_pressure,
            "last_latent_pressure": final_latent_pressure,
            "last_expression_mode": final_expression_mode,
            "last_text_output": final_text,
            "last_decision": final_decision,
            "inputs_processed": input_index,
            "session_id": session_id,
            "working_memory_turns_stored": working_turns_stored,
            "last_working_memory_turn_index": working_turn_index,
        },
    )


def _resolve_memory_store(runtime: NexusRuntime):
    facets = getattr(runtime, "facets", ())
    for facet in facets:
        if getattr(facet, "name", "") != "memory":
            continue
        store = getattr(facet, "store", None)
        if store is None:
            continue
        required = ("add_working_turn", "prune_working_memory")
        if all(hasattr(store, method) for method in required):
            return store
    return None


def _store_working_turn(
    *,
    memory_store,
    content: str,
    session_id: str,
    turn_index: int,
    dialogue_role: str,
    source_event_id: str | None,
    created_from: str,
    metadata: dict[str, Any],
) -> str | None:
    if memory_store is None:
        return None
    record = memory_store.add_working_turn(
        content=content,
        session_id=session_id,
        turn_index=turn_index,
        dialogue_role=dialogue_role,
        source_event_id=source_event_id,
        created_from=created_from,
        metadata=metadata,
    )
    return record.id
