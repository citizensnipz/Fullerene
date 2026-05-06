from __future__ import annotations

import time
from typing import Any, IO

from fullerene.continuous.models import ContinuousLoopConfig, ContinuousLoopResult
from fullerene.continuous.renderer import render_loop_frame
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.presentation import derive_presentation_vector
from fullerene.tick import TickStopTracker, build_tick_event_metadata, summarize_tick_record


def _expression_budget_active(runtime: NexusRuntime) -> bool:
    nexus_state = runtime.state.facet_state.get("nexus")
    if not isinstance(nexus_state, dict):
        return False
    gate = nexus_state.get("expression_gate")
    if not isinstance(gate, dict):
        return False
    budget_state = gate.get("budget_state")
    if not isinstance(budget_state, dict):
        return False
    if int(budget_state.get("expression_count_window") or 0) > 0:
        return True
    cooldowns = budget_state.get("cooldowns")
    return isinstance(cooldowns, dict) and len(cooldowns) > 0


def _text_status_from_record(record_dict: dict[str, Any], summary: dict[str, Any]) -> str:
    md = record_dict.get("metadata")
    if not isinstance(md, dict):
        md = {}
    expr = md.get("expression_recommendation")
    if isinstance(expr, dict):
        mode = str(expr.get("mode") or "")
        suppressed = bool(expr.get("suppressed"))
        intent = str(expr.get("suggested_intent") or "").strip()
        reason = ""
        reasons = expr.get("reasons")
        if isinstance(reasons, list) and reasons:
            reason = str(reasons[0])[:80]
        if not suppressed and mode in {"ask_user", "short_utterance"}:
            head = f"{mode}: {intent}" if intent else mode
            return f"{head} ({reason})" if reason else head

    for fr in record_dict.get("facet_results", []):
        if not isinstance(fr, dict) or fr.get("facet_name") != "verifier":
            continue
        vmd = fr.get("metadata")
        if not isinstance(vmd, dict):
            continue
        rows = vmd.get("artifact_checks")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sev = str(row.get("severity") or "").lower()
            st = str(row.get("status") or "").lower()
            if st == "failed" and sev in {"critical", "error"}:
                msg = str(row.get("message") or "").strip()
                return f"verifier escalation: {msg}" if msg else "verifier escalation"

    sig = md.get("signal_map")
    if isinstance(sig, dict):
        if str(sig.get("policy_status") or "") == "approval_required":
            return "approval required"

    pv = summary.get("presentation_vector")
    if isinstance(pv, dict):
        channel = str(pv.get("channel") or "").strip()
        if channel:
            return f"channel={channel}"
    return "(silent)"


def run_continuous_loop(
    runtime: NexusRuntime,
    config: ContinuousLoopConfig,
    *,
    output_writer: IO[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ContinuousLoopResult:
    cfg = config.clamped()
    writer = output_writer
    stop_tracker = TickStopTracker()
    summaries: list[dict[str, Any]] = []
    stop_reason: str | None = None
    stopped_early = False
    final_mode = "unknown"
    final_pressure = 0.0
    final_text = "(silent)"

    for idx in range(1, cfg.max_ticks + 1):
        suppress_expression = cfg.suppress_expression_by_default
        if _expression_budget_active(runtime):
            suppress_expression = False
        if cfg.allow_tick_expression:
            suppress_expression = False

        meta = build_tick_event_metadata(
            tick_index=idx,
            tick_count=cfg.max_ticks,
            tick_reason="continuous_loop_v0",
            suppress_expression=suppress_expression,
            extra=extra_metadata,
        )
        event = Event(event_type=EventType.SYSTEM_TICK, content="", metadata=meta)
        try:
            record = runtime.process_event(event)
        except KeyboardInterrupt:
            stopped_early = True
            stop_reason = "keyboard_interrupt"
            break
        except Exception as exc:
            stopped_early = True
            stop_reason = f"runtime_exception:{exc.__class__.__name__}"
            break

        summary = summarize_tick_record(
            record,
            state=runtime.state,
            include_presentation=True,
        )
        record_dict = record.to_dict()
        pv = summary.get("presentation_vector") or {}
        mode = str(pv.get("mode") or "unknown")
        pressure = float(summary.get("system_pressure") or 0.0)
        text_out = _text_status_from_record(record_dict, summary)

        summary["loop_mode"] = mode
        summary["loop_text"] = text_out
        summaries.append(summary)
        final_mode = mode
        final_pressure = pressure
        final_text = text_out

        if writer is not None:
            writer.write(
                render_loop_frame(
                    tick_index=idx,
                    tick_count=cfg.max_ticks,
                    mode=mode,
                    pressure=pressure,
                    text_output=text_out,
                    config=cfg,
                )
                + "\n"
            )
            writer.flush()

        if cfg.stop_on_tick_runner_stop:
            shared_reason = stop_tracker.evaluate(record, summary)
            if shared_reason:
                stopped_early = True
                stop_reason = shared_reason
                break

        if cfg.stop_on_ask_user and isinstance(record_dict.get("metadata"), dict):
            expr = record_dict["metadata"].get("expression_recommendation")
            if isinstance(expr, dict) and str(expr.get("mode")) == "ask_user":
                stopped_early = True
                stop_reason = "expression_ask_user"
                break

        if cfg.stop_on_verifier_critical and "verifier escalation" in text_out:
            stopped_early = True
            stop_reason = "verifier_escalation"
            break

        if idx < cfg.max_ticks and cfg.interval_seconds > 0:
            try:
                time.sleep(cfg.interval_seconds)
            except KeyboardInterrupt:
                stopped_early = True
                stop_reason = "keyboard_interrupt"
                break

    if writer is not None and stop_reason:
        writer.write(f"Stopped: {stop_reason}\n")

    return ContinuousLoopResult(
        tick_count=len(summaries),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        summaries=summaries,
        final_mode=final_mode,
        final_pressure=final_pressure,
        final_text_output=final_text,
        metadata={"requested_ticks": cfg.max_ticks, "interval_seconds": cfg.interval_seconds},
    )
