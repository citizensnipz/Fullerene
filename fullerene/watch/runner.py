from __future__ import annotations

import json
import sys
import time
from typing import Any, IO

from fullerene.tick.runner import TickRunResult, run_manual_ticks
from fullerene.nexus.runtime import NexusRuntime
from fullerene.watch.models import WatchConfig, WatchRunResult, WatchSnapshot
from fullerene.watch.renderer import (
    clear_screen_sequence,
    render_watch_snapshot,
    render_watch_trace,
)


def _record_metadata(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    md = record.get("metadata")
    return md if isinstance(md, dict) else {}


def _parse_timestamp(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return ""
    ts = record.get("timestamp")
    return str(ts) if ts is not None else ""


def build_watch_snapshots(
    tick_run_result: TickRunResult,
    config: WatchConfig,
) -> list[WatchSnapshot]:
    stopped_early = bool(tick_run_result.stopped_early)
    stop_reason = tick_run_result.stop_reason
    records = tick_run_result.records or []
    snapshots: list[WatchSnapshot] = []

    for idx, summary in enumerate(tick_run_result.summaries):
        if not isinstance(summary, dict):
            continue

        record = records[idx] if idx < len(records) else None
        md = _record_metadata(record)

        pv = summary.get("presentation_vector")
        pv_dict: dict[str, Any] = pv if isinstance(pv, dict) else {}

        allowed = md.get("allowed_interrupt_candidate")
        allowed_interrupt_type: str | None = None
        allowed_reason: str | None = None
        if isinstance(allowed, dict):
            raw_type = allowed.get("interrupt_type")
            allowed_interrupt_type = str(raw_type) if raw_type else None
            raw_reason = allowed.get("reason")
            allowed_reason = str(raw_reason) if raw_reason else None

        expr_reco = md.get("expression_recommendation")
        expr_reasons: list[str] | None = None
        expr_supp_reason: str | None = None
        if isinstance(expr_reco, dict):
            rr = expr_reco.get("reasons")
            if isinstance(rr, list):
                expr_reasons = [str(x) for x in rr if x is not None]
            sr = expr_reco.get("suppression_reason")
            expr_supp_reason = str(sr) if sr else None

        # Latent pressure "top entry" for trace purposes.
        top_latent_entry = None
        raw_top = md.get("top_latent_pressure_entries")
        if isinstance(raw_top, list) and raw_top:
            if isinstance(raw_top[0], dict):
                top_latent_entry = dict(raw_top[0])
        elif isinstance(md.get("latent_pressure_result"), dict):
            lpr = md.get("latent_pressure_result")
            top_entries = lpr.get("top_entries") if isinstance(lpr, dict) else None
            if isinstance(top_entries, list) and top_entries and isinstance(
                top_entries[0], dict
            ):
                top_latent_entry = dict(top_entries[0])

        snapshot = WatchSnapshot(
            tick_index=int(summary.get("tick_index") or (idx + 1)),
            tick_count=int(summary.get("tick_count") or config.ticks),
            timestamp=_parse_timestamp(record),
            decision=str(summary.get("decision") or "unknown"),
            system_pressure=float(summary.get("system_pressure") or 0.0),
            latent_pressure=float(summary.get("latent_pressure") or 0.0),
            presentation_mode=str(pv_dict.get("mode") or "unknown"),
            presentation_motion=str(pv_dict.get("motion") or "none"),
            presentation_intensity=float(pv_dict.get("intensity") or 0.0),
            presentation_channel=str(pv_dict.get("channel") or "none"),
            expression_mode=str(summary.get("expression_mode") or "silent"),
            expression_suppressed=bool(summary.get("expression_suppressed", False)),
            interrupt_candidates_count=int(summary.get("interrupt_candidates_count") or 0),
            suppressed_interrupt_count=int(
                summary.get("suppressed_interrupt_count") or 0
            ),
            allowed_interrupt_type=allowed_interrupt_type,
            internal_event_processed=bool(
                summary.get("internal_event_processed") or False
            ),
            stop_reason=stop_reason,
            stopped_early=stopped_early,
            summary_line="",
            metadata={
                "top_latent_entry": top_latent_entry,
                "expression_reasons": expr_reasons,
                "expression_suppression_reason": expr_supp_reason,
                "allowed_interrupt_reason": allowed_reason,
                "stop_tick_index": tick_run_result.stop_tick_index,
                "presentation_vector": pv_dict if config.show_presentation else {},
            },
        )

        snapshot.summary_line = render_watch_snapshot(snapshot, config=config)
        snapshots.append(snapshot)

    return snapshots


def run_watch_mode(
    runtime: NexusRuntime,
    config: WatchConfig,
    *,
    output_writer: IO[str] | None = None,
    tick_run_extra_metadata: dict[str, Any] | None = None,
) -> WatchRunResult:
    cfg = config.clamped()
    writer: IO[str] = output_writer or sys.stdout

    # Manual Tick Runner owns the stop conditions (pressure streaks, verifier
    # criticals, repeated ask_user, internal overflow, etc).
    tick_result = run_manual_ticks(
        runtime,
        total_ticks=cfg.ticks,
        suppress_expression=True,
        extra_metadata=tick_run_extra_metadata,
        include_full_records=True,
        include_presentation=True,
    )

    snapshots = build_watch_snapshots(tick_result, cfg)
    watch_run = WatchRunResult(
        tick_count=tick_result.tick_count,
        stopped_early=bool(tick_result.stopped_early),
        stop_reason=tick_result.stop_reason,
        snapshots=snapshots,
    )

    if cfg.show_json:
        payload = {"watch_run": watch_run.to_dict()}
        writer.write(json.dumps(payload, indent=2))
        if not payload["watch_run"].get("snapshots"):
            writer.write("\n")
        return watch_run

    for i, snap in enumerate(snapshots):
        if cfg.clear_screen:
            writer.write(clear_screen_sequence())
        writer.write(snap.summary_line + "\n")
        trace_line = render_watch_trace(snap) if cfg.show_trace else None
        if trace_line:
            writer.write(trace_line + "\n")
        writer.flush()
        if cfg.interval_seconds > 0.0 and i < len(snapshots) - 1:
            time.sleep(cfg.interval_seconds)

    if watch_run.stopped_early and watch_run.stop_reason:
        writer.write(
            f"stop_reason={watch_run.stop_reason} tick={len(snapshots)}/{cfg.ticks}\n"
        )

    return watch_run

