"""Manual Tick Runner v0 — bounded explicit SYSTEM_TICK sequences (no background loop)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fullerene.nexus.models import Event, EventType, NexusRecord
from fullerene.nexus.runtime import NexusRuntime

# Hard safety cap for CLI --ticks (reject above this).
TICK_HARD_CAP = 100

_HIGH_PRESSURE = 0.95
_HIGH_PRESSURE_CONSEC = 5
_ASK_USER_REPEAT_STOP = 4
_VERIFIER_CRITICAL_CONSEC = 3


def build_tick_event_metadata(
    *,
    tick_index: int,
    tick_count: int,
    tick_reason: str | None,
    suppress_expression: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge CLI/user metadata with required manual tick fields (tick fields win)."""
    base: dict[str, Any] = {
        "manual_tick": True,
        "tick_index": int(tick_index),
        "tick_count": int(tick_count),
        "suppress_expression": bool(suppress_expression),
    }
    if tick_reason:
        base["tick_reason"] = str(tick_reason)
    if extra:
        merged = {**extra, **base}
        return merged
    return base


def summarize_tick_record(record: NexusRecord) -> dict[str, Any]:
    """Compact JSON-serializable summary for one processed tick."""
    md = record.metadata if isinstance(record.metadata, dict) else {}
    sig = md.get("signal_map") if isinstance(md.get("signal_map"), dict) else {}
    lpr = md.get("latent_pressure_result")
    if not isinstance(lpr, dict):
        lpr = {}
    top_entries = md.get("top_latent_pressure_entries")
    if not isinstance(top_entries, list):
        top_entries = lpr.get("top_entries") if isinstance(lpr.get("top_entries"), list) else []
    top_type: str | None = None
    top_id: str | None = None
    if top_entries and isinstance(top_entries[0], dict):
        top_type = top_entries[0].get("entry_type") or top_entries[0].get("type")
        top_id = top_entries[0].get("id") or top_entries[0].get("entry_id")

    icand = md.get("interrupt_candidates")
    icount = len(icand) if isinstance(icand, list) else 0
    allowed = md.get("allowed_interrupt_candidate")
    allowed_id = None
    if isinstance(allowed, dict):
        allowed_id = allowed.get("id")

    sd = md.get("suppression_decisions")
    suppressed_ct = 0
    if isinstance(sd, list):
        for row in sd:
            if isinstance(row, dict) and row.get("suppressed") is True:
                suppressed_ct += 1

    ev_meta = record.event.metadata if isinstance(record.event.metadata, dict) else {}
    expr_reco = md.get("expression_recommendation")
    if not isinstance(expr_reco, dict):
        expr_reco = {}
    expr_source_cand = expr_reco.get("source_candidate_id")

    internal_processed = md.get("internal_events_processed")
    internal_ct = len(internal_processed) if isinstance(internal_processed, list) else 0

    return {
        "tick_index": ev_meta.get("tick_index"),
        "tick_count": ev_meta.get("tick_count"),
        "decision": record.decision.action.value,
        "system_pressure": float(md.get("system_pressure", sig.get("system_pressure", 0.0)) or 0.0),
        "latent_pressure": float(
            md.get("latent_pressure", sig.get("latent_pressure", 0.0)) or 0.0,
        ),
        "top_latent_entry_type": top_type,
        "top_latent_entry_id": top_id,
        "interrupt_candidates_count": icount,
        "allowed_interrupt_candidate_id": allowed_id,
        "suppressed_interrupt_count": suppressed_ct,
        "expression_mode": md.get("expression_mode") or expr_reco.get("mode"),
        "expression_suppressed": md.get("expression_suppressed", expr_reco.get("suppressed")),
        "expression_source_candidate_id": expr_source_cand,
        "internal_event_processed": bool(sig.get("internal_event_processed")),
        "internal_events_processed_count": internal_ct,
        "verifier_status": str(sig.get("verifier_status", "unknown") or "unknown"),
        "internal_events_dropped": int(md.get("internal_events_dropped") or 0),
    }


@dataclass(slots=True)
class TickRunResult:
    tick_count: int
    records: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    final_state_summary: dict[str, Any]
    stopped_early: bool
    stop_reason: str | None
    stop_tick_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verifier_has_critical_failure(record: NexusRecord) -> bool:
    for fr in record.facet_results:
        if fr.facet_name != "verifier":
            continue
        meta = fr.metadata if isinstance(fr.metadata, dict) else {}
        for key in ("artifact_checks", "schema_checks"):
            rows = meta.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sev = str(row.get("severity") or "").lower()
                st = str(row.get("status") or "").lower()
                if st == "failed" and sev in {"critical", "error"}:
                    return True
        if meta.get("verification_status") == "failed":
            return True
    return False


def run_manual_ticks(
    runtime: NexusRuntime,
    *,
    total_ticks: int,
    tick_reason: str | None = None,
    suppress_expression: bool = True,
    extra_metadata: dict[str, Any] | None = None,
    include_full_records: bool = True,
) -> TickRunResult:
    """
    Run ``total_ticks`` SYSTEM_TICK events in sequence on ``runtime``.

    Stop conditions are conservative safety rails; they set ``stopped_early`` and
    ``stop_reason`` without raising (except unexpected ``process_event`` errors
    which are caught and reported).
    """
    summaries: list[dict[str, Any]] = []
    records_out: list[dict[str, Any]] = []
    stop_reason: str | None = None
    stopped_early = False
    stop_idx: int | None = None

    consec_high = 0
    consec_critical = 0
    consec_ask_same = 0
    last_ask_key: tuple[str, str] | None = None

    for i in range(1, total_ticks + 1):
        meta = build_tick_event_metadata(
            tick_index=i,
            tick_count=total_ticks,
            tick_reason=tick_reason,
            suppress_expression=suppress_expression,
            extra=extra_metadata,
        )
        event = Event(event_type=EventType.SYSTEM_TICK, content="", metadata=meta)
        try:
            record = runtime.process_event(event)
        except Exception as exc:
            stopped_early = True
            stop_idx = i
            stop_reason = f"runtime_exception:{exc.__class__.__name__}"
            summaries.append({"tick_index": i, "error": str(exc)})
            if include_full_records:
                records_out.append({"error": str(exc), "tick_index": i})
            break

        summ = summarize_tick_record(record)
        summaries.append(summ)

        if include_full_records:
            records_out.append(record.to_dict())

        sp = float(summ.get("system_pressure") or 0.0)
        if sp >= _HIGH_PRESSURE:
            consec_high += 1
        else:
            consec_high = 0
        if consec_high >= _HIGH_PRESSURE_CONSEC:
            stopped_early = True
            stop_idx = i
            stop_reason = "consecutive_high_system_pressure"
            break

        dropped = int(summ.get("internal_events_dropped") or 0)
        if dropped > 0:
            stopped_early = True
            stop_idx = i
            stop_reason = "internal_events_overflow"
            break

        internal_ct = int(summ.get("internal_events_processed_count") or 0)
        if internal_ct > 1:
            stopped_early = True
            stop_idx = i
            stop_reason = "internal_event_recursion"
            break

        if _verifier_has_critical_failure(record):
            consec_critical += 1
        else:
            consec_critical = 0
        if consec_critical >= _VERIFIER_CRITICAL_CONSEC:
            stopped_early = True
            stop_idx = i
            stop_reason = "repeated_verifier_critical"
            break

        mode = str(summ.get("expression_mode") or "")
        cand = str(
            summ.get("expression_source_candidate_id")
            or summ.get("allowed_interrupt_candidate_id")
            or "",
        )
        if mode == "ask_user" and cand:
            key = (mode, cand)
            if last_ask_key == key:
                consec_ask_same += 1
            else:
                last_ask_key = key
                consec_ask_same = 1
            if consec_ask_same >= _ASK_USER_REPEAT_STOP:
                stopped_early = True
                stop_idx = i
                stop_reason = "repeated_expression_ask_user_same_source"
                break
        else:
            last_ask_key = None
            consec_ask_same = 0

    final_state = runtime.state
    final_summary: dict[str, Any] = {
        "event_count": final_state.event_count,
        "system_pressure": float(final_state.system_pressure or 0.0),
        "last_decision": (
            final_state.last_decision.action.value if final_state.last_decision else None
        ),
    }

    meta_out: dict[str, Any] = {
        "requested_ticks": total_ticks,
        "completed_ticks": len(summaries),
    }

    return TickRunResult(
        tick_count=len(summaries),
        records=records_out,
        summaries=summaries,
        final_state_summary=final_summary,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        stop_tick_index=stop_idx,
        metadata=meta_out,
    )
