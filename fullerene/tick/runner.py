"""Manual Tick Runner v0 — bounded explicit SYSTEM_TICK sequences (no background loop)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fullerene.nexus.models import Event, EventType, NexusRecord, NexusState
from fullerene.nexus.runtime import NexusRuntime
from fullerene.presentation import derive_presentation_vector

# Hard safety cap for CLI --ticks (reject above this).
TICK_HARD_CAP = 100

_HIGH_PRESSURE = 0.95
HIGH_PRESSURE_CONSEC = 5
LATENT_ONLY_HIGH_PRESSURE_CONSEC = 20
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


def summarize_tick_record(
    record: NexusRecord,
    *,
    state: NexusState | None = None,
    include_presentation: bool = False,
) -> dict[str, Any]:
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

    summ: dict[str, Any] = {
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
    if include_presentation:
        pv = derive_presentation_vector(record, state)
        summ["presentation_vector"] = pv.to_dict()
    return summ


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


@dataclass(slots=True)
class TickStopTracker:
    consec_high: int = 0
    latent_saturation_streak: int = 0
    consec_critical: int = 0
    consec_ask_same: int = 0
    last_ask_key: tuple[str, str] | None = None

    def evaluate(self, record: NexusRecord, summary: dict[str, Any]) -> str | None:
        sp = float(summary.get("system_pressure") or 0.0)
        lp = float(summary.get("latent_pressure") or 0.0)
        danger = _has_danger_pressure_signals(record)
        pressure_kind = "danger" if danger else ("latent_only" if lp >= _HIGH_PRESSURE else "mixed")
        summary["stop_pressure_kind"] = pressure_kind
        if sp >= _HIGH_PRESSURE:
            self.consec_high += 1
        else:
            self.consec_high = 0
        summary["high_pressure_streak"] = self.consec_high

        if sp >= _HIGH_PRESSURE and lp >= _HIGH_PRESSURE and not danger:
            self.latent_saturation_streak += 1
        else:
            self.latent_saturation_streak = 0
        summary["latent_saturation_streak"] = self.latent_saturation_streak

        if danger and self.consec_high >= HIGH_PRESSURE_CONSEC:
            return "consecutive_high_system_pressure"
        if (
            not danger
            and self.latent_saturation_streak >= LATENT_ONLY_HIGH_PRESSURE_CONSEC
        ):
            return "consecutive_high_system_pressure_latent_only"

        dropped = int(summary.get("internal_events_dropped") or 0)
        if dropped > 0:
            return "internal_events_overflow"

        internal_ct = int(summary.get("internal_events_processed_count") or 0)
        if internal_ct > 1:
            return "internal_event_recursion"

        if _verifier_has_critical_failure(record):
            self.consec_critical += 1
        else:
            self.consec_critical = 0
        if self.consec_critical >= _VERIFIER_CRITICAL_CONSEC:
            return "repeated_verifier_critical"

        mode = str(summary.get("expression_mode") or "")
        cand = str(
            summary.get("expression_source_candidate_id")
            or summary.get("allowed_interrupt_candidate_id")
            or "",
        )
        if mode == "ask_user" and cand:
            key = (mode, cand)
            if self.last_ask_key == key:
                self.consec_ask_same += 1
            else:
                self.last_ask_key = key
                self.consec_ask_same = 1
            if self.consec_ask_same >= _ASK_USER_REPEAT_STOP:
                return "repeated_expression_ask_user_same_source"
        else:
            self.last_ask_key = None
            self.consec_ask_same = 0
        return None


def _has_danger_pressure_signals(record: NexusRecord) -> bool:
    md = record.metadata if isinstance(record.metadata, dict) else {}
    sig = md.get("signal_map") if isinstance(md.get("signal_map"), dict) else {}
    if bool(sig.get("policy_blocks_act")):
        return True
    if str(sig.get("policy_status") or "").lower() == "denied":
        return True
    if _verifier_has_critical_failure(record):
        return True
    for result in record.facet_results:
        if result.facet_name != "behavior" or not isinstance(result.metadata, dict):
            continue
        reason = str(result.metadata.get("interrupt_reason") or "").lower()
        if any(tok in reason for tok in ("critical", "danger", "safety")):
            return True
    return False


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
    include_presentation: bool = False,
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

    stop_tracker = TickStopTracker()

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

        summ = summarize_tick_record(
            record,
            state=runtime.state,
            include_presentation=include_presentation,
        )
        summaries.append(summ)

        if include_full_records:
            records_out.append(record.to_dict())

        reason = stop_tracker.evaluate(record, summ)
        if reason:
            stopped_early = True
            stop_idx = i
            stop_reason = reason
            break

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
