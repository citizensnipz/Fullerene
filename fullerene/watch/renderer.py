from __future__ import annotations

import json
from typing import Any

from fullerene.watch.models import WatchConfig, WatchSnapshot


_ANSI_CLEAR_AND_HOME = "\033[2J\033[H"


def clear_screen_sequence() -> str:
    return _ANSI_CLEAR_AND_HOME


def _fmt_f(v: Any, *, ndigits: int = 2) -> str:
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return "0.00"


def render_watch_snapshot(snapshot: WatchSnapshot, config: WatchConfig) -> str:
    tick = f"tick {snapshot.tick_index}/{snapshot.tick_count}"

    parts: list[str] = [tick]
    if config.show_presentation:
        parts.append(
            f"mode={snapshot.presentation_mode}"
            f" motion={snapshot.presentation_motion}"
            f" intensity={_fmt_f(snapshot.presentation_intensity)}"
        )
    if config.show_pressure:
        parts.append(
            f"pressure={_fmt_f(snapshot.system_pressure)}/{_fmt_f(snapshot.latent_pressure)}"
        )
    if config.show_expression:
        parts.append(
            f"expression={snapshot.expression_mode}"
            f" suppressed={str(snapshot.expression_suppressed).lower()}"
        )
    if config.show_interrupts:
        allowed = snapshot.allowed_interrupt_type or "none"
        parts.append(
            f"interrupts={snapshot.interrupt_candidates_count}/"
            f"{snapshot.suppressed_interrupt_count}"
            f" allowed={allowed}"
        )
    parts.append(f"decision={snapshot.decision}")

    return " ".join(parts)


def render_watch_trace(snapshot: WatchSnapshot) -> str | None:
    md = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}

    top_latent = md.get("top_latent_entry")
    top_latent_str = ""
    if isinstance(top_latent, dict) and top_latent:
        et = top_latent.get("entry_type") or top_latent.get("type") or "unknown"
        eid = top_latent.get("id") or top_latent.get("entry_id")
        top_latent_str = f"top_latent={et}:{eid or 'none'}"

    expr_reasons = md.get("expression_reasons")
    expr_reasons_str = ""
    if isinstance(expr_reasons, list) and expr_reasons:
        expr_reasons_str = "expr_reasons=[" + ",".join(str(r) for r in expr_reasons[:5]) + "]"

    expr_supp_reason = md.get("expression_suppression_reason") or ""
    expr_supp_str = (
        f"expr_suppression_reason={expr_supp_reason}" if expr_supp_reason else ""
    )

    allowed_reason = md.get("allowed_interrupt_reason") or ""
    allowed_reason_str = (
        f"allowed_interrupt_reason={allowed_reason}" if allowed_reason else ""
    )

    stop_state = ""
    if snapshot.stop_reason:
        stop_state = f"stop_reason={snapshot.stop_reason} stopped_early={str(snapshot.stopped_early).lower()}"

    frag_parts = [p for p in (top_latent_str, expr_reasons_str, expr_supp_str, allowed_reason_str, stop_state) if p]
    if not frag_parts:
        return None
    return "trace: " + " ".join(frag_parts)


def render_watch_json_payload(watch_run: dict[str, Any]) -> str:
    return json.dumps(watch_run, indent=2)

