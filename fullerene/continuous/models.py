from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp_interval(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 1.0
    return max(0.1, min(60.0, v))


def _clamp_max_ticks(value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 100
    return max(1, min(1000, n))


@dataclass(slots=True)
class ContinuousLoopConfig:
    interval_seconds: float = 1.0
    max_ticks: int = 100
    clear_screen: bool = True
    show_text_output: bool = True
    show_expression_status: bool = True
    suppress_expression_by_default: bool = True
    allow_tick_expression: bool = False
    stop_on_tick_runner_stop: bool = True
    stop_on_ask_user: bool = False
    stop_on_verifier_critical: bool = True
    stop_on_keyboard_interrupt: bool = True
    mode_pressure_only: bool = True

    def clamped(self) -> "ContinuousLoopConfig":
        return ContinuousLoopConfig(
            interval_seconds=_clamp_interval(self.interval_seconds),
            max_ticks=_clamp_max_ticks(self.max_ticks),
            clear_screen=bool(self.clear_screen),
            show_text_output=bool(self.show_text_output),
            show_expression_status=bool(self.show_expression_status),
            suppress_expression_by_default=bool(self.suppress_expression_by_default),
            allow_tick_expression=bool(self.allow_tick_expression),
            stop_on_tick_runner_stop=bool(self.stop_on_tick_runner_stop),
            stop_on_ask_user=bool(self.stop_on_ask_user),
            stop_on_verifier_critical=bool(self.stop_on_verifier_critical),
            stop_on_keyboard_interrupt=bool(self.stop_on_keyboard_interrupt),
            mode_pressure_only=bool(self.mode_pressure_only),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContinuousLoopResult:
    tick_count: int
    stopped_early: bool
    stop_reason: str | None
    summaries: list[dict[str, Any]]
    final_mode: str | None
    final_pressure: float
    final_text_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
