from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp_ticks(value: int, *, max_ticks: int) -> int:
    try:
        ticks = int(value)
    except (TypeError, ValueError):
        ticks = 10
    ticks = max(1, ticks)
    return min(max_ticks, ticks)


def _clamp_interval_seconds(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 1.0
    return max(0.0, min(60.0, v))


@dataclass(slots=True)
class WatchConfig:
    ticks: int = 10
    interval_seconds: float = 1.0
    clear_screen: bool = False
    show_trace: bool = False
    show_json: bool = False

    show_presentation: bool = True
    show_pressure: bool = True
    show_interrupts: bool = True
    show_expression: bool = True

    stop_on_tick_runner_stop: bool = True
    max_ticks: int = 100

    def clamped(self) -> "WatchConfig":
        return WatchConfig(
            ticks=_clamp_ticks(self.ticks, max_ticks=self.max_ticks),
            interval_seconds=_clamp_interval_seconds(self.interval_seconds),
            clear_screen=bool(self.clear_screen),
            show_trace=bool(self.show_trace),
            show_json=bool(self.show_json),
            show_presentation=bool(self.show_presentation),
            show_pressure=bool(self.show_pressure),
            show_interrupts=bool(self.show_interrupts),
            show_expression=bool(self.show_expression),
            stop_on_tick_runner_stop=bool(self.stop_on_tick_runner_stop),
            max_ticks=int(self.max_ticks) if self.max_ticks else 100,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            ticks=int(data.get("ticks", cls.ticks)),
            interval_seconds=float(data.get("interval_seconds", cls.interval_seconds)),
            clear_screen=bool(data.get("clear_screen", cls.clear_screen)),
            show_trace=bool(data.get("show_trace", cls.show_trace)),
            show_json=bool(data.get("show_json", cls.show_json)),
            show_presentation=bool(
                data.get("show_presentation", cls.show_presentation)
            ),
            show_pressure=bool(data.get("show_pressure", cls.show_pressure)),
            show_interrupts=bool(data.get("show_interrupts", cls.show_interrupts)),
            show_expression=bool(data.get("show_expression", cls.show_expression)),
            stop_on_tick_runner_stop=bool(
                data.get("stop_on_tick_runner_stop", cls.stop_on_tick_runner_stop)
            ),
            max_ticks=int(data.get("max_ticks", cls.max_ticks)),
        )


@dataclass(slots=True)
class WatchSnapshot:
    tick_index: int
    tick_count: int
    timestamp: str

    decision: str

    system_pressure: float
    latent_pressure: float

    presentation_mode: str
    presentation_motion: str
    presentation_intensity: float
    presentation_channel: str

    expression_mode: str
    expression_suppressed: bool

    interrupt_candidates_count: int
    suppressed_interrupt_count: int
    allowed_interrupt_type: str | None

    internal_event_processed: bool

    stop_reason: str | None
    stopped_early: bool

    summary_line: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WatchRunResult:
    tick_count: int
    stopped_early: bool
    stop_reason: str | None
    snapshots: list[WatchSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_count": int(self.tick_count),
            "stopped_early": bool(self.stopped_early),
            "stop_reason": self.stop_reason,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

