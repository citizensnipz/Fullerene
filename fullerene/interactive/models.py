from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


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
        n = 1000
    return max(1, min(10000, n))


@dataclass(slots=True)
class InteractiveLoopConfig:
    interval_seconds: float = 1.0
    max_ticks: int = 1000
    clear_screen: bool = False
    allow_expression: bool = True
    allow_model: bool = False
    input_prompt: str = "> "
    exit_commands: list[str] = field(
        default_factory=lambda: ["exit", "quit", ":q", "/quit"]
    )
    stop_on_ask_user: bool = False
    stop_on_verifier_critical: bool = True
    show_mode_pressure_only: bool = True
    show_last_output: bool = True
    show_help_on_start: bool = True
    show_ticks: bool = False
    status_every: int = 0
    session_id: str = field(default_factory=lambda: f"interactive-{uuid4().hex}")
    working_memory_context_turns: int = 8
    working_memory_retain_turns: int = 20

    def clamped(self) -> "InteractiveLoopConfig":
        return InteractiveLoopConfig(
            interval_seconds=_clamp_interval(self.interval_seconds),
            max_ticks=_clamp_max_ticks(self.max_ticks),
            clear_screen=bool(self.clear_screen),
            allow_expression=bool(self.allow_expression),
            allow_model=bool(self.allow_model),
            input_prompt=str(self.input_prompt or "> "),
            exit_commands=[
                str(cmd).strip()
                for cmd in self.exit_commands
                if isinstance(cmd, str) and str(cmd).strip()
            ]
            or ["exit", "quit", ":q", "/quit"],
            stop_on_ask_user=bool(self.stop_on_ask_user),
            stop_on_verifier_critical=bool(self.stop_on_verifier_critical),
            show_mode_pressure_only=bool(self.show_mode_pressure_only),
            show_last_output=bool(self.show_last_output),
            show_help_on_start=bool(self.show_help_on_start),
            show_ticks=bool(self.show_ticks),
            status_every=max(0, int(self.status_every or 0)),
            session_id=str(self.session_id or f"interactive-{uuid4().hex}"),
            working_memory_context_turns=max(1, int(self.working_memory_context_turns or 8)),
            working_memory_retain_turns=max(1, int(self.working_memory_retain_turns or 20)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InteractiveLoopResult:
    ticks_processed: int
    inputs_processed: int
    stopped_early: bool
    stop_reason: str | None
    final_mode: str | None
    final_pressure: float
    final_text_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
