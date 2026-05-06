from __future__ import annotations

from fullerene.continuous.models import ContinuousLoopConfig

_ANSI_CLEAR_AND_HOME = "\033[2J\033[H"


def render_loop_frame(
    *,
    tick_index: int,
    tick_count: int,
    mode: str,
    pressure: float,
    text_output: str,
    config: ContinuousLoopConfig,
) -> str:
    if not config.mode_pressure_only:
        return (
            f"tick={tick_index} mode={mode} pressure={pressure:.2f} text={text_output}"
        )
    if config.clear_screen:
        return (
            f"{_ANSI_CLEAR_AND_HOME}"
            f"Fullerene Loop | tick {tick_index}/{tick_count}\n"
            f"mode={mode} pressure={pressure:.2f}\n"
            f"text={text_output}"
        )
    return f"tick={tick_index} mode={mode} pressure={pressure:.2f} text={text_output}"
