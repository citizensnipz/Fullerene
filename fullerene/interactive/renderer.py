from __future__ import annotations

from fullerene.interactive.models import InteractiveLoopConfig

_ANSI_CLEAR_AND_HOME = "\033[2J\033[H"


def render_interactive_frame(
    *,
    tick_index: int,
    input_index: int,
    mode: str,
    pressure: float,
    text_output: str,
    config: InteractiveLoopConfig,
) -> str:
    if not config.show_mode_pressure_only:
        return (
            f"tick={tick_index} inputs={input_index} "
            f"mode={mode} pressure={pressure:.2f} text={text_output}"
        )
    if config.clear_screen:
        lines = [
            f"{_ANSI_CLEAR_AND_HOME}Fullerene Interactive | tick {tick_index} | inputs {input_index}",
            f"mode={mode} pressure={pressure:.2f}",
        ]
        if config.show_last_output:
            lines.append(f"text={text_output}")
        lines.append("")
        lines.append(config.input_prompt)
        return "\n".join(lines)
    return f"tick={tick_index} mode={mode} pressure={pressure:.2f} text={text_output}"
