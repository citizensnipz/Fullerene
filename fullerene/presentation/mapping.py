"""Renderer-neutral hints for Presentation Vector v0."""

from __future__ import annotations

from fullerene.presentation.models import PresentationMode, PresentationMotion


def animation_hint_for(
    mode: PresentationMode,
    motion: PresentationMotion,
    *,
    is_system_tick: bool,
    intensity: float,
) -> str:
    if mode == PresentationMode.sleeping:
        return "sleeping_breathe"
    if mode == PresentationMode.warning:
        return "warning_pulse"
    if mode == PresentationMode.overloaded:
        return "overloaded_jitter"
    if mode == PresentationMode.blocked:
        return "blocked_hold"
    if mode == PresentationMode.verifying:
        return "verifying_scan" if intensity >= 0.55 else "pulse"
    if mode == PresentationMode.learning:
        return "learning_pulse"
    if mode == PresentationMode.speaking:
        return "speaking_loop"
    if mode == PresentationMode.thinking:
        return "thinking_ellipsis" if is_system_tick else "thinking_pulse"
    if mode == PresentationMode.idle:
        return "idle_blink"
    if mode == PresentationMode.listening:
        return "idle_blink"
    return "none"


def face_state_for(mode: PresentationMode) -> str:
    m = {
        PresentationMode.idle: "neutral",
        PresentationMode.listening: "focused",
        PresentationMode.thinking: "thinking",
        PresentationMode.speaking: "talking",
        PresentationMode.blocked: "blocked",
        PresentationMode.overloaded: "overloaded",
        PresentationMode.verifying: "focused",
        PresentationMode.learning: "focused",
        PresentationMode.warning: "alert",
        PresentationMode.sleeping: "sleeping",
        PresentationMode.unknown: "unknown",
    }
    return m.get(mode, "unknown")


def eye_state_for(mode: PresentationMode, motion: PresentationMotion) -> str:
    if motion == PresentationMotion.slow_blink:
        return "half_lidded"
    if motion in (PresentationMotion.blink,):
        return "blink"
    if mode == PresentationMode.sleeping:
        return "closed"
    if mode in (PresentationMode.thinking, PresentationMode.verifying):
        return "focused"
    if mode == PresentationMode.overloaded:
        return "wide"
    if mode == PresentationMode.warning:
        return "wide"
    if mode == PresentationMode.speaking:
        return "open"
    if mode == PresentationMode.listening:
        return "focused"
    return "open"


def mouth_state_for(mode: PresentationMode, motion: PresentationMotion) -> str:
    if mode == PresentationMode.speaking:
        return "speaking"
    if motion == PresentationMotion.mouth_loop:
        return "speaking"
    if mode in (PresentationMode.thinking, PresentationMode.verifying):
        return "small_open"
    if mode == PresentationMode.warning:
        return "small_open"
    if mode == PresentationMode.overloaded:
        return "line"
    if mode == PresentationMode.blocked:
        return "line"
    if mode == PresentationMode.sleeping:
        return "closed"
    return "closed"


def motion_for_mode(
    mode: PresentationMode,
    *,
    is_system_tick: bool,
    intensity: float,
) -> PresentationMotion:
    if mode == PresentationMode.unknown:
        return PresentationMotion.still
    if mode == PresentationMode.sleeping:
        return PresentationMotion.bounce
    if mode == PresentationMode.warning:
        return PresentationMotion.pulse
    if mode == PresentationMode.overloaded:
        return PresentationMotion.jitter
    if mode == PresentationMode.blocked:
        return PresentationMotion.slow_blink
    if mode == PresentationMode.verifying:
        return PresentationMotion.pulse if intensity < 0.6 else PresentationMotion.bounce
    if mode == PresentationMode.learning:
        return PresentationMotion.pulse
    if mode == PresentationMode.speaking:
        return PresentationMotion.mouth_loop
    if mode == PresentationMode.thinking:
        return PresentationMotion.ellipsis if is_system_tick else PresentationMotion.pulse
    if mode == PresentationMode.listening:
        return PresentationMotion.slow_blink
    if mode == PresentationMode.idle:
        return PresentationMotion.blink
    return PresentationMotion.still
