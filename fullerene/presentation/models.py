"""Presentation Vector v0 — UI-facing read-only projection (not a facet, not cognition)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def clamp01(value: Any) -> float:
    try:
        v = float(value)
        if v != v:  # noqa: PLC3002
            return 0.0
        return max(0.0, min(v, 1.0))
    except (TypeError, ValueError):
        return 0.0


class PresentationMode(str, Enum):
    idle = "idle"
    listening = "listening"
    thinking = "thinking"
    speaking = "speaking"
    blocked = "blocked"
    overloaded = "overloaded"
    verifying = "verifying"
    learning = "learning"
    warning = "warning"
    sleeping = "sleeping"
    unknown = "unknown"


class PresentationMotion(str, Enum):
    still = "still"
    blink = "blink"
    slow_blink = "slow_blink"
    pulse = "pulse"
    bounce = "bounce"
    jitter = "jitter"
    mouth_loop = "mouth_loop"
    ellipsis = "ellipsis"
    none = "none"


class PresentationChannel(str, Enum):
    none = "none"
    internal = "internal"
    status = "status"
    user_expression = "user_expression"
    ask_user = "ask_user"
    warning = "warning"


_PRESENTATION_MODES = frozenset(m.value for m in PresentationMode)
_PRESENTATION_MOTIONS = frozenset(m.value for m in PresentationMotion)
_PRESENTATION_CHANNELS = frozenset(c.value for c in PresentationChannel)


def _norm_mode(raw: Any) -> PresentationMode:
    s = str(raw or "").strip().lower()
    if s in _PRESENTATION_MODES:
        return PresentationMode(s)
    return PresentationMode.unknown


def _norm_motion(raw: Any) -> PresentationMotion:
    s = str(raw or "").strip().lower()
    if s in _PRESENTATION_MOTIONS:
        return PresentationMotion(s)
    return PresentationMotion.none


def _norm_channel(raw: Any) -> PresentationChannel:
    s = str(raw or "").strip().lower()
    if s in _PRESENTATION_CHANNELS:
        return PresentationChannel(s)
    return PresentationChannel.none


def _safe_str(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw)


def _bounded_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe compact metadata without huge nesting."""

    def _walk(obj: Any, d: int) -> Any:
        if d <= 0:
            return str(obj)[:128]
        if obj is None or isinstance(obj, (bool, str, int)):
            return obj
        if isinstance(obj, float):
            return 0.0 if obj != obj else round(obj, 6)  # noqa: PLC3002
        if isinstance(obj, dict):
            return {
                str(k): _walk(v, d - 1)
                for k, v in list(obj.items())[:48]
                if isinstance(k, (str, int))
            }
        if isinstance(obj, (list, tuple)):
            return [_walk(v, d - 1) for v in list(obj)[:32]]
        return str(obj)[:256]

    return _walk(dict(raw), 4) if isinstance(raw, dict) else {}


@dataclass(slots=True)
class PresentationVector:
    mode: PresentationMode
    intensity: float
    motion: PresentationMotion
    channel: PresentationChannel
    user_attention_needed: bool
    expression_active: bool
    expression_mode: str
    pressure: float
    latent_pressure: float
    confidence: float
    novelty: float
    attention_motion: float
    blocked: bool
    overloaded: bool
    warning: bool
    speaking: bool
    thinking: bool
    idle: bool
    face_state: str
    eye_state: str
    mouth_state: str
    animation_hint: str
    reason: str
    reasons: list[str]
    source_event_id: str
    source_event_type: str
    source_record_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "intensity": round(clamp01(self.intensity), 6),
            "motion": self.motion.value,
            "channel": self.channel.value,
            "user_attention_needed": bool(self.user_attention_needed),
            "expression_active": bool(self.expression_active),
            "expression_mode": _safe_str(self.expression_mode),
            "pressure": round(clamp01(self.pressure), 6),
            "latent_pressure": round(clamp01(self.latent_pressure), 6),
            "confidence": round(clamp01(self.confidence), 6),
            "novelty": round(clamp01(self.novelty), 6),
            "attention_motion": round(clamp01(self.attention_motion), 6),
            "blocked": bool(self.blocked),
            "overloaded": bool(self.overloaded),
            "warning": bool(self.warning),
            "speaking": bool(self.speaking),
            "thinking": bool(self.thinking),
            "idle": bool(self.idle),
            "face_state": _safe_str(self.face_state),
            "eye_state": _safe_str(self.eye_state),
            "mouth_state": _safe_str(self.mouth_state),
            "animation_hint": _safe_str(self.animation_hint),
            "reason": _safe_str(self.reason),
            "reasons": list(self.reasons)[:64],
            "source_event_id": _safe_str(self.source_event_id),
            "source_event_type": _safe_str(self.source_event_type),
            "source_record_id": _safe_str(self.source_record_id),
            "metadata": _bounded_metadata(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresentationVector:
        md = data.get("metadata")
        if not isinstance(md, dict):
            md = {}
        return cls(
            mode=_norm_mode(data.get("mode")),
            intensity=clamp01(data.get("intensity", 0.0)),
            motion=_norm_motion(data.get("motion")),
            channel=_norm_channel(data.get("channel")),
            user_attention_needed=bool(data.get("user_attention_needed", False)),
            expression_active=bool(data.get("expression_active", False)),
            expression_mode=_safe_str(data.get("expression_mode")),
            pressure=clamp01(data.get("pressure", 0.0)),
            latent_pressure=clamp01(data.get("latent_pressure", 0.0)),
            confidence=clamp01(data.get("confidence", 0.0)),
            novelty=clamp01(data.get("novelty", 0.0)),
            attention_motion=clamp01(data.get("attention_motion", 0.0)),
            blocked=bool(data.get("blocked", False)),
            overloaded=bool(data.get("overloaded", False)),
            warning=bool(data.get("warning", False)),
            speaking=bool(data.get("speaking", False)),
            thinking=bool(data.get("thinking", False)),
            idle=bool(data.get("idle", False)),
            face_state=_safe_str(data.get("face_state")),
            eye_state=_safe_str(data.get("eye_state")),
            mouth_state=_safe_str(data.get("mouth_state")),
            animation_hint=_safe_str(data.get("animation_hint")),
            reason=_safe_str(data.get("reason")),
            reasons=list(data.get("reasons") or []),
            source_event_id=_safe_str(data.get("source_event_id")),
            source_event_type=_safe_str(data.get("source_event_type")),
            source_record_id=_safe_str(data.get("source_record_id")),
            metadata=dict(md),
        )
