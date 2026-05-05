"""Latent pressure signal infrastructure exports."""

from fullerene.signals.latent_pressure.buffer import update_latent_pressure
from fullerene.signals.latent_pressure.models import (
    LatentPressureEntry,
    LatentPressureResult,
)

__all__ = [
    "LatentPressureEntry",
    "LatentPressureResult",
    "update_latent_pressure",
]

