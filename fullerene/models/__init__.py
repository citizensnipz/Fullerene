"""Optional model adapters for Fullerene CLI text output."""

from __future__ import annotations

from fullerene.models.base import ModelAdapter, ModelAdapterError
from fullerene.models.ollama import OllamaAdapter

__all__ = [
    "ModelAdapter",
    "ModelAdapterError",
    "OllamaAdapter",
]
