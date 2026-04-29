"""Base interface for optional text generation adapters."""

from __future__ import annotations


class ModelAdapterError(RuntimeError):
    """Raised when an optional model adapter cannot generate text."""


class ModelAdapter:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
