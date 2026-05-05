"""Optional embedding-index providers for Fullerene Memory v2.

Memory v2 uses embeddings purely as a *retrieval index*. SQLite memory rows
remain the source of truth; a missing or failing embedding index must always
fall back to deterministic v1 retrieval so tests can run offline.

Two providers ship with the runtime:

- :class:`DeterministicHashEmbeddingProvider` produces small token-hash
  vectors with no external dependency. The vectors are not semantically
  meaningful; their job is to exercise the embedding code path so tests pass
  in any environment.
- :class:`OllamaEmbeddingProvider` is a thin client for a locally running
  Ollama instance. It is opt-in only: nothing imports or starts Ollama
  unless a caller explicitly constructs one of these providers.

Embedding storage and similarity computation live alongside the providers so
the retrieval layer has a single import surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

# Tokenizer reused for hash embeddings. Splitting on the same boundaries as
# `fullerene.memory.scoring.tokenize` keeps deterministic vectors aligned with
# tag overlap and keyword overlap.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.casefold()) if len(token) >= 2]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal protocol for embedding providers.

    Implementations must:

    - Return a fixed-length ``list[float]`` for each call to :meth:`embed`.
    - Be deterministic when given identical input (so retrieval is stable).
    - Either raise :class:`EmbeddingProviderError` on failure or return an
      empty list. Callers always treat both as a soft fallback signal.
    """

    name: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for ``text`` or raise on failure."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce a vector.

    Retrieval treats this as a soft fallback signal: deterministic hybrid
    retrieval still runs without semantic similarity.
    """


@dataclass(slots=True)
class DeterministicHashEmbeddingProvider:
    """Hash-bag embedding provider for offline tests / fallback.

    Produces a fixed-length unit-norm vector by hashing tokens into buckets.
    The vector is not semantically meaningful; identical inputs produce
    identical vectors, and similar token sets share buckets, which is enough
    to exercise the similarity code path in tests.
    """

    name: str = "deterministic_hash_v0"
    dimensions: int = 64

    def __post_init__(self) -> None:
        self.dimensions = max(int(self.dimensions), 8)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimensions
        vector = [0.0] * self.dimensions
        for token in _token_set(text):
            bucket = (
                int.from_bytes(
                    hashlib.sha1(token.encode("utf-8")).digest()[:4],
                    "big",
                    signed=False,
                )
                % self.dimensions
            )
            sign = 1.0 if token[0] < "n" else -1.0
            vector[bucket] += sign * 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]


@dataclass(slots=True)
class OllamaEmbeddingProvider:
    """Optional thin client for a local Ollama embedding endpoint.

    This provider does not require Ollama to be installed at import time. It
    only attempts a network call when :meth:`embed` runs and raises
    :class:`EmbeddingProviderError` on any failure so retrieval can fall back
    to deterministic v1 scoring.
    """

    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    dimensions: int = 768
    timeout_seconds: float = 5.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimensions
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingProviderError(
                f"Ollama embeddings request failed: {exc}"
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingProviderError(
                f"Ollama embeddings response was not valid JSON: {exc}"
            ) from exc
        embedding = decoded.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingProviderError(
                "Ollama embeddings response missing 'embedding' array"
            )
        try:
            vector = [float(component) for component in embedding]
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"Ollama embeddings response had non-numeric components: {exc}"
            ) from exc
        if not vector:
            raise EmbeddingProviderError("Ollama returned an empty embedding")
        return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector is empty, dimensions disagree, or
    either norm is zero. Result is clamped to ``[-1.0, 1.0]``.
    """
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        a_value = float(a)
        b_value = float(b)
        dot += a_value * b_value
        left_norm += a_value * a_value
        right_norm += b_value * b_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    raw = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def build_embedding_provider(
    spec: str | None,
) -> EmbeddingProvider | None:
    """Resolve a CLI/config spec string into an embedding provider.

    Supported specs:

    - ``None`` or empty - return ``None`` (embeddings disabled).
    - ``"deterministic"`` or ``"hash"`` - in-process deterministic provider.
    - ``"ollama:<model>"`` - optional local Ollama provider.
    """
    if spec is None:
        return None
    cleaned = str(spec).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"deterministic", "hash", "deterministic_hash", "deterministic-hash"}:
        return DeterministicHashEmbeddingProvider()
    if lowered.startswith("ollama:"):
        model_name = cleaned.split(":", 1)[1].strip()
        if not model_name:
            return None
        return OllamaEmbeddingProvider(model=model_name)
    return None


def safe_embed(
    provider: EmbeddingProvider | None,
    text: str,
) -> tuple[list[float] | None, str | None]:
    """Embed ``text`` using ``provider`` and never raise.

    Returns ``(vector, error)``. ``vector`` is ``None`` when embeddings are
    disabled or the provider failed; ``error`` carries the exception message
    so callers can surface a single inspectable status string.
    """
    if provider is None:
        return None, None
    try:
        vector = provider.embed(text)
    except EmbeddingProviderError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - any provider failure is soft-fallback
        return None, f"{type(exc).__name__}: {exc}"
    if not vector:
        return None, "empty_vector"
    return list(vector), None


def serialize_vector(vector: Iterable[float]) -> str:
    """Return a JSON encoding for storage."""
    return json.dumps([float(value) for value in vector])


def deserialize_vector(raw: str | None) -> list[float] | None:
    """Decode a stored vector or return ``None`` when missing/invalid."""
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list):
        return None
    try:
        return [float(component) for component in decoded]
    except (TypeError, ValueError):
        return None
