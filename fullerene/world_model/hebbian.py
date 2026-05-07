"""Deterministic Hebbian-style belief edge weight updates (World Model v2)."""


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def strengthen(weight: float, amount: float) -> float:
    """new_weight = weight + (1 - weight) * amount; amount clamped to [0,1]."""
    w = clamp01(weight)
    a = clamp01(amount)
    return clamp01(w + (1.0 - w) * a)


def weaken(weight: float, amount: float) -> float:
    """new_weight = weight - weight * amount."""
    w = clamp01(weight)
    a = clamp01(amount)
    return clamp01(w - w * a)
