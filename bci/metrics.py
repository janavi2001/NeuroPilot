"""BCI performance metrics."""

from __future__ import annotations

import math


def wolpaw_bits(accuracy: float, targets: int) -> float:
    """Information transfer per selection (Wolpaw et al. 2000), forced choice."""
    if accuracy >= 1.0:
        return math.log2(targets)
    if accuracy <= 1.0 / targets:
        return 0.0
    return math.log2(targets) + accuracy * math.log2(accuracy) + (1 - accuracy) * math.log2((1 - accuracy) / (targets - 1))
