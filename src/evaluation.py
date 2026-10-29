"""Evaluation helpers and physical-domain constraint utilities."""

from __future__ import annotations

import numpy as np

from .config import PHYSICAL_BOUNDS, TARGET_LABELS


def mean_r2(scores: dict[str, float]) -> float:
    values = [scores[target] for target in TARGET_LABELS if target in scores]
    return float(np.mean(values)) if values else float("nan")


def clip_physical(predictions: dict[str, float]) -> dict[str, float]:
    result = dict(predictions)
    for target, (low, high) in PHYSICAL_BOUNDS.items():
        if target in result:
            result[target] = min(high, max(low, float(result[target])))
    return result