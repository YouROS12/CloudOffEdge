"""Routing decision: when to escalate from edge to cloud.

Implements the Paper 2 result: route iff `margin < threshold`. Margin =
top1 softmax prob - top2 softmax prob. Default threshold (0.20) was
selected on the val split to hit ~20% offload at 95% of cloud accuracy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def margin_from_probs(probs: np.ndarray) -> float:
    """probs: 1D array of class probabilities."""
    p = np.sort(probs)[::-1]
    return float(p[0] - p[1]) if len(p) >= 2 else float(p[0])


def entropy_from_probs(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def should_route(margin: float, threshold: float) -> bool:
    """Returns True iff this sample should be escalated to the cloud."""
    return margin < threshold
