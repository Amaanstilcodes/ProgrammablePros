"""Calibration Engine Module — Genesis 2.0 Track EVA 02.

Implements probability calibration using Platt scaling (logistic transformation)
to calibrate raw probabilities output by Laya. Includes Brier score calculation
for empirical evaluation of probability calibration quality.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


class CalibrationEngine:
    """Post-hoc probability calibration engine using Platt scaling.

    Transforms raw model output P_raw into well-calibrated posterior probability:
        logit(p) = ln(p / (1 - p))
        P_calibrated = 1 / (1 + exp(A * logit(p) + B))
    where A and B are learned parameters (A < 0, B ~ 0 for well-calibrated base models).
    """

    def __init__(self, a: float = -1.0, b: float = 0.0) -> None:
        self.a: float = a
        self.b: float = b
        self._is_fitted: bool = False

    @staticmethod
    def _clamp(p: float, eps: float = 1e-5) -> float:
        """Clamp probability to avoid division by zero or log(0)."""
        return max(eps, min(1.0 - eps, float(p)))

    def predict(self, raw_prob: float) -> float:
        """Map raw probability to calibrated probability via Platt scaling."""
        p = self._clamp(raw_prob)
        logit = math.log(p / (1.0 - p))
        exponent = self.a * logit + self.b
        exponent = max(-50.0, min(50.0, exponent))
        calibrated = 1.0 / (1.0 + math.exp(exponent))
        return float(self._clamp(calibrated))

    def brier_score(self, probs: List[float], labels: List[int]) -> float:
        """Calculate Brier score."""
        if not probs or len(probs) != len(labels):
            return 0.0
        return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)
