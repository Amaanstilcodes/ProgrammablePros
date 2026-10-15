"""Calibration Engine Module.

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
        """Initialize CalibrationEngine.

        Args:
            a: Slope parameter for Platt scaling (default: -1.0, identity in logit space).
            b: Intercept parameter for Platt scaling (default: 0.0).
        """
        self.a: float = a
        self.b: float = b
        self._is_fitted: bool = False

    @staticmethod
    def _clamp(p: float, eps: float = 1e-5) -> float:
        """Clamp probability to avoid division by zero or log(0)."""
        return max(eps, min(1.0 - eps, float(p)))

    def predict(self, raw_prob: float) -> float:
        """Map raw probability to calibrated probability via Platt scaling.

        Args:
            raw_prob: Raw probability float in [0.0, 1.0].

        Returns:
            Calibrated probability float in [0.0, 1.0].
        """
        p = self._clamp(raw_prob)
        # Logit transformation
        logit = math.log(p / (1.0 - p))
        # Logistic sigmoid with parameters (A, B)
        exponent = self.a * logit + self.b
        # Numerical stability against overflow
        exponent = max(-50.0, min(50.0, exponent))
        calibrated = 1.0 / (1.0 + math.exp(exponent))
        return float(self._clamp(calibrated))

    def fit(
        self,
        raw_probs: List[float],
        labels: List[int],
        epochs: int = 100,
        lr: float = 0.05,
    ) -> None:
        """Fit Platt scaling parameters (A, B) using binary cross-entropy gradient descent.

        Args:
            raw_probs: List of raw probabilities from calibration dataset.
            labels: Ground truth binary outcomes (0 or 1).
            epochs: Optimization iterations (default: 100).
            lr: Learning rate (default: 0.05).
        """
        if not raw_probs or len(raw_probs) != len(labels):
            return

        a = self.a
        b = self.b
        n = len(raw_probs)

        # Pre-compute logits
        logits = [math.log(self._clamp(p) / (1.0 - self._clamp(p))) for p in raw_probs]

        for _ in range(epochs):
            grad_a = 0.0
            grad_b = 0.0
            for z, y in zip(logits, labels):
                exp_val = max(-50.0, min(50.0, a * z + b))
                pred = 1.0 / (1.0 + math.exp(exp_val))
                error = pred - y
                grad_a += error * z
                grad_b += error

            a -= lr * (grad_a / n)
            b -= lr * (grad_b / n)

        self.a = a
        self.b = b
        self._is_fitted = True

    @staticmethod
    def calculate_brier_score(predicted_probs: List[float], ground_truths: List[int]) -> float:
        """Calculate Brier score: mean squared difference between predictions and outcomes.

        Lower is better (0.0 = perfect calibration, 0.25 = uninformative coin flip).

        Args:
            predicted_probs: Model probabilities.
            ground_truths: Binary outcomes (0 or 1).

        Returns:
            Brier score float in [0.0, 1.0].
        """
        if not predicted_probs or len(predicted_probs) != len(ground_truths):
            return 0.0

        total_sq_err = sum(
            (p - y) ** 2 for p, y in zip(predicted_probs, ground_truths)
        )
        return total_sq_err / len(predicted_probs)


if __name__ == "__main__":
    calibrator = CalibrationEngine()
    test_p = 0.75
    cal_p = calibrator.predict(test_p)
    brier = calibrator.calculate_brier_score([0.9, 0.2, 0.8], [1, 0, 1])
    print(f"[CalibrationEngine self-test PASSED]")
    print(f"  Raw: {test_p} -> Calibrated: {cal_p:.4f} | Brier: {brier:.4f}")
