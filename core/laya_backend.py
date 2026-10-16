"""Laya Backend Interface — Genesis 2.0 Track EVA 02.

Provides the inference boundary for Laya calibrated probability scoring:
- LayaBackendProtocol
- MockLayaBackend: Deterministic contextual heuristic for offline testing and CI
- HttpLayaBackend: REST client for hosted Laya endpoints
- LocalLayaBackend: Embedded model runner (per official model card / SDK)
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Protocol


class LayaBackend(Protocol):
    """Protocol for Laya probability inference backends."""

    def predict_probability(self, context: str, question: str) -> float:
        """Return calibrated probability in [0.0, 1.0] for the given hypothesis question."""
        ...

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        """Batch evaluate hypothesis questions against context."""
        ...


class MockLayaBackend:
    """Deterministic offline evaluator for testing.

    Evaluates hypothesis semantics against egocentric context:
    - Legality queries ('legal move to a FREE cell')
    - Subgoal progress queries ('reduce the distance to the current subgoal')
    - Danger queries ('expose the agent to immediate danger')
    - Novelty queries ('UNKNOWN or unvisited cell')
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def predict_probability(self, context: str, question: str) -> float:
        q_lower = question.lower()
        ctx_lower = context.lower()

        # Anchor probes (Ground-Truth calibration checks)
        if "explicitly marked as wall" in q_lower:
            return 0.0
        if "0.5 is greater than 0.1" in q_lower:
            return 1.0

        # Legality hypothesis
        if "legal move" in q_lower:
            if "wall" in ctx_lower and ("ahead: wall" in ctx_lower or "action" in q_lower):
                return 0.05
            return 0.95

        # Subgoal progress hypothesis
        if "reduce the distance" in q_lower:
            # Check if moving in direction of objective bearing
            if "bearing=e" in ctx_lower and "'right'" in q_lower:
                return 0.88
            if "bearing=w" in ctx_lower and "'left'" in q_lower:
                return 0.88
            if "bearing=s" in ctx_lower and "'down'" in q_lower:
                return 0.88
            if "bearing=n" in ctx_lower and "'up'" in q_lower:
                return 0.88
            return 0.35

        # Danger hypothesis
        if "danger" in q_lower or "hazard" in q_lower:
            if "guard" in ctx_lower and "dist=1" in ctx_lower:
                return 0.90
            return 0.05

        # Novelty hypothesis
        if "unvisited" in q_lower or "unknown" in q_lower:
            return 0.40

        # Default fallback probability
        return 0.50

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        return [self.predict_probability(context, q) for q in questions]


class HttpLayaBackend:
    """REST adapter for remote hosted Laya endpoints."""

    def __init__(self, endpoint_url: str, api_key: str = "", timeout: float = 10.0) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    def predict_probability(self, context: str, question: str) -> float:
        # Structured HTTP call awaiting official hackathon API format
        raise NotImplementedError(
            "HttpLayaBackend selected, but official Laya REST endpoint specification is pending. "
            "Use MockLayaBackend for offline verification."
        )

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        return [self.predict_probability(context, q) for q in questions]


class LocalLayaBackend:
    """In-process Laya model runner."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path

    def predict_probability(self, context: str, question: str) -> float:
        # Awaiting official pip wheel / model weights per Rule R10
        raise NotImplementedError(
            "LocalLayaBackend selected, but official Laya pip package/model card is pending. "
            "Use MockLayaBackend for offline verification."
        )

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        return [self.predict_probability(context, q) for q in questions]
