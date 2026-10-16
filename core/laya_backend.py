"""Laya Backend Interface — Genesis 2.0 Track EVA 02.

Provides the inference boundary for Laya calibrated probability scoring:
- LayaBackend Protocol
- MockLayaBackend: Context-aware probability evaluator supporting Snake and 2048 hypothesis questions
- HttpLayaBackend: REST client for hosted Laya endpoints
- LocalLayaBackend: In-process Laya model runner
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

    Parses hypothesis semantics against context string:
    - Legality queries ('legal move')
    - Progress toward subgoal (evaluates relative food bearing or merge opportunities)
    - Danger / collision queries (detects wall proximity or body adjacency)
    - Reversal penalty handling
    - Failure-adaptive warning handling ('WARNING: ...')
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

        # Extract candidate action from question
        action = None
        for a in ("up", "down", "left", "right"):
            if f"'{a}'" in q_lower or f" {a} " in q_lower or f"action '{a}'" in q_lower:
                action = a
                break

        # Check for failure warning
        if f"warning: in the previous turn, taking action '{action}'" in q_lower or (action and f"w!{action}" in ctx_lower):
            # Suppress probability heavily
            if "collision" in q_lower or "danger" in q_lower:
                return 0.95
            return 0.05

        # ── SNAKE HYPOTHESIS EVALUATION ──────────────────────────────────
        if "h:(" in ctx_lower and "f:(" in ctx_lower:
            h_match = re.search(r"h:\((\d+),(\d+)\)", ctx_lower)
            f_match = re.search(r"f:\((\d+),(\d+)\)", ctx_lower)
            w_match = re.search(r"w:(\d+)/(\d+)/(\d+)/(\d+)", ctx_lower)
            d_match = re.search(r"d:(\w+)", ctx_lower)
            b_match = re.search(r"b:(\d)(\d)(\d)", ctx_lower)

            hx, hy = (int(h_match.group(1)), int(h_match.group(2))) if h_match else (0, 0)
            fx, fy = (int(f_match.group(1)), int(f_match.group(2))) if f_match else (0, 0)
            cur_dir = d_match.group(1) if d_match else "right"
            d_up, d_down, d_left, d_right = (
                (int(w_match.group(1)), int(w_match.group(2)), int(w_match.group(3)), int(w_match.group(4)))
                if w_match else (10, 10, 10, 10)
            )

            opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
            is_reversal = (action == opposites.get(cur_dir))
            wall_dist = {"up": d_up, "down": d_down, "left": d_left, "right": d_right}.get(action or "", 10)

            # Check body collision ahead from context bits
            body_adj = 0
            if b_match and action:
                ba, bl, br = int(b_match.group(1)), int(b_match.group(2)), int(b_match.group(3))
                # Map relative to action
                left_of = {"up": "left", "down": "right", "left": "down", "right": "up"}
                right_of = {"up": "right", "down": "left", "left": "up", "right": "down"}
                if action == cur_dir:
                    body_adj = ba
                elif action == left_of.get(cur_dir):
                    body_adj = bl
                elif action == right_of.get(cur_dir):
                    body_adj = br

            # Danger hypothesis (Q2)
            if "collision" in q_lower or "danger" in q_lower:
                if is_reversal:
                    return 0.99
                if wall_dist <= 0 or body_adj == 1:
                    return 0.98
                if wall_dist == 1:
                    return 0.40
                return 0.05

            # Progress hypothesis (Q1)
            if "progresses" in q_lower or "reaching the food" in q_lower:
                if is_reversal or wall_dist <= 0 or body_adj == 1:
                    return 0.01

                cur_dist = abs(hx - fx) + abs(hy - fy)
                delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
                dx, dy = delta.get(action or "", (0, 0))
                new_dist = abs(hx + dx - fx) + abs(hy + dy - fy)

                if new_dist < cur_dist:
                    return 0.92
                if new_dist > cur_dist:
                    return 0.08
                return 0.45

        # ── 2048 HYPOTHESIS EVALUATION ───────────────────────────────────
        if "t:" in ctx_lower and "e:" in ctx_lower:
            e_match = re.search(r"e:(\d+)", ctx_lower)
            empty_count = int(e_match.group(1)) if e_match else 4

            # Danger hypothesis (Q2)
            if "risking game over" in q_lower or "danger" in q_lower:
                if empty_count <= 2:
                    return 0.85 if action in ("up", "right") else 0.20
                return 0.08 if action in ("down", "left") else 0.35

            # Progress / Structure hypothesis (Q1)
            if "progresses toward" in q_lower or "changes the board" in q_lower:
                # Prioritize bottom-left corner strategy (Down > Left > Right >> Up)
                if action == "down":
                    return 0.95
                if action == "left":
                    return 0.88
                if action == "right":
                    return 0.40
                return 0.10

        # Generic fallback
        if "legal move" in q_lower or "progresses" in q_lower:
            return 0.70
        if "danger" in q_lower or "collision" in q_lower:
            return 0.10

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
        raise NotImplementedError(
            "HttpLayaBackend selected, but official Laya REST endpoint specification is pending. "
            "Use MockLayaBackend for offline verification."
        )

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        return [self.predict_probability(context, q) for q in questions]


class LocalLayaBackend:
    """In-process Laya model runner using the installed 'laya' package (convaiinnovations/laya)."""

    def __init__(self, model_id: str = "convaiinnovations/laya") -> None:
        self.model_id = model_id
        self._agent: Any = None

    def _ensure_loaded(self) -> None:
        if self._agent is None:
            import laya
            self._agent = laya.load(model_id_or_path=self.model_id)

    def warmup(self) -> None:
        """Run dummy inference to eliminate first-step latency penalty."""
        self._ensure_loaded()
        dummy_q = {"w": {"type": "noul", "instructions": "warmup"}}
        self._agent.predict(state="warmup", questions=dummy_q)

    def model_revision(self) -> str:
        return f"laya-local:{self.model_id}:v0.4.8"

    def predict_probability(self, context: str, question: str) -> float:
        """Evaluate a single hypothesis question using the real Laya model."""
        res = self.batch_predict(context, [question])
        return res[0]

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        """Batch evaluate hypothesis questions against context using real Laya model in one forward pass."""
        self._ensure_loaded()
        q_dict: dict[str, dict[str, Any]] = {}
        for idx, q_text in enumerate(questions):
            q_dict[f"q_{idx}"] = {
                "type": "noul",
                "instructions": q_text,
            }

        out = self._agent.predict(state=context, questions=q_dict)
        answers = out.get("answers", {})

        probs: list[float] = []
        for idx in range(len(questions)):
            q_key = f"q_{idx}"
            ans = answers.get(q_key, {})
            # Read 'noul' which is P(true) in [0.0, 1.0]
            val = ans.get("noul", ans.get("answer_confidence", 0.5))
            probs.append(float(val))

        return probs
