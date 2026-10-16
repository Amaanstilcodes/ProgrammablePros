"""Laya Backend Interface — Genesis 2.0 Track EVA 02.

HACKATHON DAY PRODUCTION IMPLEMENTATION — ZERO MOCKS ALLOWED.
Uses the real Laya neural decision engine (convaiinnovations/laya).
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from typing import Any, Protocol


# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Polyfill torch.accelerator for PyTorch 2.5.x compatibility with transformers 5.x
try:
    import torch

    if not hasattr(torch, "accelerator"):
        class _DummyAccelerator:
            @staticmethod
            def current_accelerator():
                return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            @staticmethod
            def is_available():
                return torch.cuda.is_available()
            @staticmethod
            def device_count():
                return torch.cuda.device_count() if torch.cuda.is_available() else 0
        torch.accelerator = _DummyAccelerator  # type: ignore[attr-defined]
except ImportError:
    torch = None  # type: ignore[assignment]



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
    - Danger / collision queries (detects wall proximity or body adjacency)
    - Reversal penalty handling
    - Failure-adaptive warning handling ('WARNING: ...')
    """

    def __init__(self, seed: int = 42) -> None:
        if os.getenv("STRICT_HACKATHON_REAL_ONLY") == "1":
            raise RuntimeError(
                "CRITICAL: MockLayaBackend is forbidden on HACKATHON DAY. ZERO MOCKS ALLOWED. "
                "All inference must use the real Laya model (LocalLayaBackend: convaiinnovations/laya)."
            )
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
                left_of = {"up": "left", "down": "right", "left": "down", "right": "up"}
                right_of = {"up": "right", "down": "left", "left": "up", "right": "down"}
                if action == cur_dir:
                    body_adj = ba
                elif action == left_of.get(cur_dir):
                    body_adj = bl
                elif action == right_of.get(cur_dir):
                    body_adj = br

            # Danger hypothesis
            if "collision" in q_lower or "danger" in q_lower:
                if is_reversal:
                    return 0.99
                if wall_dist <= 0 or body_adj == 1:
                    return 0.98
                if wall_dist == 1:
                    return 0.40
                return 0.05

            # Progress hypothesis
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

            if "risking game over" in q_lower or "danger" in q_lower:
                if empty_count <= 2:
                    return 0.85 if action in ("up", "right") else 0.20
                return 0.08 if action in ("down", "left") else 0.35

            if "progresses toward" in q_lower or "changes the board" in q_lower:
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
            "Use LocalLayaBackend with real weights for local inference."
        )

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        return [self.predict_probability(context, q) for q in questions]


class LocalLayaBackend:
    """In-process Laya model runner using the installed 'laya' package (convaiinnovations/laya).

    Runs the real neural model locally with zero mocks.
    """

    def __init__(self, model_id: str = "convaiinnovations/laya") -> None:
        self.model_id = model_id
        self._agent: Any = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._agent is None:
            print("\n" + "=" * 60)
            print(f"=== 🧠 LOADING REAL LAYA MODEL: {self.model_id} ===")
            print("=" * 60)
            t0 = time.perf_counter()
            import laya
            try:
                self._agent = laya.load(model_id_or_path=self.model_id)
            except Exception as exc:
                raise RuntimeError(
                    f"FATAL: Failed to load real Laya model '{self.model_id}': {exc}. "
                    "ZERO MOCKS ALLOWED on Hackathon Day."
                ) from exc
            load_ms = int((time.perf_counter() - t0) * 1000)
            print(f"=== ✅ REAL LAYA MODEL LOADED SUCCESSFULLY ({load_ms}ms) ===")
            print("=" * 60 + "\n")

    def warmup(self) -> None:
        """Run dummy inference to eliminate first-step latency penalty."""
        self._ensure_loaded()
        dummy_q = {"w": {"type": "noul", "instructions": "warmup"}}
        self._agent.predict(state={"context": "warmup"}, questions=dummy_q)

    def model_revision(self) -> str:
        return f"laya-real:{self.model_id}:v0.4.6"

    def predict_probability(self, context: str, question: str) -> float:
        """Evaluate a single hypothesis question using the real Laya model."""
        res = self.batch_predict(context, [question])
        return res[0]

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        """Batch evaluate hypothesis questions against context using real Laya model in one forward pass."""
        self._ensure_loaded()
        if not questions:
            return []

        q_dict: dict[str, dict[str, Any]] = {}
        for idx, q_text in enumerate(questions):
            q_dict[f"q_{idx}"] = {
                "type": "noul",
                "instructions": q_text,
            }

        print("=== 🧠 REAL LAYA INFERENCE EXECUTING ===")
        t0 = time.perf_counter()
        try:
            out = self._agent.predict(state={"context": context}, questions=q_dict)
        except Exception as exc:
            raise RuntimeError(
                f"FATAL: Real Laya model prediction failed: {exc}. "
                "ZERO MOCKS ALLOWED on Hackathon Day."
            ) from exc

        infer_ms = int((time.perf_counter() - t0) * 1000)
        print(f"=== ✅ REAL LAYA INFERENCE COMPLETE ({infer_ms}ms) ===")

        answers = out.get("answers", {})

        probs: list[float] = []
        for idx in range(len(questions)):
            q_key = f"q_{idx}"
            ans = answers.get(q_key, {})
            # Read 'noul' which is P(true) in [0.0, 1.0]
            if "noul" in ans:
                val = float(ans["noul"])
            elif "answer_confidence" in ans:
                val = float(ans["answer_confidence"])
            elif "confidence" in ans:
                val = float(ans["confidence"])
            else:
                raise ValueError(f"Unexpected Laya output format for question '{q_key}': {ans}")
            probs.append(val)

        return probs


__all__ = [
    "LayaBackend",
    "LocalLayaBackend",
    "HttpLayaBackend",
    "MockLayaBackend",
]
