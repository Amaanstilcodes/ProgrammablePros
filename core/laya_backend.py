"""Laya Backend Interface — Genesis 2.0 Track EVA 02.

HACKATHON DAY PRODUCTION IMPLEMENTATION — ZERO MOCKS ALLOWED.
Uses the real Laya neural decision engine (convaiinnovations/laya).
"""

from __future__ import annotations

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


class LayaBackend(Protocol):
    """Protocol for Laya probability inference backends."""

    def predict_probability(self, context: str, question: str) -> float:
        """Return calibrated probability in [0.0, 1.0] for the given hypothesis question."""
        ...

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        """Batch evaluate hypothesis questions against context."""
        ...


class MockLayaBackend:
    """Mock backend disabled on Hackathon Day."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "CRITICAL: MockLayaBackend is forbidden on HACKATHON DAY. ZERO MOCKS ALLOWED. "
            "All inference must use the real Laya model (LocalLayaBackend: convaiinnovations/laya)."
        )

    def predict_probability(self, context: str, question: str) -> float:
        raise RuntimeError("ZERO MOCKS ALLOWED on Hackathon Day.")

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        raise RuntimeError("ZERO MOCKS ALLOWED on Hackathon Day.")


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
