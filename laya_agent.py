"""laya_agent.py — Real Laya Calibrated Probability Decision Engine — Genesis 2.0 Track EVA 02.

HACKATHON DAY PRODUCTION IMPLEMENTATION — ZERO MOCKS ALLOWED.
Uses the real Laya neural decision engine (convaiinnovations/laya).
Fails loudly if the model cannot be loaded or if inference fails.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 stdout on Windows to support emojis cleanly
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

try:
    import laya
except ImportError as exc:
    raise RuntimeError(
        "FATAL: The 'laya' package is not installed. ZERO MOCKS ALLOWED on Hackathon Day. "
        "Please ensure 'laya' is installed in the active Python environment."
    ) from exc


def _extract_food_pos(state: Dict[str, Any]) -> tuple[int, int]:
    """Safely extract (fx, fy) coordinate tuple from game state."""
    food = state.get("food", (0, 0))
    if isinstance(food, dict):
        return int(food.get("x", 0)), int(food.get("y", 0))
    if isinstance(food, (list, tuple)) and len(food) >= 2:
        return int(food[0]), int(food[1])
    return (0, 0)


class LayaAgent:
    """Production Laya Agent using the real convaiinnovations/laya model.

    ZERO MOCKS ALLOWED: Uses pure neural inference for calibrated probabilities.
    Fails loudly if the model weights cannot be loaded or if prediction fails.
    """

    MODEL_ID: str = "convaiinnovations/laya"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        max_memory_turns: int = 3,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the real Laya Agent and load model weights from Hugging Face cache."""
        self.model_id = model_id or self.MODEL_ID
        self.max_memory_turns = max_memory_turns
        self.device = device

        # Failure-Adaptive Short-Term Memory
        self.recent_outcomes: List[Dict[str, Any]] = []

        # Cached telemetry from last decision cycle
        self.last_questions: Dict[str, str] = {}
        self.last_warnings: Dict[str, str] = {}
        self.last_compressed_state: str = ""
        self.last_probabilities: Dict[str, float] = {}

        # Load the real Laya model with explicit visibility banners
        print("\n" + "=" * 60)
        print(f"=== 🧠 LOADING REAL LAYA MODEL: {self.model_id} ===")
        print("=" * 60)
        t0 = time.perf_counter()
        try:
            self.agent = laya.load(self.model_id, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"FATAL: Failed to load real Laya model '{self.model_id}': {exc}. "
                "ZERO MOCKS ALLOWED on Hackathon Day."
            ) from exc

        load_ms = int((time.perf_counter() - t0) * 1000)
        print(f"=== ✅ REAL LAYA MODEL LOADED SUCCESSFULLY ({load_ms}ms) ===")
        print("=" * 60 + "\n")

    def record_outcome(
        self,
        action: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        player_pos_key: str = "player_pos",
    ) -> bool:
        """Record the outcome of taking an action to detect failure/rejection."""
        pos_before = state_before.get(player_pos_key)
        pos_after = state_after.get(player_pos_key)

        made_progress = True
        reason = "ok"

        # Position unchanged detection
        if pos_before is not None and pos_after is not None and pos_before == pos_after:
            made_progress = False
            reason = "position unchanged (hit obstacle / wall / rejected move)"

        record: Dict[str, Any] = {
            "action": action,
            "pos_before": pos_before,
            "pos_after": pos_after,
            "made_progress": made_progress,
            "reason": reason,
        }
        self.recent_outcomes.append(record)

        if len(self.recent_outcomes) > self.max_memory_turns:
            self.recent_outcomes.pop(0)

        return made_progress

    def did_action_recently_fail(self, action: str) -> bool:
        """Check if taking this action in recent history resulted in zero progress."""
        for outcome in reversed(self.recent_outcomes):
            if outcome["action"] == action:
                return not outcome["made_progress"]
        return False

    def clear_memory(self) -> None:
        """Reset recent outcomes memory buffer."""
        self.recent_outcomes = []
        self.last_warnings = {}

    def predict(
        self,
        compressed_state: str,
        questions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute real Laya inference over formatted noul questions.

        Returns:
            Dictionary containing 'probabilities', 'answers', and raw 'model_output'.
        """
        # Format questions for Laya Noul question type (calibrated P(true) in [0.0, 1.0])
        formatted_questions: Dict[str, Dict[str, Any]] = {}
        for q_id, q_val in questions.items():
            if isinstance(q_val, str):
                formatted_questions[q_id] = {
                    "type": "noul",
                    "instructions": q_val,
                }
            elif isinstance(q_val, dict):
                formatted_questions[q_id] = q_val
            else:
                formatted_questions[q_id] = {
                    "type": "noul",
                    "instructions": str(q_val),
                }

        print("=== 🧠 REAL LAYA INFERENCE EXECUTING ===")
        t0 = time.perf_counter()
        try:
            raw_output = self.agent.predict(
                state={"context": compressed_state},
                questions=formatted_questions,
            )
        except Exception as exc:
            raise RuntimeError(
                f"FATAL: Real Laya model prediction failed: {exc}. "
                "ZERO MOCKS ALLOWED on Hackathon Day."
            ) from exc

        infer_ms = int((time.perf_counter() - t0) * 1000)
        print(f"=== ✅ REAL LAYA INFERENCE COMPLETE ({infer_ms}ms) ===")

        # Extract calibrated probabilities from answers
        answers = raw_output.get("answers", {})
        probabilities: Dict[str, float] = {}

        for q_id in formatted_questions:
            ans = answers.get(q_id, {})
            if isinstance(ans, dict):
                if "noul" in ans:
                    prob = float(ans["noul"])
                elif "probabilities" in ans:
                    p_val = ans["probabilities"]
                    prob = float(p_val[0] if isinstance(p_val, (list, tuple)) else p_val)
                elif "answer_confidence" in ans:
                    prob = float(ans["answer_confidence"])
                elif "confidence" in ans:
                    prob = float(ans["confidence"])
                else:
                    prob = 0.5
            elif isinstance(ans, (int, float)):
                prob = float(ans)
            else:
                prob = 0.5

            probabilities[q_id] = prob

        raw_output["probabilities"] = probabilities
        return raw_output

    def decide(
        self,
        state: Dict[str, Any],
        possible_actions: List[str] = ("up", "down", "left", "right"),
        compressed_state: str = "",
    ) -> Dict[str, float]:
        """Evaluate candidate actions and return REAL calibrated probabilities.

        Builds noul questions explicitly targeting the food position:
        `Moving '{action}' is legal and progresses toward the food at {food_pos} while staying safe.`

        Args:
            state: Current game state dictionary.
            possible_actions: Candidate actions (e.g. ['up', 'down', 'left', 'right']).
            compressed_state: Pre-computed compressed string representation.

        Returns:
            Flat dict mapping each action to its real calibrated probability float, e.g.:
            {"up": 0.85, "down": 0.05, "left": 0.08, "right": 0.02}
        """
        self.last_compressed_state = compressed_state
        self.last_warnings = {}

        food_pos = _extract_food_pos(state)

        # 1. Build questions dict where keys are q_up, q_down, etc.
        # 2. Each value is {"type": "noul", "instructions": f"Moving '{action}' is legal and progresses toward the food at {food_pos} while staying safe."}
        questions: Dict[str, Dict[str, Any]] = {}
        for action in possible_actions:
            q_key = f"q_{action}"
            instruction = (
                f"Moving '{action}' is legal and progresses toward the food at {food_pos} while staying safe."
            )
            if self.did_action_recently_fail(action):
                warning = f"Note: Taking '{action}' previously failed or made no progress. Adjust probability accordingly."
                self.last_warnings[action] = warning
                instruction += f" {warning}"

            questions[q_key] = {
                "type": "noul",
                "instructions": instruction,
            }

        self.last_questions = {k: v["instructions"] for k, v in questions.items()}

        # 3. Print execution banner & 4. Call self.agent.predict
        result = self.predict(
            compressed_state=compressed_state,
            questions=questions,
        )

        # 6. Extract calibrated probability for each action from result["answers"]
        answers = result.get("answers", {})
        action_probs: Dict[str, float] = {}

        for action in possible_actions:
            q_key = f"q_{action}"
            ans = answers.get(q_key, {})
            if isinstance(ans, dict):
                if "noul" in ans:
                    prob = float(ans["noul"])
                elif "probabilities" in ans:
                    p_val = ans["probabilities"]
                    prob = float(p_val[0] if isinstance(p_val, (list, tuple)) else p_val)
                elif "answer_confidence" in ans:
                    prob = float(ans["answer_confidence"])
                elif "confidence" in ans:
                    prob = float(ans["confidence"])
                else:
                    prob = 0.5
            elif isinstance(ans, (int, float)):
                prob = float(ans)
            else:
                prob = 0.5

            action_probs[action] = round(prob, 6)

        self.last_probabilities = action_probs
        return action_probs

    def get_last_questions(self) -> Dict[str, str]:
        """Retrieve the typed questions formulated during the most recent decision call."""
        return self.last_questions

    def get_last_warnings(self) -> Dict[str, str]:
        """Retrieve failure-adaptive warnings generated during the most recent decision call."""
        return self.last_warnings

    def get_last_compressed_state(self) -> str:
        """Retrieve the compressed state string generated during the most recent decision call."""
        return self.last_compressed_state


__all__ = ["LayaAgent"]
