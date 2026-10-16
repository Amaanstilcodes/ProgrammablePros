"""Compliance Layer Module — Genesis 2.0 Track EVA 02.

Enforces non-negotiable competition invariants (PDF §4 & §8):
- I1: Exactly four keys: 'up', 'down', 'left', 'right'.
- I2: All probabilities finite numbers in [0.0, 1.0].
- I3: Sum strictly bounded in [0.98, 1.02]. Residual absorbed into dominant action to equal 1.000000.
- I4: decision.chosen == log_line.action.
- I5: decision.confidence == probabilities[chosen] (exact float equality).
- I6: decision.source in {"laya", "fallback"}.
- I7: Fallback moves <= 10% of total accepted session moves.

CRITICAL ARCHITECTURAL INVARIANT:
- `probabilities` is derived ONLY from Laya's primary action hypotheses.
- `chosen` is selected by the multi-objective EV synthesis (which may differ from argmax(probabilities)).
- EV scores MUST NEVER be softmaxed into the `probabilities` field.
"""

from __future__ import annotations

import math
from typing import Any

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

OPPOSITES: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

FALLBACK_PREFERENCE: dict[str, list[str]] = {
    "up": ["left", "right", "down"],
    "down": ["right", "left", "up"],
    "left": ["up", "down", "right"],
    "right": ["down", "up", "left"],
}


class ComplianceLayer:
    """Stateless validator and normalizer enforcing Genesis 2.0 rules."""

    def normalize_probabilities(
        self,
        raw_laya_probabilities: dict[str, float],
    ) -> dict[str, float]:
        """Normalize Laya's primary action probabilities to sum to 1.000000 in [0.98, 1.02].

        Does NOT take EV scores. Converts raw primary action predictions into a strict
        4-key probability simplex.
        """
        # 1. Extract values for all 4 directions
        extracted: dict[str, float] = {}
        for d in DIRECTIONS:
            val = float(raw_laya_probabilities.get(d, 0.0))
            if math.isnan(val) or math.isinf(val) or val < 0.0:
                val = 0.0
            extracted[d] = val

        total = sum(extracted.values())

        if total <= 1e-12:
            # Degenerate case: equal distribution
            probs = {d: 0.25 for d in DIRECTIONS}
        else:
            probs = {d: extracted[d] / total for d in DIRECTIONS}

        # Force exact sum to 1.000000 by adjusting the dominant action
        current_sum = sum(probs.values())
        residual = current_sum - 1.0
        if abs(residual) > 1e-12:
            dominant = max(probs, key=lambda d: probs[d])
            probs[dominant] = max(0.0, min(1.0, probs[dominant] - residual))

        # Re-verify invariant [0.98, 1.02]
        final_sum = sum(probs.values())
        if not (0.98 <= final_sum <= 1.02):
            raise ValueError(f"Normalized probability sum {final_sum:.6f} outside [0.98, 1.02].")

        return {d: round(probs[d], 6) for d in DIRECTIONS}

    def check_snake_safety_fallback(
        self,
        current_state: dict[str, Any],
        proposed_action: str,
    ) -> str | None:
        """Detect illegal 180-degree reversal in Snake and supply safe fallback."""
        proposed = proposed_action.strip().lower()

        current_dir_raw = (
            current_state.get("direction")
            or current_state.get("current_direction")
            or current_state.get("snake_direction")
            or ""
        )
        current_dir = str(current_dir_raw).strip().lower()

        if not current_dir or current_dir not in OPPOSITES:
            return None

        # Reversal detected: choose alternative safe direction
        if proposed == OPPOSITES[current_dir]:
            for alt in FALLBACK_PREFERENCE.get(proposed, []):
                if alt != OPPOSITES[current_dir]:
                    return alt
            return current_dir

        return None

    def build_decision_block(
        self,
        questions: list[str],
        normalized_probs: dict[str, float],
        chosen_action: str,
        source: str,
    ) -> dict[str, Any]:
        """Construct canonical decision block satisfying Invariants I1-I6."""
        chosen = chosen_action.strip().lower()

        if chosen not in normalized_probs:
            raise ValueError(f"Chosen action '{chosen}' not in normalized probabilities: {list(normalized_probs.keys())}")

        if source not in ("laya", "fallback"):
            raise ValueError(f"Invalid source '{source}'. Must be 'laya' or 'fallback'.")

        confidence = normalized_probs[chosen]

        return {
            "questions": list(questions),
            "probabilities": dict(normalized_probs),
            "chosen": chosen,
            "confidence": confidence,
            "source": source,
        }

    def verify_fallback_budget(
        self,
        fallback_count: int,
        accepted_count: int,
        max_ratio: float = 0.10,
    ) -> tuple[bool, float]:
        """Check whether session fallback count is within allowed threshold (PDF §4 <= 10%)."""
        if accepted_count == 0:
            return True, 0.0
        ratio = fallback_count / accepted_count
        return ratio <= max_ratio, ratio
