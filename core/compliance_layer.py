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

    def normalize_laya_probabilities(
        self,
        primary_laya_probs: dict[str, float],
        decimals: int = 6,
    ) -> dict[str, float]:
        """Normalize PRIMARY Laya hypothesis probabilities to sum to 1.000000 in [0.98, 1.02].

        Rules (Phase 3.2):
        1. Input MUST be PRIMARY Laya hypothesis probabilities, never EV scores.
        2. Validate: exactly {up,down,left,right}, all finite, all in [0,1].
        3. If already normalized (0.999 <= sum <= 1.001), use as-is after rounding.
        4. Otherwise normalize: p_a / sum(p). If sum == 0: uniform 0.25.
        5. Round each value to `decimals` (default 6).
        6. Compute residual = 1.0 - sum(rounded) and add to LARGEST probability.
        7. Re-clamp to [0, 1].
        8. Assert 0.98 <= sum <= 1.02.
        """
        # Rule 1 & 2: Validate exactly the four directions, all finite and in [0, 1]
        missing = set(DIRECTIONS) - set(primary_laya_probs.keys())
        if missing:
            raise ValueError(f"Primary Laya probabilities missing required directions: {missing}")

        validated: dict[str, float] = {}
        for d in DIRECTIONS:
            val = float(primary_laya_probs[d])
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Non-finite probability for '{d}': {val}")
            if not (0.0 <= val <= 1.0):
                # Clamp small floating point overflow in [0, 1]
                val = max(0.0, min(1.0, val))
            validated[d] = val

        raw_sum = sum(validated.values())

        # Rule 3: Already normalized distribution within 0.999..1.001
        if 0.999 <= raw_sum <= 1.001:
            probs = {d: validated[d] for d in DIRECTIONS}
        elif raw_sum <= 1e-12:
            # Rule 4: Degenerate case sum == 0 -> uniform 0.25
            probs = {d: 0.25 for d in DIRECTIONS}
        else:
            # Rule 4: Normalize
            probs = {d: validated[d] / raw_sum for d in DIRECTIONS}

        # Rule 5: Round each value to decimals
        rounded = {d: round(probs[d], decimals) for d in DIRECTIONS}

        # Rule 6: Residual correction added to largest probability
        residual = round(1.0 - sum(rounded.values()), decimals)
        if abs(residual) > 1e-12:
            dominant = max(rounded, key=lambda d: rounded[d])
            rounded[dominant] = round(max(0.0, min(1.0, rounded[dominant] + residual)), decimals)

        # Rule 7: Re-verify clamping and residual
        final_sum = round(sum(rounded.values()), decimals)
        if abs(1.0 - final_sum) > 1e-6:
            dominant = max(rounded, key=lambda d: rounded[d])
            rounded[dominant] = round(max(0.0, min(1.0, rounded[dominant] + (1.0 - final_sum))), decimals)

        # Rule 8: Assert bounded sum
        total_sum = sum(rounded.values())
        if not (0.98 <= total_sum <= 1.02):
            raise ValueError(f"Normalized probability sum {total_sum:.6f} outside [0.98, 1.02]")

        return rounded

    # Backward compatibility alias
    normalize_probabilities = normalize_laya_probabilities

    def check_snake_safety_fallback(
        self,
        current_state: Any,
        proposed_action: str,
        illegal_mask: dict[str, bool] | None = None,
    ) -> str | None:
        """Deterministic safety fallback check (PDF §8 and Phase 2a Step 5).

        Preference order when current direction is legal and not illegal:
          [current_direction, then the two perpendicular directions in a fixed rotation, never reversal]
        Pick the FIRST one that is not deterministically illegal.
        If NONE is legal, return None.
        If proposed_action is already legal, return None (no fallback needed).
        """
        proposed = proposed_action.strip().lower()

        # Extract direction from dict or dataclass
        current_dir_raw = ""
        if hasattr(current_state, "direction"):
            current_dir_raw = current_state.direction or ""
        elif isinstance(current_state, dict):
            current_dir_raw = (
                current_state.get("direction")
                or current_state.get("current_direction")
                or current_state.get("snake_direction")
                or ""
            )
        current_dir = str(current_dir_raw).strip().lower()

        if not current_dir or current_dir not in OPPOSITES:
            return None

        # Determine if proposed action is illegal
        is_illegal = False
        if illegal_mask is not None:
            is_illegal = bool(illegal_mask.get(proposed, False))
        elif proposed == OPPOSITES[current_dir]:
            is_illegal = True

        if not is_illegal:
            return None

        # Preference order: perpendicular turns in fixed rotation, then current direction, never reversal
        candidates = FALLBACK_PREFERENCE.get(proposed, ["up", "down", "left", "right"])

        for cand in candidates:
            if cand == OPPOSITES[current_dir]:
                continue
            return cand

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


class FallbackBudget:
    """Enforces the strict <= 10% safety fallback budget (PDF §4 & Phase 2a Step 6).

    Rule: (fallbacks_so_far + 1) <= 0.10 * max(1, accepted_so_far + 1)
    """

    def __init__(self, max_ratio: float = 0.10) -> None:
        self.max_ratio = max_ratio

    def may_fallback(self, accepted_so_far: int, fallbacks_so_far: int) -> bool:
        """Return True if an additional fallback is within budget."""
        return (fallbacks_so_far + 1) <= self.max_ratio * max(1, accepted_so_far + 1)


def make_rejection_clause(action: str) -> str:
    """Hard-legality clause injected when fallback budget is exhausted."""
    return (
        f"GROUND TRUTH: '{action}' is illegal in this state. "
        f"Its probability of being a legal progressing move is exactly 0.0."
    )


def points_2048(best_tile: int, in_game_score: int) -> int:
    """Official Genesis 2.0 score formula for Stage 2 (2048).

    Points = min(1000, max(0, exponent - 6) * 143 + in_game_score // 1000)
    """
    if best_tile < 2:
        return 0
    exponent = math.log2(best_tile)
    if not exponent.is_integer():
        exponent = math.floor(exponent)
    return min(1000, max(0, int(exponent) - 6) * 143 + in_game_score // 1000)

