"""Compliance Layer Module — Genesis 2.0 EVA 02.

Bridges ProgrammablePros internal EV/Laya pipeline to the strict PDF schema:
- Normalises arbitrary EV scores → exactly {"up", "down", "left", "right"} summing to 1.000
- Detects illegal Snake reverse-direction moves and substitutes a safe fallback
- Constructs the exact ``decision`` block required by the log schema
"""

from __future__ import annotations

import math
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Bidirectional opposites used to enforce Snake anti-reversal rule (PDF §8)
_OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# Ordered fallback preferences when a reversal must be replaced
_FALLBACK_PREFERENCE: dict[str, list[str]] = {
    "up":    ["left", "right", "down"],
    "down":  ["right", "left", "up"],
    "left":  ["up", "down", "right"],
    "right": ["down", "up", "left"],
}


# ─────────────────────────────────────────────────────────────────────────────
# ComplianceLayer
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceLayer:
    """Stateless bridge between our EV pipeline and the Genesis 2.0 log schema.

    All methods are pure functions (no mutable state) and safe for reuse
    across game steps.
    """

    # ── Probability normalisation ─────────────────────────────────────────────

    def normalize_probabilities_for_log(
        self,
        ev_scores_or_raw_probs: dict[str, Any],
    ) -> dict[str, float]:
        """Map internal action scores to exactly {up, down, left, right} summing to 1.000.

        Algorithm:
        1. Extract a scalar value per direction from the input dict.
           Accepts nested AdvancedEVCalculator dicts (using 'ev' key) or raw floats.
        2. Apply softmax to convert arbitrary scores to a probability simplex.
        3. Force-sum to exactly 1.000 by adjusting the highest-value direction
           by the residual (sum - 1.000), guaranteeing sum ∈ [0.98, 1.02].

        Args:
            ev_scores_or_raw_probs: Dict mapping direction strings to either
                float scores or nested dicts with an 'ev' key (AdvancedEVCalculator output).

        Returns:
            Dict with exactly four keys: "up", "down", "left", "right", values
            summing to precisely 1.000 with no NaN/Infinity.
        """
        # 1. Extract scalar score per direction ─────────────────────────────
        raw: dict[str, float] = {}
        for direction in DIRECTIONS:
            val = ev_scores_or_raw_probs.get(direction, 0.0)
            if isinstance(val, dict):
                # AdvancedEVCalculator format: {ev: float, p_legal: float, ...}
                val = float(val.get("ev", val.get("p_subgoal", 0.0)))
            raw[direction] = float(val)

        # 2. Softmax for numerical stability ─────────────────────────────────
        max_val = max(raw.values())
        exp_vals = {d: math.exp(raw[d] - max_val) for d in DIRECTIONS}
        total_exp = sum(exp_vals.values())

        if total_exp == 0.0 or math.isnan(total_exp) or math.isinf(total_exp):
            # Degenerate fallback: uniform distribution
            probs = {d: 0.25 for d in DIRECTIONS}
        else:
            probs = {d: exp_vals[d] / total_exp for d in DIRECTIONS}

        # 3. Force exact sum to 1.000 ─────────────────────────────────────────
        current_sum = sum(probs.values())
        residual = current_sum - 1.0
        if abs(residual) > 1e-10:
            # Absorb residual into the highest-probability direction
            dominant = max(probs, key=lambda d: probs[d])
            probs[dominant] = probs[dominant] - residual

        # 4. Clamp all values to [0.0, 1.0] to guard against floating-point drift
        probs = {d: max(0.0, min(1.0, probs[d])) for d in DIRECTIONS}

        # 5. Final guard: verify sum is in [0.98, 1.02]
        final_sum = sum(probs.values())
        assert 0.98 <= final_sum <= 1.02, (
            f"Probability sum {final_sum:.6f} out of mandated [0.98, 1.02] window."
        )

        return {d: round(probs[d], 6) for d in DIRECTIONS}

    # ── Snake safety / anti-reversal check ───────────────────────────────────

    def check_snake_safety_fallback(
        self,
        current_state: dict[str, Any],
        proposed_action: str,
    ) -> str | None:
        """Detect and block an illegal Snake reverse-direction move (PDF §8).

        The game engine will reject (and may penalise) any move that is the
        exact opposite of the snake's current travel direction.  We intercept
        it here and substitute the highest-preference safe direction.

        Args:
            current_state:   Raw state dict from the game API; must contain
                             ``"direction"`` or ``"current_direction"`` key.
            proposed_action: Candidate move string (e.g. "up").

        Returns:
            Safe replacement action string if a reversal was detected,
            ``None`` if the proposed action is already safe.
        """
        proposed = proposed_action.strip().lower()

        # Extract the snake's current travel direction from the API state
        current_dir_raw: str = (
            current_state.get("direction")
            or current_state.get("current_direction")
            or current_state.get("snake_direction")
            or ""
        )
        current_dir = str(current_dir_raw).strip().lower()

        if not current_dir or current_dir not in _OPPOSITE:
            # Cannot determine direction; assume safe
            return None

        if proposed == _OPPOSITE[current_dir]:
            # Reversal detected — choose the first available safe alternative
            for fallback in _FALLBACK_PREFERENCE.get(proposed, []):
                if fallback != _OPPOSITE[current_dir]:
                    return fallback
            # Absolute last resort: continue straight ahead
            return current_dir

        return None

    # ── Decision block construction ───────────────────────────────────────────

    def build_decision_block(
        self,
        questions: list[str],
        normalized_probs: dict[str, float],
        chosen_action: str,
        source: str,
    ) -> dict[str, Any]:
        """Construct the exact ``decision`` block required by the PDF schema.

        Enforces:
        - ``confidence`` == ``probabilities[chosen]`` (exact equality)
        - ``chosen`` == ``action`` field in the parent log line
        - ``source`` is "laya" or "fallback"
        - ``probabilities`` has exactly four keys: up/down/left/right

        Args:
            questions:         List of hypothesis strings sent to Laya for this step.
            normalized_probs:  Output of ``normalize_probabilities_for_log()`` — must
                               have exactly the four direction keys.
            chosen_action:     The action actually sent to the game API.
            source:            "laya" if selected by the EV engine,
                               "fallback" if forced by safety check.

        Returns:
            Dict conforming exactly to the PDF ``decision`` block schema.

        Raises:
            ValueError: If ``chosen_action`` is not a key in ``normalized_probs``.
        """
        chosen = chosen_action.strip().lower()

        if chosen not in normalized_probs:
            raise ValueError(
                f"chosen_action '{chosen}' not found in normalized_probs keys: "
                f"{list(normalized_probs.keys())}"
            )

        if source not in ("laya", "fallback"):
            raise ValueError(
                f"source must be 'laya' or 'fallback', got: '{source}'"
            )

        confidence: float = normalized_probs[chosen]

        return {
            "questions": list(questions),
            "probabilities": dict(normalized_probs),  # shallow copy — must be immutable
            "chosen": chosen,
            "confidence": confidence,
            "source": source,
        }

    # ── Fallback budget tracker ───────────────────────────────────────────────

    def compute_fallback_rate(self, log_lines: list[dict[str, Any]]) -> float:
        """Calculate the fraction of moves sourced from "fallback" (PDF §4 ≤ 10%).

        Args:
            log_lines: List of fully logged move dicts (with ``decision`` key).

        Returns:
            Fraction in [0.0, 1.0]. Returns 0.0 if log_lines is empty.
        """
        if not log_lines:
            return 0.0
        fallback_count = sum(
            1
            for line in log_lines
            if line.get("decision", {}).get("source") == "fallback"
        )
        return fallback_count / len(log_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cl = ComplianceLayer()

    # ── Test 1: normalize arbitrary EV scores ──────────────────────────────
    ev_scores = {"up": 8.5, "down": -3.1, "left": 2.0, "right": 6.4}
    probs = cl.normalize_probabilities_for_log(ev_scores)
    assert set(probs.keys()) == {"up", "down", "left", "right"}, "Missing keys"
    total = sum(probs.values())
    assert 0.98 <= total <= 1.02, f"Sum out of range: {total}"
    print(f"[ComplianceLayer] normalize_probabilities_for_log PASSED — sum={total:.6f}")
    for d, p in probs.items():
        print(f"  {d}: {p:.4f}")

    # ── Test 2: normalize nested AdvancedEVCalculator dicts ────────────────
    ev_nested = {
        "up":    {"ev": 8.5,  "p_legal": 0.9},
        "down":  {"ev": -3.1, "p_legal": 0.8},
        "left":  {"ev": 2.0,  "p_legal": 0.7},
        "right": {"ev": 6.4,  "p_legal": 0.95},
    }
    probs2 = cl.normalize_probabilities_for_log(ev_nested)
    total2 = sum(probs2.values())
    assert 0.98 <= total2 <= 1.02, f"Nested sum out of range: {total2}"
    print(f"[ComplianceLayer] normalize nested EV dicts PASSED — sum={total2:.6f}")

    # ── Test 3: snake safety fallback ─────────────────────────────────────
    state_moving_up = {"direction": "up"}
    result = cl.check_snake_safety_fallback(state_moving_up, "down")  # reversal
    assert result is not None, "Should have detected reversal"
    assert result != "down", "Fallback must not be the reversal direction"
    print(f"[ComplianceLayer] snake_safety_fallback PASSED — fallback='{result}'")

    no_reversal = cl.check_snake_safety_fallback(state_moving_up, "right")
    assert no_reversal is None, "Non-reversal should return None"
    print(f"[ComplianceLayer] snake_safety_no_reversal PASSED")

    # ── Test 4: build_decision_block ──────────────────────────────────────
    block = cl.build_decision_block(
        questions=["Hypothesis Q1", "Hypothesis Q2"],
        normalized_probs=probs,
        chosen_action="up",
        source="laya",
    )
    assert block["chosen"] == "up"
    assert block["confidence"] == probs["up"]
    assert block["source"] == "laya"
    assert len(block["probabilities"]) == 4
    print(f"[ComplianceLayer] build_decision_block PASSED — confidence={block['confidence']:.4f}")

    print("\n[ComplianceLayer] ALL SELF-TESTS PASSED")
