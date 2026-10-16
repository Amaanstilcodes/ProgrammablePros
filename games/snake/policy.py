"""Snake Decision Policy — Genesis 2.0 Track EVA 02.

Selects the action from Laya's evaluated probabilities and EV scores.
Enforces Rule R4:
- Safety fallback allowed ONLY to avoid a fatal 180-degree reverse move.
- Logged with source="fallback".
"""

from __future__ import annotations

from typing import Any

from core.compliance_layer import ComplianceLayer


class SnakeDecisionPolicy:
    """Combines EV score argmax with anti-reversal safety guard."""

    def __init__(self) -> None:
        self.compliance = ComplianceLayer()

    def select_action(
        self,
        action_probabilities: dict[str, float],
        ev_scores: dict[str, float],
        legal_actions: list[str],
        state_before: dict[str, Any],
    ) -> tuple[str, str]:
        """Select action via EV argmax, intercepting fatal reversals with safe fallback.

        Returns:
            Tuple of (chosen_action, source) where source is 'laya' or 'fallback'.
        """
        # 1. Proposed action: Argmax of Expected Value scores
        if ev_scores:
            proposed_action = max(ev_scores, key=lambda a: ev_scores[a])
        else:
            proposed_action = max(action_probabilities, key=lambda a: action_probabilities[a])

        # 2. Safety fallback check (anti-reversal rule PDF §8)
        fallback = self.compliance.check_snake_safety_fallback(state_before, proposed_action)
        if fallback is not None:
            return fallback, "fallback"

        return proposed_action, "laya"
