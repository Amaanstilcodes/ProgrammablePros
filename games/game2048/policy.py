"""2048 Decision Policy — Genesis 2.0 Track EVA 02.

Selects action from multi-objective EV synthesis over Laya probabilities.
In 2048, no fatal anti-reversal exists; fallback triggers only if all moves risk deadlock.
"""

from __future__ import annotations

from typing import Any


class Game2048DecisionPolicy:
    """Evaluates multi-objective EV scores for 2048 action selection."""

    def select_action(
        self,
        action_probabilities: dict[str, float],
        ev_scores: dict[str, float],
        legal_actions: list[str],
        state_before: dict[str, Any],
    ) -> tuple[str, str]:
        """Select action via EV argmax."""
        if ev_scores:
            chosen = max(ev_scores, key=lambda a: ev_scores[a])
        else:
            chosen = max(action_probabilities, key=lambda a: action_probabilities[a])

        return chosen, "laya"
