"""Snake Subgoal Arbiter — Genesis 2.0 Track EVA 02.

Arbitrates tactical subgoals (e.g. FOOD, EVADE, SPACE) to steer hypothesis wording.
Rule R3: Subgoal arbitration chooses the WORDING of hypotheses, never an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SnakeSubgoal:
    target_type: str  # FOOD | SURVIVAL | SPACE
    description: str
    target_pos: tuple[int, int] | None


class SnakeSubgoalArbiter:
    """Selects high-level tactical objective framing for hypothesis formulation."""

    def arbitrate(self, state: dict[str, Any], memory: Any) -> SnakeSubgoal:
        head = state.get("head", (0, 0))
        food = state.get("food", (5, 5))
        snake_body = state.get("snake", [])

        # Default objective: navigate to food
        return SnakeSubgoal(
            target_type="FOOD",
            description=f"Advance toward food at {food}",
            target_pos=food,
        )
