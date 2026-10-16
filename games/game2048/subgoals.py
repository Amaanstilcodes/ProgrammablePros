"""2048 Subgoal Arbiter — Genesis 2.0 Track EVA 02.

Arbitrates tactical subgoals (CORNER_ANCHOR, MERGE, COMPACT) to steer hypothesis wording.
Rule R3: Subgoal arbitration chooses the WORDING of hypotheses, never an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Game2048Subgoal:
    target_type: str  # CORNER_ANCHOR | MERGE | COMPACT
    description: str


class Game2048SubgoalArbiter:
    """Selects tactical phrasing focus for 2048 queries."""

    def arbitrate(self, state: dict[str, Any], memory: Any) -> Game2048Subgoal:
        max_tile = state.get("max_tile", 2)
        merges = state.get("available_merges", 0)

        if merges > 0:
            return Game2048Subgoal(
                target_type="MERGE",
                description=f"Combine matching tiles to clear space and build toward {max_tile * 2}",
            )

        return Game2048Subgoal(
            target_type="CORNER_ANCHOR",
            description=f"Maintain largest tile ({max_tile}) in the anchor corner",
        )
