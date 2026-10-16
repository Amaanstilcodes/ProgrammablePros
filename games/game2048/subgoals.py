"""2048 Subgoal Arbiter — Genesis 2.0 Track EVA 02 / Phase 2b.

Selects tactical subgoals to frame hypothesis wording for Laya:
- P1: SURVIVE (empty_count <= 2)
- P2: RESTORE_CORNER (max_tile_at_designated_corner == False)
- P3: KEEP_CORNER (default)

Subgoal contains NO actions, NO directions, and NO paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from games.game2048.features import Board2048Features, Game2048FeatureExtractor


@dataclass(frozen=True)
class Subgoal2048:
    """Tactical subgoal representation for 2048 hypothesis framing."""

    name: str
    goal_phrase: str
    priority: int
    rationale: str

    @property
    def target_type(self) -> str:
        return self.name

    @property
    def description(self) -> str:
        return self.goal_phrase


class Board2048SubgoalArbiter:
    """Selects tactical objective via deterministic priority cascade."""

    def select(self, features: Board2048Features, memory: Any = None) -> Subgoal2048:
        # P1 SURVIVE
        if features.empty_count <= 2:
            return Subgoal2048(
                name="SURVIVE",
                goal_phrase="creating empty space and merging tiles to keep the board playable",
                priority=1,
                rationale="Critical board congestion: prioritize open space creation and tile merges.",
            )

        # P2 RESTORE_CORNER
        if not features.max_tile_at_designated_corner:
            return Subgoal2048(
                name="RESTORE_CORNER",
                goal_phrase="returning the largest tile to the bottom-left corner and restoring a descending snake chain",
                priority=2,
                rationale="Largest tile has shifted away from designated anchor; prioritize corner recovery.",
            )

        # P3 KEEP_CORNER
        return Subgoal2048(
            name="KEEP_CORNER",
            goal_phrase="keeping the largest tile fixed in the bottom-left corner while merging the next largest tiles along the bottom row and left column",
            priority=3,
            rationale="Maintain stable monotonic snake chain anchored at (3, 0).",
        )

    def arbitrate(self, state_or_features: Any, memory: Any = None) -> Subgoal2048:
        """Backward-compatibility entry point for orchestrator."""
        if isinstance(state_or_features, Board2048Features):
            return self.select(state_or_features, memory)
        extractor = Game2048FeatureExtractor()
        features = extractor.extract_features(state_or_features, memory)
        return self.select(features, memory)


# Alias for backward compatibility
Game2048SubgoalArbiter = Board2048SubgoalArbiter
Subgoal = Subgoal2048
