"""Snake Subgoal Arbiter — Genesis 2.0 Track EVA 02 / Phase 2a.

Selects the tactical subgoal from deterministic board features and memory.
Rule R3: Subgoal arbitration selects the WORDING of the hypothesis, NEVER an action.
Subgoal contains NO actions, NO directions, and NO paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from games.snake.features import FeatureExtractor, SnakeFeatures


@dataclass(frozen=True)
class Subgoal:
    """Tactical subgoal representation for hypothesis framing."""

    name: str
    goal_phrase: str
    priority: int
    rationale: str

    @property
    def target_type(self) -> str:
        """Backward-compatibility alias."""
        return self.name

    @property
    def description(self) -> str:
        """Backward-compatibility alias."""
        return self.goal_phrase


class SnakeSubgoalArbiter:
    """Determines high-level objective via deterministic priority cascade:

    P1: SURVIVE    (cramped or body_ahead or wall_ahead)
    P2: REGROUP    (food_stall_turns >= 8 and occupied_fraction > 0.35)
    P3: CHASE_FOOD (otherwise)
    """

    def select(self, features: SnakeFeatures, memory: Any = None) -> Subgoal:
        """Deterministic priority cascade over features and memory."""
        # P1 SURVIVE
        if features.cramped or features.body_ahead or features.wall_ahead:
            return Subgoal(
                name="SURVIVE",
                goal_phrase="preserving an escape route and avoiding collision",
                priority=1,
                rationale="Self-preservation precedes food when in immediate danger or cramped quarters.",
            )

        # P2 REGROUP
        stall_turns = features.food_stall_turns
        if memory is not None and hasattr(memory, "food_stall_turns"):
            stall_turns = memory.food_stall_turns

        if stall_turns >= 8 and features.occupied_fraction > 0.35:
            return Subgoal(
                name="REGROUP",
                goal_phrase="reaching a more open region of the board near the tail",
                priority=2,
                rationale="Break a prolonged food stall before the body coils.",
            )

        # P3 CHASE_FOOD
        if features.food is None:
            phrase = "exploring open space and increasing the chance of encountering food while avoiding collision"
        else:
            phrase = "reaching the food while keeping an escape route open"

        return Subgoal(
            name="CHASE_FOOD",
            goal_phrase=phrase,
            priority=3,
            rationale="Seek nutritional targets to progress while maintaining clearance.",
        )

    def arbitrate(self, state_or_features: Any, memory: Any = None) -> Subgoal:
        """Adapter accepting either SnakeFeatures or raw state."""
        if isinstance(state_or_features, SnakeFeatures):
            return self.select(state_or_features, memory)
        extractor = FeatureExtractor()
        features = extractor.extract_features(state_or_features, memory)
        return self.select(features, memory)


# Alias for backward compatibility
SnakeSubgoal = Subgoal
