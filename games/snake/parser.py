"""Snake State Parser — Genesis 2.0 Track EVA 02.

Parses raw server state from TopGameAI Snake arena.
Strictly non-mutating: converts raw state into an immutable GameStateView.
Supports both food-exposed and food-not-exposed modes.
"""

from __future__ import annotations

import copy
from typing import Any

from core.protocols import GameStateView, SchemaError, StateParser


class SnakeStateParser:
    """Parses and validates raw Snake API JSON dictionaries into GameStateView."""

    def validate(self, raw_state: dict[str, Any]) -> None:
        """Validate raw state dictionary against contract schema.

        Raises:
            SchemaError: if required structural fields are missing or malformed.
        """
        if not isinstance(raw_state, dict):
            raise SchemaError(f"Expected dict for raw_state, got {type(raw_state).__name__}")

        # Check for presence of snake body or board representation
        has_snake = any(k in raw_state for k in ("snake", "body", "snake_body", "player_pos"))
        if not has_snake:
            raise SchemaError("Missing required snake body coordinates in state payload")

    def parse(self, raw_state: dict[str, Any], session_id: str = "") -> GameStateView:
        """Convert raw server response into an immutable GameStateView without mutating raw_state."""
        self.validate(raw_state)
        # Deepcopy to guarantee zero mutation
        raw_copy = copy.deepcopy(raw_state)

        # Snake body coordinates: list of [x, y] with head at index 0
        raw_snake = (
            raw_copy.get("snake")
            or raw_copy.get("body")
            or raw_copy.get("snake_body")
            or []
        )
        snake: list[tuple[int, int]] = []
        if isinstance(raw_snake, (list, tuple)):
            for pt in raw_snake:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    snake.append((int(pt[0]), int(pt[1])))
                elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                    snake.append((int(pt["x"]), int(pt["y"])))
        elif "player_pos" in raw_copy:
            pos = raw_copy["player_pos"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                snake.append((int(pos[0]), int(pos[1])))

        head = snake[0] if snake else (0, 0)

        # Food coordinates: may be a point, list of points, or None if unexposed
        raw_food = (
            raw_copy.get("food")
            or raw_copy.get("food_pos")
            or raw_copy.get("fruit")
        )
        food: tuple[int, int] | None = None
        if isinstance(raw_food, (list, tuple)) and len(raw_food) > 0:
            if isinstance(raw_food[0], (list, tuple)):
                food = (int(raw_food[0][0]), int(raw_food[0][1]))
            elif len(raw_food) >= 2:
                food = (int(raw_food[0]), int(raw_food[1]))
        elif isinstance(raw_food, dict) and "x" in raw_food and "y" in raw_food:
            food = (int(raw_food["x"]), int(raw_food["y"]))

        # Travel direction
        direction = str(
            raw_copy.get("direction")
            or raw_copy.get("current_direction")
            or raw_copy.get("snake_direction")
            or "right"
        ).strip().lower()

        # Score & game_over
        score = int(raw_copy.get("score", max(0, len(snake) - 3) * 60))
        game_over = bool(
            raw_copy.get("game_over")
            or raw_copy.get("done")
            or raw_copy.get("finished")
            or False
        )
        step = int(raw_copy.get("step", 0)) if "step" in raw_copy else None
        width = int(raw_copy.get("width", 20)) if "width" in raw_copy else None
        height = int(raw_copy.get("height", 20)) if "height" in raw_copy else None

        sid = session_id or str(raw_copy.get("session_id", ""))

        extra = {k: v for k, v in raw_copy.items() if k not in (
            "snake", "body", "snake_body", "food", "food_pos", "fruit",
            "direction", "current_direction", "score", "game_over", "done", "finished", "step", "width", "height"
        )}

        return GameStateView(
            game="snake",
            session_id=sid,
            raw=raw_copy,
            head=head,
            body=tuple(snake),
            direction=direction,
            food=food,
            width=width,
            height=height,
            score=score,
            step=step,
            game_over=game_over,
            extra=extra,
        )

    def is_game_over(self, raw_state: dict[str, Any]) -> bool:
        """Check whether the game has ended."""
        return bool(
            raw_state.get("game_over")
            or raw_state.get("done")
            or raw_state.get("finished")
            or False
        )

    def is_rejected(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        response: dict[str, Any],
    ) -> bool:
        """Determine whether move was rejected.

        Detection order:
        1. Explicit server flag in response
        2. Unchanged snake head coordinates
        """
        if response.get("result") == "rejected" or response.get("rejected") is True:
            return True
        if response.get("result") == "applied":
            return False

        view_before = self.parse(before)
        view_after = self.parse(after)
        if view_before.head is not None and view_after.head is not None:
            return view_before.head == view_after.head
        return False

    def score_of(self, raw_state: dict[str, Any]) -> int:
        """Extract score from state."""
        return int(raw_state.get("score", 0))
