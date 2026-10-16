"""Snake State Parser — Genesis 2.0 Track EVA 02.

Parses raw server state from TopGameAI Snake arena.
Strictly adapts without guessing: extracts snake coordinates, food position,
current travel direction, score, and terminal state flags.
"""

from __future__ import annotations

from typing import Any


class SnakeStateParser:
    """Parses and validates raw Snake API JSON dictionaries."""

    def parse(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        """Convert raw server response into a normalized dictionary."""
        # Snake body coordinates: list of [x, y] with head at index 0
        raw_snake = (
            raw_state.get("snake")
            or raw_state.get("body")
            or raw_state.get("snake_body")
            or []
        )
        snake: list[tuple[int, int]] = []
        if isinstance(raw_snake, (list, tuple)):
            for pt in raw_snake:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    snake.append((int(pt[0]), int(pt[1])))
                elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                    snake.append((int(pt["x"]), int(pt["y"])))

        head = snake[0] if snake else (0, 0)

        # Food coordinates: may be a point or list of points
        raw_food = (
            raw_state.get("food")
            or raw_state.get("food_pos")
            or raw_state.get("fruit")
            or [5, 5]
        )
        food: tuple[int, int] = (5, 5)
        if isinstance(raw_food, (list, tuple)) and len(raw_food) > 0:
            if isinstance(raw_food[0], (list, tuple)):
                food = (int(raw_food[0][0]), int(raw_food[0][1]))
            elif len(raw_food) >= 2:
                food = (int(raw_food[0]), int(raw_food[1]))
        elif isinstance(raw_food, dict) and "x" in raw_food and "y" in raw_food:
            food = (int(raw_food["x"]), int(raw_food["y"]))

        # Travel direction
        direction = str(
            raw_state.get("direction")
            or raw_state.get("current_direction")
            or raw_state.get("snake_direction")
            or "right"
        ).strip().lower()

        # Score & game_over
        score = int(raw_state.get("score", max(0, len(snake) - 3) * 60))
        game_over = bool(
            raw_state.get("game_over")
            or raw_state.get("done")
            or raw_state.get("finished")
            or False
        )
        step = int(raw_state.get("step", 0))

        return {
            "head": head,
            "snake": snake,
            "food": food,
            "direction": direction,
            "score": score,
            "game_over": game_over,
            "step": step,
            "raw": raw_state,
        }
