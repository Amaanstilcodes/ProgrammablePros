"""Local Snake Game Engine — Genesis 2.0 Track EVA 02.

Implements the local Snake game according to Track2_PS2_support.pdf §4.1:
- 20x20 grid (configurable)
- Win by eating 15 foods
- Cannot reverse direction in a single move (illegal -> rejected)
- Hitting a wall or own body ends the game (applied, game_over=True)
- Step cap: 300 ACCEPTED moves (rejected moves do not advance accepted step)
- Score = foods_eaten * 60, +100 if won, capped at 1000
- Food position is exposed in state (for local testing)
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any


class SnakeGame:
    """Local Snake simulation matching the TopGameAI hosted API response schema."""

    OPPOSITE_DIRECTIONS: dict[str, str] = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }

    DELTA: dict[str, tuple[int, int]] = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset game to initial state with a fresh session_id."""
        if seed is not None:
            self._rng = random.Random(seed)

        self.session_id = f"local_snake_{uuid.uuid4().hex[:8]}"
        self.direction = "right"
        # Initial snake: head at (5, 10), length 3 horizontal facing right
        self.snake: list[list[int]] = [[5, 10], [4, 10], [3, 10]]
        self.foods_eaten = 0
        self.score = 0
        self.game_over = False
        self.step = 0  # accepted steps
        self.food = self._spawn_food()
        return self.get_state()

    def _spawn_food(self) -> list[int]:
        """Spawn food in a free cell not occupied by snake body."""
        occupied = {(seg[0], seg[1]) for seg in self.snake}
        free_cells = [
            [x, y]
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not free_cells:
            return [-1, -1]
        return self._rng.choice(free_cells)

    def _compute_score(self) -> int:
        """Calculate score based on foods eaten and win condition."""
        s = self.foods_eaten * 60
        if self.foods_eaten >= 15:
            s += 100
        return min(1000, s)

    def get_state(self) -> dict[str, Any]:
        """Return current state conforming to arena API schema."""
        return {
            "session_id": self.session_id,
            "game": "snake",
            "width": self.width,
            "height": self.height,
            "snake": [list(seg) for seg in self.snake],
            "direction": self.direction,
            "food": list(self.food),
            "score": self.score,
            "foods_eaten": self.foods_eaten,
            "game_over": self.game_over,
            "step": self.step,
        }

    def move(self, action: str) -> dict[str, Any]:
        """Execute a move action.

        Returns:
            dict containing state_after, result, game_over, api_ms
        """
        t0 = time.perf_counter()
        act = str(action).strip().lower()

        if self.game_over:
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_after": self.get_state(),
                "result": "rejected",
                "game_over": True,
                "api_ms": api_ms,
            }

        # 1. Action validation
        if act not in self.DELTA:
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_after": self.get_state(),
                "result": "rejected",
                "game_over": self.game_over,
                "api_ms": api_ms,
            }

        # 2. Reversal check (Rule: Cannot reverse direction in a single move)
        if len(self.snake) > 1 and act == self.OPPOSITE_DIRECTIONS.get(self.direction):
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_after": self.get_state(),
                "result": "rejected",
                "game_over": self.game_over,
                "api_ms": api_ms,
            }

        # 3. Step execution (Applied move)
        self.step += 1
        self.direction = act

        dx, dy = self.DELTA[act]
        head_x, head_y = self.snake[0]
        new_head = [head_x + dx, head_y + dy]

        # Wall collision check
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self.game_over = True
            self.score = self._compute_score()
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_after": self.get_state(),
                "result": "applied",
                "game_over": True,
                "api_ms": api_ms,
            }

        # Check self collision (excluding tail if not eating, or entire body if eating)
        will_eat = (new_head == self.food)
        body_to_check = self.snake if will_eat else self.snake[:-1]
        if new_head in body_to_check:
            self.game_over = True
            self.score = self._compute_score()
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_after": self.get_state(),
                "result": "applied",
                "game_over": True,
                "api_ms": api_ms,
            }

        # Valid move: advance head
        self.snake.insert(0, new_head)
        if will_eat:
            self.foods_eaten += 1
            self.score = self._compute_score()
            if self.foods_eaten >= 15:
                self.game_over = True  # Won the game
            else:
                self.food = self._spawn_food()
        else:
            self.snake.pop()

        # Step cap check: 300 accepted moves
        if self.step >= 300:
            self.game_over = True

        api_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "state_after": self.get_state(),
            "result": "applied",
            "game_over": self.game_over,
            "api_ms": api_ms,
        }
