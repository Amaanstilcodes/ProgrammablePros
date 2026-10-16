"""Snake Client Wrapper — Genesis 2.0 Track EVA 02.

HTTP REST client conforming to the GameClient protocol for TopGameAI Snake arena.
Supports offline mock simulation when is_mock=True.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from game_clients import MoveResult, SnakeClient


class SnakeClientWrapper:
    """GameClient implementation for Snake."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
        is_mock: bool = False,
    ) -> None:
        self._is_mock = is_mock
        self._agent_name = agent_name
        self._client: SnakeClient | None = None if is_mock else SnakeClient(api_key=api_key, agent_name=agent_name)
        # Mock simulation state
        self._step_count = 0
        self._snake: list[list[int]] = [[5, 5], [5, 4], [5, 3]]
        self._food: list[int] = [8, 5]
        self._direction: str = "right"
        self._score: int = 0
        self._game_over: bool = False

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        name = agent_name or self._agent_name
        if not self._is_mock and self._client is not None:
            return self._client.start_game(name)

        # Mock start
        self._step_count = 0
        self._snake = [[5, 5], [5, 4], [5, 3]]
        self._food = [8, 5]
        self._direction = "right"
        self._score = 0
        self._game_over = False
        initial_state = {
            "snake": [list(p) for p in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": False,
            "step": 0,
        }
        return f"MOCK-SNAKE-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        if not self._is_mock and self._client is not None:
            return self._client.send_move(session_id, action)

        # Mock move execution
        t0 = time.perf_counter()
        act = action.strip().lower()
        self._step_count += 1

        deltas = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        dx, dy = deltas.get(act, (0, 0))
        head = self._snake[0]
        new_head = [head[0] + dx, head[1] + dy]

        # 16x16 grid bounds check
        if new_head[0] < 0 or new_head[0] >= 16 or new_head[1] < 0 or new_head[1] >= 16:
            result = "rejected"
            new_head = head
        else:
            result = "applied"
            self._direction = act
            self._snake.insert(0, new_head)
            if new_head == self._food:
                self._score += 60
                self._food = [(self._food[0] + 5) % 15, (self._food[1] + 3) % 15]
            else:
                self._snake.pop()

        api_ms = (time.perf_counter() - t0) * 1000.0
        state_after = {
            "snake": [list(p) for p in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": self._game_over,
            "step": self._step_count,
        }
        return MoveResult(
            state_after=state_after,
            result=result,
            game_over=self._game_over,
            api_ms=round(api_ms, 3),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
