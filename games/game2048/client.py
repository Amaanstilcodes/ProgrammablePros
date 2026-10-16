"""2048 Client Wrapper — Genesis 2.0 Track EVA 02.

HTTP REST client conforming to the GameClient protocol for TopGameAI 2048 arena.
Supports offline mock simulation when is_mock=True.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any

from game_clients import Game2048Client, MoveResult


class Game2048ClientWrapper:
    """GameClient implementation for 2048."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
        is_mock: bool = False,
    ) -> None:
        self._is_mock = is_mock
        self._agent_name = agent_name
        self._client: Game2048Client | None = None if is_mock else Game2048Client(api_key=api_key, agent_name=agent_name)
        # Mock simulation state
        self._step_count = 0
        self._board = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score = 16
        self._game_over = False

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        name = agent_name or self._agent_name
        if not self._is_mock and self._client is not None:
            return self._client.start_game(name)

        # Mock start
        self._step_count = 0
        self._board = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score = 16
        self._game_over = False
        initial_state = {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": 8,
            "game_over": False,
            "step": 0,
        }
        return f"MOCK-2048-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        if not self._is_mock and self._client is not None:
            return self._client.send_move(session_id, action, state_before=state_before)

        # Mock move execution
        t0 = time.perf_counter()
        self._step_count += 1

        # Spawn a new tile in an empty cell
        empty_coords = [(r, c) for r in range(4) for c in range(4) if self._board[r][c] == 0]
        if empty_coords:
            r, c = random.choice(empty_coords)
            self._board[r][c] = random.choice([2, 4])
            self._score += self._board[r][c]

        max_tile = max(max(row) for row in self._board)
        api_ms = (time.perf_counter() - t0) * 1000.0

        state_after = {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": max_tile,
            "game_over": self._game_over,
            "step": self._step_count,
        }
        return MoveResult(
            state_after=state_after,
            result="applied",
            game_over=self._game_over,
            api_ms=round(api_ms, 3),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
