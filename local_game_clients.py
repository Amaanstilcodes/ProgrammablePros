"""Local Game Clients — Genesis 2.0 Track EVA 02.

Provides client wrappers that mimic the hosted TopGameAI API exactly,
allowing local runs to use the exact same signature as hosted game sessions.
"""

from __future__ import annotations

import time
from typing import Any

from game2048 import Game2048
from snake_game import SnakeGame


class LocalSnakeClient:
    """Client wrapper for local SnakeGame simulation."""

    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self.seed = seed
        self._game: SnakeGame | None = None

    def start_game(self, agent_name: str = "ProgrammablePros") -> tuple[str, dict[str, Any]]:
        """Start a new game session.

        Returns:
            (session_id, initial_state)
        """
        self._game = SnakeGame(width=self.width, height=self.height, seed=self.seed)
        state = self._game.get_state()
        return state["session_id"], state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a move to the local game engine.

        Returns:
            dict containing state_after, result, game_over, api_ms
        """
        if self._game is None:
            raise RuntimeError("Game session not started. Call start_game() first.")
        return self._game.move(action)

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Inspect current state."""
        if self._game is None:
            raise RuntimeError("Game session not started. Call start_game() first.")
        return self._game.get_state()

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Retrieve final state of the game."""
        return self.get_state(session_id)

    def close(self) -> None:
        """Close client resources."""
        self._game = None


class LocalGame2048Client:
    """Client wrapper for local Game2048 simulation."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._game: Game2048 | None = None

    def start_game(self, agent_name: str = "ProgrammablePros") -> tuple[str, dict[str, Any]]:
        """Start a new 2048 game session.

        Returns:
            (session_id, initial_state)
        """
        self._game = Game2048(seed=self.seed)
        state = self._game.get_state()
        return state["session_id"], state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a move to the 2048 game engine.

        Returns:
            dict containing state_before, state_after, result, game_over, api_ms
        """
        if self._game is None:
            raise RuntimeError("Game session not started. Call start_game() first.")
        return self._game.move(action)

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Inspect current state."""
        if self._game is None:
            raise RuntimeError("Game session not started. Call start_game() first.")
        return self._game.get_state()

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Retrieve final state of the game."""
        return self.get_state(session_id)

    def close(self) -> None:
        """Close client resources."""
        self._game = None
