"""game_clients.py — Compatibility Re-export Shim — Genesis 2.0 Track EVA 02.

Deprecated: Use `games.snake.client` and `games.game2048.client` directly.
Kept to ensure legacy scripts and viewers run without breaking imports.
"""

from __future__ import annotations

from core.protocols import (
    ContractNotDiscoveredError,
    GameStateView,
    MoveResult,
    SessionStart,
)
from games.game2048.client import Game2048Client
from games.snake.client import BaseGameClient, GameAPIError, SnakeClient

__all__ = [
    "BaseGameClient",
    "SnakeClient",
    "Game2048Client",
    "MoveResult",
    "SessionStart",
    "GameStateView",
    "ContractNotDiscoveredError",
    "GameAPIError",
]
