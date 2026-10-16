"""2048 Memory Module — Genesis 2.0 Track EVA 02.

Tracks historical board transitions and repeated rejection outcomes for 2048.
"""

from __future__ import annotations

from typing import Any


class Game2048Memory:
    """Tracks board history and rejection streaks."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.rejected_actions: list[str] = []

    def observe(
        self,
        before_state: dict[str, Any],
        action: str,
        after_state: dict[str, Any],
    ) -> None:
        board_before = before_state.get("board")
        board_after = after_state.get("board")

        # Track silent rejection if board was unchanged
        if board_before is not None and board_after is not None and board_before == board_after:
            self.rejected_actions.append(action)
        else:
            self.rejected_actions.clear()

        self.history.append({
            "action": action,
            "max_tile": after_state.get("max_tile", 0),
            "score": after_state.get("score", 0),
        })
        if len(self.history) > 30:
            self.history.pop(0)
