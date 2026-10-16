"""Snake Game Memory — Genesis 2.0 Track EVA 02.

Maintains spatial occupancy and visit history across Snake turns.
"""

from __future__ import annotations

from typing import Any


class SnakeMemory:
    """Tracks cell visit counts and historical collision outcomes."""

    def __init__(self) -> None:
        self.visits: dict[tuple[int, int], int] = {}
        self.recent_moves: list[dict[str, Any]] = []

    def observe(
        self,
        before_state: dict[str, Any],
        action: str,
        after_state: dict[str, Any],
    ) -> None:
        head = after_state.get("head")
        if head is not None and isinstance(head, tuple):
            self.visits[head] = self.visits.get(head, 0) + 1

        self.recent_moves.append({
            "action": action,
            "before_head": before_state.get("head"),
            "after_head": head,
            "score": after_state.get("score", 0),
        })
        if len(self.recent_moves) > 20:
            self.recent_moves.pop(0)

    def get_visit_count(self, pos: tuple[int, int]) -> int:
        return self.visits.get(pos, 0)
