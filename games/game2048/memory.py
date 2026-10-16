"""2048 Game Memory — Genesis 2.0 Track EVA 02 / Phase 2b.

Maintains max tile progress, score delta merge estimation, empty cell history,
rejections, and corner violation streak.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class Board2048Memory:
    """Stateful memory for tracking 2048 game transitions and stall warnings."""

    def __init__(self) -> None:
        self.max_tile_history: list[tuple[int, int]] = []
        self.score_history: list[tuple[int, int]] = []
        self.empty_count_history: deque[int] = deque(maxlen=8)
        self.rejection_history: deque[tuple[str, int]] = deque(maxlen=8)
        self.recent_actions: deque[str] = deque(maxlen=4)
        self.corner_violation_streak: int = 0
        self.stall_turns: int = 0
        self.merge_count_total: int = 0
        
        self._step: int = 0
        self._last_result: str = "applied"
        self._last_action: str = ""
        self._last_empty_count: int = 16

    def _extract_board(self, state: Any) -> list[list[int]]:
        if state is None:
            return [[0] * 4 for _ in range(4)]
        board = getattr(state, "board", None)
        if board is None and isinstance(state, dict):
            board = state.get("board") or state.get("grid") or [[0] * 4 for _ in range(4)]
        return [[int(cell) for cell in row] for row in board]

    def _extract_score(self, state: Any) -> int:
        if state is None:
            return 0
        score = getattr(state, "score", None)
        if score is None and isinstance(state, dict):
            score = state.get("score", 0)
        return int(score or 0)

    def _extract_max_tile(self, state: Any) -> int:
        if state is None:
            return 0
        best = getattr(state, "best_tile", None)
        if best is None and isinstance(state, dict):
            best = state.get("best_tile", state.get("max_tile"))
        if best is not None:
            return int(best)
        board = self._extract_board(state)
        return max(max(row) for row in board) if board else 0

    def observe_transition(
        self,
        before: Any,
        action: str,
        after: Any,
        result: str,
    ) -> None:
        """Observe step transition and maintain non-regression tracking."""
        act_clean = action.strip().lower()
        self._last_action = act_clean
        self._last_result = result

        if result == "rejected":
            # Record rejection; do NOT update stall_turns or empty count history
            self.rejection_history.append((act_clean, self._step))
            self._step += 1
            return

        # Move accepted
        self.recent_actions.append(act_clean)

        board_before = self._extract_board(before)
        board_after = self._extract_board(after)
        max_before = self._extract_max_tile(before)
        max_after = self._extract_max_tile(after)
        score_before = self._extract_score(before)
        score_after = self._extract_score(after)

        empty_after = sum(1 for r in range(4) for c in range(4) if board_after[r][c] == 0)
        self._last_empty_count = empty_after
        self.empty_count_history.append(empty_after)

        # Max tile tracking
        if max_after > max_before or not self.max_tile_history:
            self.max_tile_history.append((self._step, max_after))
            self.stall_turns = 0
        else:
            self.stall_turns += 1

        # Corner tracking: designated corner is (3, 0) bottom-left
        if board_after[3][0] == max_after:
            self.corner_violation_streak = 0
        else:
            self.corner_violation_streak += 1

        # Score delta and estimated merges
        score_delta = score_after - score_before
        if score_delta > 0:
            # Approximate number of merges from score delta
            self.merge_count_total += max(1, score_delta // 4)

        self.score_history.append((self._step, score_after))
        self._step += 1

    def observe(self, before: Any, action: str, after: Any) -> None:
        """Backward-compatibility entry point for orchestrator."""
        self.observe_transition(before, action, after, "applied")

    def warning_text(self) -> str:
        """Return at most ONE failure-adaptive warning string in strict priority order:

        1. Last action rejected.
        2. Corner violation streak >= 3.
        3. Empty cells <= 2.
        4. Stall turns >= 12.
        5. Empty string.
        """
        # Priority 1: Rejected
        if self._last_result == "rejected" and self._last_action:
            return (
                f"WARNING: '{self._last_action}' did not change the board and was rejected. "
                f"Choose a different direction."
            )

        # Priority 2: Corner violation streak >= 3
        if self.corner_violation_streak >= 3:
            return (
                f"WARNING: The largest tile has left the bottom-left corner for "
                f"{self.corner_violation_streak} turns. Prioritize restoring the corner arrangement."
            )

        # Priority 3: Low empty cells
        if self._last_empty_count <= 2 and self.empty_count_history:
            return (
                f"WARNING: Only {self._last_empty_count} empty cells remain. "
                f"Prioritize survival and merging over position."
            )

        # Priority 4: Stalling
        if self.stall_turns >= 12:
            return (
                f"WARNING: No new maximum tile for {self.stall_turns} turns. "
                f"Prefer moves that create merge opportunities and open space."
            )

        return ""

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot for rich diagnostic telemetry."""
        return {
            "max_tile_history": list(self.max_tile_history),
            "score_history": list(self.score_history),
            "empty_count_history": list(self.empty_count_history),
            "rejection_history": list(self.rejection_history),
            "recent_actions": list(self.recent_actions),
            "corner_violation_streak": self.corner_violation_streak,
            "stall_turns": self.stall_turns,
            "merge_count_total_est": self.merge_count_total,
            "step": self._step,
        }


# Alias for backward compatibility
Game2048Memory = Board2048Memory
