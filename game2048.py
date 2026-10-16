"""Local 2048 Game Engine — Genesis 2.0 Track EVA 02.

Implements the local 2048 game according to Track2_PS2_support.pdf §4.2:
- 4x4 grid, merge equal adjacent tiles once per move
- After each applied move, spawn a 2 (90%) or 4 (10%) in a random empty cell
- Move that doesn't change the board -> rejected, NOT a step
- Step cap: 2000 accepted moves
- Points = min(1000, max(0, (log2(best_tile) - 6) * 143 + floor(score / 1000)))
"""

from __future__ import annotations

import math
import random
import time
import uuid
from typing import Any


class Game2048:
    """Local 2048 simulation matching the TopGameAI hosted API response schema."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset game to initial 4x4 board with two spawned tiles."""
        if seed is not None:
            self._rng = random.Random(seed)

        self.session_id = f"local_2048_{uuid.uuid4().hex[:8]}"
        self.board: list[list[int]] = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.step = 0
        self.game_over = False

        # Spawn two initial tiles
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def _spawn_tile(self) -> bool:
        """Spawn a 2 (90%) or 4 (10%) in a random empty cell."""
        empty = [
            (r, c)
            for r in range(4)
            for c in range(4)
            if self.board[r][c] == 0
        ]
        if not empty:
            return False
        r, c = self._rng.choice(empty)
        self.board[r][c] = 2 if self._rng.random() < 0.9 else 4
        return True

    def _calculate_points(self) -> int:
        """Official 2048 points formula."""
        best_tile = max(max(row) for row in self.board)
        if best_tile < 2:
            return 0
        exponent = int(math.log2(best_tile))
        return min(1000, max(0, (exponent - 6) * 143 + self.score // 1000))

    def get_state(self) -> dict[str, Any]:
        """Return state dictionary matching the official schema."""
        best_tile = max(max(row) for row in self.board)
        points = self._calculate_points()
        return {
            "session_id": self.session_id,
            "game": "2048",
            "board": [list(row) for row in self.board],
            "score": self.score,
            "best_tile": best_tile,
            "points": points,
            "game_over": self.game_over,
            "step": self.step,
        }

    @staticmethod
    def _slide_and_merge_row(row: list[int]) -> tuple[list[int], int]:
        """Slide non-zero tiles left and merge once per pair. Returns (new_row, gained_score)."""
        tiles = [x for x in row if x != 0]
        merged: list[int] = []
        gained = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                val = tiles[i] * 2
                merged.append(val)
                gained += val
                i += 2
            else:
                merged.append(tiles[i])
                i += 1

        while len(merged) < 4:
            merged.append(0)

        return merged, gained

    def _simulate_slide(self, board: list[list[int]], action: str) -> tuple[list[list[int]], int]:
        """Simulate a move on a 4x4 board without side-effects."""
        act = action.strip().lower()
        new_board = [list(row) for row in board]
        total_gained = 0

        if act == "left":
            for r in range(4):
                new_row, gained = self._slide_and_merge_row(new_board[r])
                new_board[r] = new_row
                total_gained += gained
        elif act == "right":
            for r in range(4):
                reversed_row = list(reversed(new_board[r]))
                new_row, gained = self._slide_and_merge_row(reversed_row)
                new_board[r] = list(reversed(new_row))
                total_gained += gained
        elif act == "up":
            for c in range(4):
                col = [new_board[r][c] for r in range(4)]
                new_col, gained = self._slide_and_merge_row(col)
                for r in range(4):
                    new_board[r][c] = new_col[r]
                total_gained += gained
        elif act == "down":
            for c in range(4):
                col = [new_board[r][c] for r in range(4)]
                new_col, gained = self._slide_and_merge_row(list(reversed(col)))
                new_col = list(reversed(new_col))
                for r in range(4):
                    new_board[r][c] = new_col[r]
                total_gained += gained
        else:
            return new_board, 0

        return new_board, total_gained

    def _is_move_possible(self) -> bool:
        """Check if any legal move is available in the current board."""
        for act in ("up", "down", "left", "right"):
            b, _ = self._simulate_slide(self.board, act)
            if b != self.board:
                return True
        return False

    def move(self, action: str) -> dict[str, Any]:
        """Execute a move on the 2048 board.

        Returns:
            dict with state_before, state_after, result, game_over, api_ms
        """
        t0 = time.perf_counter()
        state_before = self.get_state()
        act = str(action).strip().lower()

        if self.game_over:
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_before": state_before,
                "state_after": self.get_state(),
                "result": "rejected",
                "game_over": True,
                "api_ms": api_ms,
            }

        new_board, gained_score = self._simulate_slide(self.board, act)

        # Rejection: If board is unchanged after move -> rejected
        if new_board == self.board:
            api_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "state_before": state_before,
                "state_after": self.get_state(),
                "result": "rejected",
                "game_over": self.game_over,
                "api_ms": api_ms,
            }

        # Applied move
        self.board = new_board
        self.score += gained_score
        self.step += 1

        # Spawn tile
        self._spawn_tile()

        # Check termination: no moves possible or step cap reached (2000)
        if not self._is_move_possible() or self.step >= 2000:
            self.game_over = True

        state_after = self.get_state()
        api_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "state_before": state_before,
            "state_after": state_after,
            "result": "applied",
            "game_over": self.game_over,
            "api_ms": api_ms,
        }
