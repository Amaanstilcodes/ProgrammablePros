"""2048-Specific Decision Brain — Genesis 2.0 Track EVA 02.

Adapts the decision pipeline for 2048:
- Board2048Compressor: Compresses 4x4 board into <= 200 char string
- Board2048SubgoalManager: Priority cascade (P1: SURVIVE if empty <= 2, P2: KEEP_CORNER)
- Board2048HypothesisBuilder: 8 typed questions (4 directions x 2 questions)
- Board2048EVCalculator: Multi-term expected value with structure reward & danger penalty
"""

from __future__ import annotations

import math
from typing import Any


class Board2048Compressor:
    """Compresses 2048 board into a dense prompt string (<= 200 chars)."""

    def __init__(self) -> None:
        self.last_failed_action: str | None = None

    def record_failure(self, action: str) -> None:
        self.last_failed_action = action

    def clear_failure(self) -> None:
        self.last_failed_action = None

    def compress(self, state: dict[str, Any]) -> str:
        board = state.get("board", [[0] * 4 for _ in range(4)])
        score = state.get("score", 0)
        step = state.get("step", 0)
        steps_rem = max(0, 2000 - step)

        # Max tile & log2
        best_tile = 0
        best_pos = (0, 0)
        empty_count = 0
        for r in range(4):
            for c in range(4):
                val = board[r][c]
                if val == 0:
                    empty_count += 1
                elif val > best_tile:
                    best_tile = val
                    best_pos = (r, c)

        log2_best = int(math.log2(best_tile)) if best_tile >= 2 else 0

        # Position descriptor (corner: (0,0),(0,3),(3,0),(3,3))
        r, c = best_pos
        if (r in (0, 3)) and (c in (0, 3)):
            pos_desc = "corner"
        elif r in (0, 3) or c in (0, 3):
            pos_desc = "edge"
        else:
            pos_desc = "center"

        # Smoothness: sum of |log2(a) - log2(b)| for adjacent non-zero pairs
        smoothness = 0
        for r_idx in range(4):
            for c_idx in range(4):
                val = board[r_idx][c_idx]
                if val <= 0:
                    continue
                lv = int(math.log2(val))
                if c_idx + 1 < 4 and board[r_idx][c_idx + 1] > 0:
                    smoothness += abs(lv - int(math.log2(board[r_idx][c_idx + 1])))
                if r_idx + 1 < 4 and board[r_idx + 1][c_idx] > 0:
                    smoothness += abs(lv - int(math.log2(board[r_idx + 1][c_idx])))

        # Monotonicity check
        mono = 0
        for r_idx in range(4):
            row = board[r_idx]
            if all(row[i] >= row[i + 1] for i in range(3)) or all(row[i] <= row[i + 1] for i in range(3)):
                mono += 1
        for c_idx in range(4):
            col = [board[r_idx][c_idx] for r_idx in range(4)]
            if all(col[i] >= col[i + 1] for i in range(3)) or all(col[i] <= col[i + 1] for i in range(3)):
                mono += 1

        warn = f" W!{self.last_failed_action}" if self.last_failed_action else ""

        compact = (
            f"T:{best_tile}(2^{log2_best})@{pos_desc}({r},{c}) "
            f"E:{empty_count} Sm:{smoothness} M:{mono} "
            f"Sc:{score} S:{steps_rem}{warn}"
        )
        return compact[:200]


class Board2048SubgoalManager:
    """Deterministic priority cascade for 2048 subgoals."""

    def get_current_subgoal(self, state: dict[str, Any]) -> str:
        board = state.get("board", [[0] * 4 for _ in range(4)])
        empty_count = sum(1 for row in board for val in row if val == 0)

        # Priority 1: SURVIVE if empty cells <= 2
        if empty_count <= 2:
            return "creating empty space and merging to survive"
        # Priority 2: KEEP_CORNER (default)
        return "keeping the largest tile in the bottom-left corner and merging along the bottom row and left column"


class Board2048HypothesisBuilder:
    """Builds 8 typed hypothesis questions (4 directions x 2 questions)."""

    DIRECTIONS: list[str] = ["up", "down", "left", "right"]

    def build_hypotheses(self, state: dict[str, Any], subgoal: str, compressed: str) -> list[str]:
        questions: list[str] = []
        for action in self.DIRECTIONS:
            q1 = (
                f"Context: {compressed}. Hypothesis: Sliding '{action}' changes "
                f"the board and progresses toward {subgoal}. What is the calibrated "
                f"probability (0.0 to 1.0) that this is true?"
            )
            q2 = (
                f"Context: {compressed}. Hypothesis: Sliding '{action}' leaves "
                f"the board with <= 1 empty cell and no merges, risking game over. "
                f"What is the calibrated probability (0.0 to 1.0) that this is true?"
            )
            questions.append(q1)
            questions.append(q2)
        return questions


class Board2048EVCalculator:
    """Calculates EV for each direction based on progress, structure, and danger probabilities."""

    def __init__(self) -> None:
        self.move_history: list[str] = []

    def record_move(self, action: str) -> None:
        self.move_history.append(action)
        if len(self.move_history) > 10:
            self.move_history.pop(0)

    @staticmethod
    def _simulate_slide_board(board: list[list[int]], action: str) -> list[list[int]]:
        """Lightweight simulation to see if action moves max tile or changes board."""
        def slide_row(r: list[int]) -> list[int]:
            tiles = [x for x in r if x != 0]
            merged: list[int] = []
            i = 0
            while i < len(tiles):
                if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                    merged.append(tiles[i] * 2)
                    i += 2
                else:
                    merged.append(tiles[i])
                    i += 1
            while len(merged) < 4:
                merged.append(0)
            return merged

        new_b = [list(row) for row in board]
        if action == "left":
            for r in range(4):
                new_b[r] = slide_row(new_b[r])
        elif action == "right":
            for r in range(4):
                new_b[r] = list(reversed(slide_row(list(reversed(new_b[r])))))
        elif action == "up":
            for c in range(4):
                col = [new_b[r][c] for r in range(4)]
                sc = slide_row(col)
                for r in range(4):
                    new_b[r][c] = sc[r]
        elif action == "down":
            for c in range(4):
                col = [new_b[r][c] for r in range(4)]
                sc = list(reversed(slide_row(list(reversed(col)))))
                for r in range(4):
                    new_b[r][c] = sc[r]
        return new_b

    def calculate(
        self,
        state: dict[str, Any],
        raw_probs: dict[str, dict[str, float]],
        subgoal: str,
    ) -> tuple[str, dict[str, Any]]:
        """Calculate EV per direction.

        EV(a) = P_progress(a) * 12.0 + P_structure(a) * 4.0 - P_danger(a) * 30.0 - repeat_penalty(a)
        """
        board = state.get("board", [[0] * 4 for _ in range(4)])
        best_tile = max(max(row) for row in board)

        ev_breakdown: dict[str, Any] = {}
        best_action = "down"
        best_ev = -1e9

        for action in ("up", "down", "left", "right"):
            probs = raw_probs.get(action, {"q1": 0.25, "q2": 0.25})
            p_progress = float(probs.get("q1", 0.25))
            p_danger = float(probs.get("q2", 0.25))

            # Structure term: 1.0 if action moves/keeps max tile at bottom-left (r=3, c=0), else 0.0
            p_structure = 0.0
            simulated_b = self._simulate_slide_board(board, action)
            if simulated_b != board:
                # If bottom-left tile in simulated board is the best tile
                if simulated_b[3][0] == best_tile:
                    p_structure = 1.0
                elif action in ("down", "left"):
                    p_structure = 0.5

            # Repeat penalty: +3.0 if same as last 2 moves
            repeat_penalty = 0.0
            if len(self.move_history) >= 2 and self.move_history[-1] == action and self.move_history[-2] == action:
                repeat_penalty = 3.0

            # Illegal penalty: if board is unchanged by action (would be rejected)
            illegal_penalty = 0.0
            if simulated_b == board:
                illegal_penalty = 50.0

            ev = (p_progress * 12.0) + (p_structure * 4.0) - (p_danger * 30.0) - repeat_penalty - illegal_penalty

            ev_breakdown[action] = {
                "ev": round(ev, 3),
                "p_progress": round(p_progress, 4),
                "p_structure": round(p_structure, 2),
                "p_danger": round(p_danger, 4),
                "repeat_penalty": repeat_penalty,
                "illegal_penalty": illegal_penalty,
            }

            if ev > best_ev:
                best_ev = ev
                best_action = action

        return best_action, ev_breakdown
