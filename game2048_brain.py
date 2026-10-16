"""game2048_brain.py — 2048 Perception, Hypothesis, and EV Pipeline — Genesis 2.0 Track EVA 02.

HACKATHON DAY PRODUCTION IMPLEMENTATION — STRONG CORNER STRATEGY:
- Board2048Compressor: Compresses 4x4 grid into <= 200 char string with corner lock and descending chain scores.
- Board2048HypothesisBuilder: Formulates noul questions targeting corner anchoring (row 3, col 0) and descending chains.
- Board2048EVCalculator: Multi-term Expected Value prioritizing bottom-left corner retention (LEFT > DOWN > UP > RIGHT):
    EV = (P_progress * 20.0) - (P_danger * 50.0) + corner_bonus - illegal_penalty - repeat_penalty
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

LEGAL_ACTIONS: List[str] = ["left", "down", "up", "right"]
ACTION_PRIORITY: Dict[str, int] = {"left": 4, "down": 3, "up": 2, "right": 1}


def parse_2048_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse real TopGameAI 2048 arena state into standardized fields."""
    raw_board = state.get("grid") or state.get("board") or [[0] * 4 for _ in range(4)]
    board: List[List[int]] = []
    if isinstance(raw_board, list):
        for row in raw_board:
            if isinstance(row, list):
                board.append([int(x) for x in row])
            else:
                board.append([0, 0, 0, 0])
    while len(board) < 4:
        board.append([0, 0, 0, 0])
    for i in range(4):
        while len(board[i]) < 4:
            board[i].append(0)

    score = int(state.get("score", 0))
    step = int(state.get("steps", state.get("step", 0)))
    done = bool(state.get("done", state.get("game_over", False)))
    won = bool(state.get("won", False))

    max_tile = 0
    empty_count = 0
    for r in range(4):
        for c in range(4):
            val = board[r][c]
            if val == 0:
                empty_count += 1
            elif val > max_tile:
                max_tile = val

    # Corner check: bottom-left corner is row 3, col 0
    in_bl_corner = 1 if (board[3][0] == max_tile and max_tile > 0) else 0

    return {
        "board": board,
        "grid": board,
        "max_tile": max_tile,
        "best_tile": max_tile,
        "empty_count": empty_count,
        "corner": in_bl_corner,
        "score": score,
        "step": step,
        "steps": step,
        "done": done,
        "won": won,
    }


def compute_chain_score(board: List[List[int]]) -> int:
    """Calculate descending chain score along row 3 (bottom) and col 0 (left).

    Row 3: (3,0) >= (3,1) >= (3,2) >= (3,3)
    Col 0: (3,0) >= (2,0) >= (1,0) >= (0,0) (ascending upward from bottom-left corner)
    """
    score = 0
    # Bottom row descending from col 0 to col 3
    for c in range(3):
        if board[3][c] > 0 and board[3][c] >= board[3][c + 1]:
            score += 1
        elif board[3][c] > 0 and board[3][c + 1] == 0:
            score += 1

    # Left column descending from row 3 up to row 0
    for r in (3, 2, 1):
        if board[r][0] > 0 and board[r][0] >= board[r - 1][0]:
            score += 1
        elif board[r][0] > 0 and board[r - 1][0] == 0:
            score += 1

    return score


class Board2048Compressor:
    """Compresses 2048 game state into a dense prompt string (<= 200 chars).

    Format:
    max={max_tile} empty={empty_count} corner={1|0} chain={chain_score} score={score}
    """

    def __init__(self) -> None:
        self.last_failed_action: str | None = None

    def record_failure(self, action: str) -> None:
        self.last_failed_action = action

    def clear_failure(self) -> None:
        self.last_failed_action = None

    def compress(self, state: Dict[str, Any]) -> str:
        parsed = parse_2048_state(state)
        board = parsed["board"]
        max_tile = parsed["max_tile"]
        empty_count = parsed["empty_count"]
        corner = parsed["corner"]
        score = parsed["score"]
        chain = compute_chain_score(board)

        row_str = "/".join(",".join(str(x) for x in row) for row in board)
        warn = f" W!{self.last_failed_action}" if self.last_failed_action else ""

        compact = (
            f"max={max_tile} empty={empty_count} corner={corner} chain={chain} "
            f"score={score} B:{row_str}{warn}"
        )
        return compact[:200]


class Board2048HypothesisBuilder:
    """Builds noul typed questions for Laya targeting corner retention and descending chain."""

    DIRECTIONS: List[str] = ["up", "down", "left", "right"]

    def build_hypotheses(self, state: Dict[str, Any], compressed: str = "") -> Dict[str, Dict[str, Any]]:
        """Return dict of noul typed questions keyed by q_{action}_progress and q_{action}_danger."""
        parsed = parse_2048_state(state)
        max_tile = parsed["max_tile"]

        questions: Dict[str, Dict[str, Any]] = {}
        for action in self.DIRECTIONS:
            # Q1 (progress): merges tiles & keeps max tile in bottom-left corner with descending chain
            questions[f"q_{action}_progress"] = {
                "type": "noul",
                "instructions": (
                    f"Sliding '{action}' merges tiles and keeps the highest tile ({max_tile}) in the "
                    f"bottom-left corner while building a descending chain along the bottom row and left column."
                ),
            }
            # Q2 (danger): leaves board with <= 2 empty spaces and no merges
            questions[f"q_{action}_danger"] = {
                "type": "noul",
                "instructions": (
                    f"Sliding '{action}' leaves the board with <= 2 empty spaces and no immediate "
                    f"merges, risking game over."
                ),
            }
        return questions

    def build_hypothesis_strings(self, state: Dict[str, Any], compressed: str = "") -> List[str]:
        """Return list of primary question strings for official logging compliance."""
        parsed = parse_2048_state(state)
        max_tile = parsed["max_tile"]
        return [
            (
                f"Sliding '{action}' merges tiles and keeps the highest tile ({max_tile}) in the "
                f"bottom-left corner while building a descending chain along the bottom row and left column."
            )
            for action in self.DIRECTIONS
        ]


class Board2048EVCalculator:
    """Calculates Expected Value (EV) per candidate action for 2048 with a strong corner strategy.

    Move Preference Hierarchy:
      LEFT > DOWN > UP > RIGHT
      - LEFT: Merges and compacts tiles into the bottom-left corner along row 3.
      - DOWN: Compacts tiles into the bottom-left corner along col 0.
      - UP / RIGHT: Penalized as emergency moves because they dislodge the corner anchor.
    """

    def __init__(self) -> None:
        self.move_history: List[str] = []

    def record_move(self, action: str) -> None:
        self.move_history.append(action)
        if len(self.move_history) > 10:
            self.move_history.pop(0)

    @staticmethod
    def simulate_slide_board(board: List[List[int]], action: str) -> List[List[int]]:
        """Simulate a 2048 slide on a 4x4 grid."""
        def slide_line(line: List[int]) -> List[int]:
            non_zero = [x for x in line if x != 0]
            merged: List[int] = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged.append(non_zero[i] * 2)
                    i += 2
                else:
                    merged.append(non_zero[i])
                    i += 1
            while len(merged) < 4:
                merged.append(0)
            return merged

        new_b = [list(r) for r in board]
        if action == "left":
            for r in range(4):
                new_b[r] = slide_line(new_b[r])
        elif action == "right":
            for r in range(4):
                rev = list(reversed(new_b[r]))
                new_b[r] = list(reversed(slide_line(rev)))
        elif action == "up":
            for c in range(4):
                col = [new_b[r][c] for r in range(4)]
                res = slide_line(col)
                for r in range(4):
                    new_b[r][c] = res[r]
        elif action == "down":
            for c in range(4):
                col = [new_b[r][c] for r in range(4)]
                rev = list(reversed(col))
                res = list(reversed(slide_line(rev)))
                for r in range(4):
                    new_b[r][c] = res[r]

        return new_b

    def calculate(
        self,
        state: Dict[str, Any],
        raw_probs: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Calculate Expected Value for each candidate direction using the Strong Corner Strategy.

        Args:
            state: Game state dictionary.
            raw_probs: Probability mapping per action.

        Returns:
            Tuple of (best_action, ev_breakdown_dict).
        """
        parsed = parse_2048_state(state)
        board = parsed["board"]
        max_tile = parsed["max_tile"]
        is_max_in_corner = (board[3][0] == max_tile and max_tile > 0)

        ev_breakdown: Dict[str, Any] = {}
        best_action = "left"
        best_ev = -1e9

        for action in ("left", "down", "up", "right"):
            prob_info = raw_probs.get(action, {})
            if isinstance(prob_info, dict):
                p_progress = float(prob_info.get("progress", prob_info.get("q1", 0.25)))
                p_danger = float(prob_info.get("danger", prob_info.get("q2", 0.25)))
            elif isinstance(prob_info, (int, float)):
                p_progress = float(prob_info)
                p_danger = 0.25
            else:
                p_progress = 0.25
                p_danger = 0.25

            # 1. Base EV from calibrated probabilities
            ev = (p_progress * 20.0) - (p_danger * 50.0)

            # 2. Corner lock preference: strongly favor LEFT and DOWN
            if action in ("left", "down"):
                ev += 15.0  # Strongly prefer LEFT and DOWN
            elif action in ("up", "right"):
                ev -= 20.0  # Penalize UP and RIGHT (pulls max tile from corner)

            # 3. Max tile position adjustment
            if not is_max_in_corner:
                if action in ("left", "down"):
                    ev += 25.0  # Extra bonus to restore corner
                elif action in ("up", "right"):
                    ev -= 30.0  # Extra penalty for worsening corner position

            # 4. Check simulated board change
            simulated_b = self.simulate_slide_board(board, action)
            board_changed = (simulated_b != board)

            illegal_penalty = 0.0
            if not board_changed:
                illegal_penalty = 100.0
                ev -= 100.0

            # If currently in corner, heavily penalize moves that dislodge max tile from (3, 0)
            if is_max_in_corner and board_changed:
                sim_max = max(max(r) for r in simulated_b)
                if simulated_b[3][0] != sim_max:
                    ev -= 40.0  # Severe penalty for dislodging max tile from bottom-left corner

            # 5. Repeat penalty: if last 2 moves == action
            repeat_penalty = 0.0
            if len(self.move_history) >= 2 and self.move_history[-1] == action and self.move_history[-2] == action:
                repeat_penalty = 10.0
                ev -= 10.0

            ev_breakdown[action] = {
                "ev": round(ev, 3),
                "p_progress": round(p_progress, 4),
                "p_danger": round(p_danger, 4),
                "board_changed": board_changed,
                "illegal_penalty": illegal_penalty,
                "repeat_penalty": repeat_penalty,
            }

            # Select best action: higher EV, breaking ties by ACTION_PRIORITY (LEFT > DOWN > UP > RIGHT)
            if ev > best_ev or (abs(ev - best_ev) < 1e-6 and ACTION_PRIORITY[action] > ACTION_PRIORITY.get(best_action, 0)):
                best_ev = ev
                best_action = action

        return best_action, ev_breakdown


__all__ = [
    "Board2048Compressor",
    "Board2048HypothesisBuilder",
    "Board2048EVCalculator",
    "compute_chain_score",
    "parse_2048_state",
]
