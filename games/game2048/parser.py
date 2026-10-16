"""2048 State Parser — Genesis 2.0 Track EVA 02.

Parses raw server state from TopGameAI 2048 arena.
Extracts 4x4 board matrix, score, maximum tile value, and terminal flags.
"""

from __future__ import annotations

from typing import Any


class Game2048StateParser:
    """Parses and validates 2048 state dictionaries."""

    def parse(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        raw_board = (
            raw_state.get("board")
            or raw_state.get("grid")
            or raw_state.get("tiles")
            or [[0] * 4 for _ in range(4)]
        )

        board: list[list[int]] = []
        for r in range(4):
            row = []
            for c in range(4):
                val = 0
                if r < len(raw_board) and c < len(raw_board[r]):
                    val = int(raw_board[r][c])
                row.append(val)
            board.append(row)

        score = int(raw_state.get("score", 0))
        max_tile = int(raw_state.get("max_tile", max(max(row) for row in board)))
        game_over = bool(
            raw_state.get("game_over")
            or raw_state.get("done")
            or raw_state.get("finished")
            or False
        )
        step = int(raw_state.get("step", 0))

        return {
            "board": board,
            "score": score,
            "max_tile": max_tile,
            "game_over": game_over,
            "step": step,
            "raw": raw_state,
        }
