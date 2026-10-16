"""2048 State Parser — Genesis 2.0 Track EVA 02.

Parses raw server state from TopGameAI 2048 arena.
Extracts 4x4 board matrix, score, maximum tile value, and terminal flags.
Guarantees zero mutation of incoming raw dictionaries.
"""

from __future__ import annotations

import copy
from typing import Any

from core.protocols import GameStateView, SchemaError, StateParser


class Game2048StateParser:
    """Parses and validates 2048 state dictionaries into GameStateView."""

    def validate(self, raw_state: dict[str, Any]) -> None:
        """Validate raw state dictionary against contract schema.

        Raises:
            SchemaError: if required structural fields are missing or malformed.
        """
        if not isinstance(raw_state, dict):
            raise SchemaError(f"Expected dict for raw_state, got {type(raw_state).__name__}")

        has_board = any(k in raw_state for k in ("board", "grid", "tiles"))
        if not has_board:
            raise SchemaError("Missing required board matrix in 2048 state payload")

    def parse(self, raw_state: dict[str, Any], session_id: str = "") -> GameStateView:
        """Convert raw server response into an immutable GameStateView."""
        self.validate(raw_state)
        raw_copy = copy.deepcopy(raw_state)

        raw_board = (
            raw_copy.get("board")
            or raw_copy.get("grid")
            or raw_copy.get("tiles")
            or [[0] * 4 for _ in range(4)]
        )

        board_rows: list[tuple[int, ...]] = []
        for r in range(4):
            row: list[int] = []
            for c in range(4):
                val = 0
                if r < len(raw_board) and c < len(raw_board[r]):
                    try:
                        val = int(raw_board[r][c])
                    except (TypeError, ValueError):
                        val = 0
                row.append(val)
            board_rows.append(tuple(row))

        board_tuple = tuple(board_rows)

        # Compute best tile safely
        computed_best = max(max(row) for row in board_tuple) if board_tuple else 0
        best_tile = int(raw_copy.get("best_tile", raw_copy.get("max_tile", computed_best)))

        score = int(raw_copy.get("score", 0))
        game_over = bool(
            raw_copy.get("game_over")
            or raw_copy.get("done")
            or raw_copy.get("finished")
            or False
        )
        step = int(raw_copy.get("step", 0)) if "step" in raw_copy else None
        sid = session_id or str(raw_copy.get("session_id", ""))

        extra = {k: v for k, v in raw_copy.items() if k not in ("board", "grid", "tiles", "score", "max_tile", "best_tile", "game_over", "done", "finished", "step")}

        return GameStateView(
            game="2048",
            session_id=sid,
            raw=raw_copy,
            board=board_tuple,
            score=score,
            best_tile=best_tile,
            step=step,
            game_over=game_over,
            extra=extra,
        )

    def is_game_over(self, raw_state: dict[str, Any]) -> bool:
        """Check whether the 2048 game has terminated."""
        return bool(
            raw_state.get("game_over")
            or raw_state.get("done")
            or raw_state.get("finished")
            or False
        )

    def is_rejected(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        response: dict[str, Any],
    ) -> bool:
        """Determine whether move was rejected (no change in board or explicit flag)."""
        if response.get("result") == "rejected" or response.get("rejected") is True:
            return True
        if response.get("result") == "applied":
            return False

        view_before = self.parse(before)
        view_after = self.parse(after)
        if view_before.board is not None and view_after.board is not None:
            return view_before.board == view_after.board
        return False

    def score_of(self, raw_state: dict[str, Any]) -> int:
        """Extract score from state."""
        return int(raw_state.get("score", 0))
