"""2048 Feature Extractor — Genesis 2.0 Track EVA 02.

Computes high-signal heuristic features from the 4x4 matrix:
- Empty cell count (board fluidity)
- Max tile value and corner alignment status
- Number of adjacent mergeable pairs
- Monotonicity gradient
"""

from __future__ import annotations

from typing import Any


class Game2048FeatureExtractor:
    """Extracts tactical state features for 2048."""

    def extract_features(
        self,
        parsed_state: dict[str, Any],
        memory: Any,
    ) -> dict[str, Any]:
        board: list[list[int]] = parsed_state["board"]

        empty_cells = sum(row.count(0) for row in board)
        max_tile = parsed_state["max_tile"]

        # Find location of max tile
        max_r, max_c = 0, 0
        for r in range(4):
            for c in range(4):
                if board[r][c] == max_tile:
                    max_r, max_c = r, c

        is_corner = (max_r in (0, 3)) and (max_c in (0, 3))

        # Count available merges (horizontal & vertical)
        merges = 0
        for r in range(4):
            for c in range(3):
                if board[r][c] > 0 and board[r][c] == board[r][c + 1]:
                    merges += 1
        for r in range(3):
            for c in range(4):
                if board[r][c] > 0 and board[r][c] == board[r + 1][c]:
                    merges += 1

        # Monotonicity gradient across rows and columns
        monotonicity = 0
        for r in range(4):
            for c in range(3):
                if board[r][c] >= board[r][c + 1]:
                    monotonicity += 1
        for c in range(4):
            for r in range(3):
                if board[r][c] >= board[r + 1][c]:
                    monotonicity += 1

        return {
            "empty_cells": empty_cells,
            "max_tile": max_tile,
            "max_tile_pos": (max_c, max_r),
            "is_corner": is_corner,
            "available_merges": merges,
            "monotonicity": monotonicity,
            "score": parsed_state["score"],
        }
