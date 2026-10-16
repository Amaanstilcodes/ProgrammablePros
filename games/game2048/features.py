"""2048 Feature Extraction — Genesis 2.0 Track EVA 02 / Phase 2b.

Produces Board2048Features dataclass and deterministic compact context string (<= 400 chars).
Contains NO search, NO minimax, NO expectimax, NO lookahead.
Isolates legality check in `is_board_changing(board, action)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
DESIGNATED_CORNER: tuple[int, int] = (3, 0)  # Bottom-Left
ALL_CORNERS: set[tuple[int, int]] = {(0, 0), (0, 3), (3, 0), (3, 3)}

# Snake-shaped path anchored at bottom-left (3,0)
SNAKE_PATH: tuple[tuple[int, int], ...] = (
    (3, 0), (3, 1), (3, 2), (3, 3),
    (2, 3), (2, 2), (2, 1), (2, 0),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (0, 3), (0, 2), (0, 1), (0, 0),
)


@dataclass(frozen=True)
class Board2048Features:
    """Rich feature representation of current 2048 board and memory."""

    board: tuple[tuple[int, ...], ...]
    empty_count: int
    max_tile: int
    max_tile_log2: int
    max_tile_positions: list[tuple[int, int]]
    max_tile_in_corner: bool
    max_tile_at_designated_corner: bool

    # Monotonicity & topology (current board only)
    row_monotonicity: float
    col_monotonicity: float
    corner_chain_score: float
    smoothness: float
    merge_pairs: dict[str, int]

    # Per-direction local geometric facts
    edge_occupancy: dict[str, int]
    empty_on_edge: dict[str, int]
    max_tile_distance_to_corner: dict[str, int]

    # Memory-derived
    empty_count_trend: int
    rejection_history: list[tuple[str, int]]
    recent_actions: tuple[str, ...]
    stall_turns: int
    warning_text: str

    def __getitem__(self, key: str) -> Any:
        """Backward-compatibility mapping for existing tests and viewers."""
        if hasattr(self, key):
            return getattr(self, key)
        if key == "is_corner":
            return self.max_tile_in_corner
        if key == "empty_cells":
            return self.empty_count
        if key == "available_merges":
            return sum(self.merge_pairs.values())
        if key == "best_tile":
            return self.max_tile
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def is_board_changing(board: tuple[tuple[int, ...], ...], action: str) -> bool:
    """Legality check: does sliding in this direction change the board?

    Isolates single-step slide simulation for legality ONLY (no search, no lookahead).
    """
    act = action.strip().lower()
    if act not in DIRECTIONS:
        return False

    def _line_moves(line: list[int]) -> bool:
        filtered = [x for x in line if x != 0]
        # Check merges
        for i in range(len(filtered) - 1):
            if filtered[i] == filtered[i + 1]:
                return True
        # Check shifting into zeros
        if filtered != [x for x in line[:len(filtered)] if x != 0]:
            return True
        return False

    if act == "left":
        return any(_line_moves(list(row)) for row in board)
    elif act == "right":
        return any(_line_moves(list(reversed(row))) for row in board)
    elif act == "up":
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            if _line_moves(col):
                return True
        return False
    elif act == "down":
        for c in range(4):
            col = [board[r][c] for r in reversed(range(4))]
            if _line_moves(col):
                return True
        return False
    return False


def compute_corner_chain_score(board: tuple[tuple[int, ...], ...]) -> float:
    """Calculate fraction of non-increasing adjacent pairs along bottom-left snake path."""
    values = [board[r][c] for (r, c) in SNAKE_PATH if board[r][c] > 0]
    if len(values) <= 1:
        return 1.0

    non_increasing = sum(1 for i in range(len(values) - 1) if values[i] >= values[i + 1])
    return round(non_increasing / (len(values) - 1), 4)


def compute_monotonicity(board: tuple[tuple[int, ...], ...]) -> tuple[float, float]:
    """Calculate row and column monotonicity scores in [0.0, 1.0]."""
    # Rows
    row_scores: list[float] = []
    for r in range(4):
        row = [board[r][c] for c in range(4) if board[r][c] > 0]
        if len(row) <= 1:
            row_scores.append(1.0)
            continue
        inc = sum(1 for i in range(len(row) - 1) if row[i] <= row[i + 1])
        dec = sum(1 for i in range(len(row) - 1) if row[i] >= row[i + 1])
        row_scores.append(max(inc, dec) / (len(row) - 1))
    row_mono = round(sum(row_scores) / 4.0, 4)

    # Columns
    col_scores: list[float] = []
    for c in range(4):
        col = [board[r][c] for r in range(4) if board[r][c] > 0]
        if len(col) <= 1:
            col_scores.append(1.0)
            continue
        inc = sum(1 for i in range(len(col) - 1) if col[i] <= col[i + 1])
        dec = sum(1 for i in range(len(col) - 1) if col[i] >= col[i + 1])
        col_scores.append(max(inc, dec) / (len(col) - 1))
    col_mono = round(sum(col_scores) / 4.0, 4)

    return row_mono, col_mono


def compute_smoothness(board: tuple[tuple[int, ...], ...]) -> float:
    """Calculate smoothness score in [0.0, 1.0] from log2 differences of adjacent tiles."""
    total_diff = 0.0
    pair_count = 0

    for r in range(4):
        for c in range(4):
            val = board[r][c]
            if val == 0:
                continue
            log_val = math.log2(val)
            # Right neighbor
            if c < 3 and board[r][c + 1] > 0:
                total_diff += abs(log_val - math.log2(board[r][c + 1]))
                pair_count += 1
            # Down neighbor
            if r < 3 and board[r + 1][c] > 0:
                total_diff += abs(log_val - math.log2(board[r + 1][c]))
                pair_count += 1

    if pair_count == 0:
        return 1.0

    avg_diff = total_diff / pair_count
    # Map avg_diff to [0, 1]: 0 diff -> 1.0; 4+ diff -> 0.0
    smooth = max(0.0, min(1.0, 1.0 - (avg_diff / 4.0)))
    return round(smooth, 4)


def compute_merge_pairs(board: tuple[tuple[int, ...], ...]) -> dict[str, int]:
    """Count adjacent equal non-zero pairs that would merge in each direction."""
    # Horizontal merges
    h_merges = 0
    for r in range(4):
        non_zero = [board[r][c] for c in range(4) if board[r][c] > 0]
        for i in range(len(non_zero) - 1):
            if non_zero[i] == non_zero[i + 1]:
                h_merges += 1

    # Vertical merges
    v_merges = 0
    for c in range(4):
        non_zero = [board[r][c] for r in range(4) if board[r][c] > 0]
        for i in range(len(non_zero) - 1):
            if non_zero[i] == non_zero[i + 1]:
                v_merges += 1

    return {
        "up": v_merges,
        "down": v_merges,
        "left": h_merges,
        "right": h_merges,
    }


class Game2048FeatureExtractor:
    """Extracts Board2048Features from parsed state view and memory."""

    def extract_features(
        self,
        state_view: Any,
        memory: Any = None,
    ) -> Board2048Features:
        """Extract all required Phase 2b features."""
        # Extract board
        board_raw = getattr(state_view, "board", None)
        if board_raw is None and isinstance(state_view, dict):
            board_raw = state_view.get("board") or state_view.get("grid") or [[0] * 4 for _ in range(4)]

        board_rows: list[tuple[int, ...]] = []
        for r in range(4):
            row: list[int] = []
            for c in range(4):
                val = 0
                if r < len(board_raw) and c < len(board_raw[r]):
                    val = int(board_raw[r][c])
                row.append(val)
            board_rows.append(tuple(row))
        board: tuple[tuple[int, ...], ...] = tuple(board_rows)

        # Basic statistics
        empty_count = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
        max_tile = max(max(row) for row in board) if board else 0
        max_tile_log2 = int(math.log2(max_tile)) if max_tile > 1 else 0

        max_tile_positions = [(r, c) for r in range(4) for c in range(4) if board[r][c] == max_tile]
        max_tile_in_corner = any(pos in ALL_CORNERS for pos in max_tile_positions)
        max_tile_at_designated_corner = DESIGNATED_CORNER in max_tile_positions

        # Monotonicity & chain
        row_mono, col_mono = compute_monotonicity(board)
        chain_score = compute_corner_chain_score(board)
        smoothness = compute_smoothness(board)
        merge_pairs = compute_merge_pairs(board)

        # Edges
        # up: row 0, down: row 3, left: col 0, right: col 3
        edge_occupancy = {
            "up": sum(1 for c in range(4) if board[0][c] > 0),
            "down": sum(1 for c in range(4) if board[3][c] > 0),
            "left": sum(1 for r in range(4) if board[r][0] > 0),
            "right": sum(1 for r in range(4) if board[r][3] > 0),
        }
        empty_on_edge = {
            "up": 4 - edge_occupancy["up"],
            "down": 4 - edge_occupancy["down"],
            "left": 4 - edge_occupancy["left"],
            "right": 4 - edge_occupancy["right"],
        }

        # Max tile distance to designated corner (3, 0) after slide axis consideration:
        # Designated corner: row=3, col=0
        # If max tile is at (r, c):
        # slide up -> moves row towards 0 -> dist to row 3 is abs(0 - 3) = 3
        # slide down -> moves row towards 3 -> dist to row 3 is abs(3 - 3) = 0
        # slide left -> moves col towards 0 -> dist to col 0 is abs(0 - 0) = 0
        # slide right -> moves col towards 3 -> dist to col 0 is abs(3 - 0) = 3
        max_tile_distance_to_corner = {
            "up": 3,
            "down": 0,
            "left": 0,
            "right": 3,
        }

        # Memory derived
        empty_count_trend = 0
        rejection_history: list[tuple[str, int]] = []
        recent_actions: tuple[str, ...] = ()
        stall_turns = 0
        warning_text = ""

        if memory is not None:
            if hasattr(memory, "empty_count_history") and len(memory.empty_count_history) >= 3:
                empty_count_trend = empty_count - memory.empty_count_history[-3]
            if hasattr(memory, "rejection_history"):
                rejection_history = list(memory.rejection_history)
            if hasattr(memory, "recent_actions"):
                recent_actions = tuple(memory.recent_actions)
            if hasattr(memory, "stall_turns"):
                stall_turns = memory.stall_turns
            if hasattr(memory, "warning_text"):
                warning_text = memory.warning_text()

        return Board2048Features(
            board=board,
            empty_count=empty_count,
            max_tile=max_tile,
            max_tile_log2=max_tile_log2,
            max_tile_positions=max_tile_positions,
            max_tile_in_corner=max_tile_in_corner,
            max_tile_at_designated_corner=max_tile_at_designated_corner,
            row_monotonicity=row_mono,
            col_monotonicity=col_mono,
            corner_chain_score=chain_score,
            smoothness=smoothness,
            merge_pairs=merge_pairs,
            edge_occupancy=edge_occupancy,
            empty_on_edge=empty_on_edge,
            max_tile_distance_to_corner=max_tile_distance_to_corner,
            empty_count_trend=empty_count_trend,
            rejection_history=rejection_history,
            recent_actions=recent_actions,
            stall_turns=stall_turns,
            warning_text=warning_text,
        )

    def extract(self, view: Any, memory: Any = None) -> Board2048Features:
        return self.extract_features(view, memory)

    def to_context_string(self, features: Board2048Features, subgoal_name: str = "KEEP_CORNER") -> str:
        return format_context_string(features, subgoal_name)


def format_context_string(features: Board2048Features, subgoal_name: str = "KEEP_CORNER") -> str:
    """Format features and active subgoal into deterministic string (target <= 300, max 400 chars)."""
    max_pos_str = f"({features.max_tile_positions[0][0]},{features.max_tile_positions[0][1]})" if features.max_tile_positions else "(3,0)"
    corner_bit = 1 if features.max_tile_at_designated_corner else 0

    merges_str = f"merges=[U{features.merge_pairs.get('up', 0)},D{features.merge_pairs.get('down', 0)},L{features.merge_pairs.get('left', 0)},R{features.merge_pairs.get('right', 0)}]"
    edge_str = f"edge=[U{features.edge_occupancy.get('up', 0)},D{features.edge_occupancy.get('down', 0)},L{features.edge_occupancy.get('left', 0)},R{features.edge_occupancy.get('right', 0)}]"
    maxdist_str = f"maxdist=[U{features.max_tile_distance_to_corner.get('up', 0)},D{features.max_tile_distance_to_corner.get('down', 0)},L{features.max_tile_distance_to_corner.get('left', 0)},R{features.max_tile_distance_to_corner.get('right', 0)}]"

    tokens = [
        "game=2048",
        f"max={features.max_tile}",
        f"maxlog={features.max_tile_log2}",
        f"maxpos={max_pos_str}",
        f"corner={corner_bit}",
        f"chain={features.corner_chain_score:.2f}",
        f"smooth={features.smoothness:.2f}",
        f"empty={features.empty_count}",
        f"rowmono={features.row_monotonicity:.2f}",
        f"colmono={features.col_monotonicity:.2f}",
        merges_str,
        edge_str,
        maxdist_str,
        f"emptytrend={features.empty_count_trend}",
        f"stall={features.stall_turns}",
        f"subgoal={subgoal_name}",
    ]

    if features.warning_text:
        tokens.append(features.warning_text)

    ctx = " ".join(tokens)
    if len(ctx) > 400:
        ctx = ctx[:400]
    return ctx
