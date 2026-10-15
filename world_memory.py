"""World Memory Module (Object Permanence).

Maintains spatial memory across game turns:
- Grid mapping of explored, confirmed walls, hazards, and free tiles.
- Visit counters for cell exploration decay.
- Empirical action outcome tracking per coordinate (successful moves vs attempts).
- Local neighborhood perception queries in egocentric coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Cell status constants
STATUS_FREE: str = "FREE"
STATUS_WALL: str = "WALL"
STATUS_UNKNOWN: str = "UNKNOWN"
STATUS_HAZARD: str = "HAZARD"
STATUS_LASER: str = "LASER"
STATUS_TRAP: str = "TRAP"

# Cardinal direction vectors (assuming +X is right, +Y is down)
CARDINAL_DELTAS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


class WorldMemory:
    """Persistent spatial memory and empirical action outcome tracking.

    Attributes:
        cells: Mapping of (x, y) coordinates to cell status ('WALL', 'FREE', 'UNKNOWN', etc.).
        visits: Mapping of (x, y) coordinates to entry counts.
        outcomes: Mapping of ((x, y), action) -> [successful_moves, total_attempts].
        recent_failures: Ring buffer of recent failed actions for short-term adaptation.
    """

    def __init__(self, max_failure_history: int = 10) -> None:
        """Initialize empty spatial and outcome memory.

        Args:
            max_failure_history: Maximum number of recent failed actions to retain.
        """
        self.cells: Dict[Tuple[int, int], str] = {}
        self.visits: Dict[Tuple[int, int], int] = {}
        self.outcomes: Dict[Tuple[Tuple[int, int], str], List[int]] = {}
        self.recent_failures: List[Dict[str, Any]] = []
        self._max_failure_history: int = max_failure_history

    # ──────────────────────────────────────────────────────────────
    # Position normalizer
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_pos(pos: Any) -> Tuple[int, int]:
        """Convert scalar (1D) or list/tuple (2D) positions to (x, y) tuples."""
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        if isinstance(pos, (int, float)):
            return int(pos), 0
        return 0, 0

    # ──────────────────────────────────────────────────────────────
    # State observation & update
    # ──────────────────────────────────────────────────────────────
    def observe(
        self,
        before_state: Dict[str, Any],
        action: str,
        after_state: Dict[str, Any],
    ) -> None:
        """Update spatial map, visit counts, and empirical action outcomes.

        Args:
            before_state: State dictionary before the action was taken.
            action: Action executed by the agent (e.g., 'up', 'down', 'left', 'right', 'sprint').
            after_state: State dictionary after the action completed.
        """
        before_pos = self._normalize_pos(before_state.get("player_pos", (0, 0)))
        after_pos = self._normalize_pos(after_state.get("player_pos", before_pos))
        action_norm = action.strip().lower()

        # The cell the agent is currently occupying is confirmed FREE
        self.mark_cell(after_pos[0], after_pos[1], STATUS_FREE)
        self.visits[after_pos] = self.visits.get(after_pos, 0) + 1

        # Track outcome key: ((x, y), action)
        outcome_key = (before_pos, action_norm)
        if outcome_key not in self.outcomes:
            self.outcomes[outcome_key] = [0, 0]  # [successes, attempts]

        self.outcomes[outcome_key][1] += 1

        # Check for movement failure (position unchanged after directional move)
        cardinal_dir = action_norm.upper()
        is_directional = cardinal_dir in CARDINAL_DELTAS or action_norm == "sprint"

        if is_directional:
            if after_pos == before_pos:
                # Failed to move: identify intended target cell and mark as WALL
                delta = CARDINAL_DELTAS.get(cardinal_dir)
                if delta is None and action_norm == "sprint":
                    # Sprint uses current facing direction
                    facing = str(before_state.get("facing", "right")).upper()
                    delta = CARDINAL_DELTAS.get(facing, (1, 0))

                if delta is not None:
                    target_x = before_pos[0] + delta[0]
                    target_y = before_pos[1] + delta[1]
                    self.mark_cell(target_x, target_y, STATUS_WALL)

                # Record failure into recent failure memory
                failure_entry = {
                    "pos": before_pos,
                    "action": action_norm,
                    "reason": "Position unchanged (hit obstacle / wall)",
                    "step": after_state.get("current_step", before_state.get("current_step", 0)),
                }
                self.recent_failures.append(failure_entry)
                if len(self.recent_failures) > self._max_failure_history:
                    self.recent_failures.pop(0)
            else:
                # Move succeeded
                self.outcomes[outcome_key][0] += 1

        # Explicit hidden wall discovery reported by game environment
        hidden_hit = after_state.get("hidden_wall_discovered")
        if hidden_hit:
            hx, hy = self._normalize_pos(hidden_hit)
            self.mark_cell(hx, hy, STATUS_WALL)
            failure_entry = {
                "pos": before_pos,
                "action": action_norm,
                "reason": f"Discovered hidden wall at ({hx},{hy})",
                "step": after_state.get("current_step", 0),
            }
            self.recent_failures.append(failure_entry)
            if len(self.recent_failures) > self._max_failure_history:
                self.recent_failures.pop(0)

        # Ingest visible grid if exposed in environment state
        visible_grid = after_state.get("visible_grid")
        if isinstance(visible_grid, list) and visible_grid:
            self._ingest_visible_grid(visible_grid)

    def _ingest_visible_grid(self, grid: List[List[str]]) -> None:
        """Scan visible grid tiles to populate static features."""
        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                coord = (x, y)
                if ch == "W":
                    self.cells[coord] = STATUS_WALL
                elif ch in (".", "S", "K", "E"):
                    if coord not in self.cells or self.cells[coord] != STATUS_WALL:
                        self.cells[coord] = STATUS_FREE
                elif ch == "L":
                    self.cells[coord] = STATUS_LASER
                elif ch == "T":
                    self.cells[coord] = STATUS_TRAP

    # ──────────────────────────────────────────────────────────────
    # Cell query and mutation
    # ──────────────────────────────────────────────────────────────
    def mark_cell(self, x: int, y: int, status: str) -> None:
        """Set the known status of a coordinate."""
        self.cells[(x, y)] = status

    def get_cell(self, x: int, y: int) -> str:
        """Retrieve known cell status, defaulting to UNKNOWN."""
        return self.cells.get((x, y), STATUS_UNKNOWN)

    def get_visit_count(self, x: int, y: int) -> int:
        """Get the number of times coordinate (x, y) has been entered."""
        return self.visits.get((x, y), 0)

    def get_success_rate(self, x: int, y: int, action: str) -> Optional[float]:
        """Compute empirical success rate for taking action from (x, y).

        Returns:
            Success float in [0.0, 1.0], or None if no prior attempts exist.
        """
        key = ((x, y), action.strip().lower())
        if key not in self.outcomes:
            return None
        successes, attempts = self.outcomes[key]
        return successes / attempts if attempts > 0 else None

    # ──────────────────────────────────────────────────────────────
    # Egocentric local context
    # ──────────────────────────────────────────────────────────────
    def get_local_context(self, x: int, y: int, facing_direction: str = "right") -> Dict[str, str]:
        """Calculate known cell statuses of immediate 4 neighbors relative to facing direction.

        Facing coordinates:
            UP:    Ahead=(x, y-1), Behind=(x, y+1), Left=(x-1, y), Right=(x+1, y)
            RIGHT: Ahead=(x+1, y), Behind=(x-1, y), Left=(x, y-1), Right=(x, y+1)
            DOWN:  Ahead=(x, y+1), Behind=(x, y-1), Left=(x+1, y), Right=(x-1, y)
            LEFT:  Ahead=(x-1, y), Behind=(x+1, y), Left=(x, y+1), Right=(x, y-1)

        Args:
            x: Current player X coordinate.
            y: Current player Y coordinate.
            facing_direction: Orientation string ('UP', 'DOWN', 'LEFT', 'RIGHT', case-insensitive).

        Returns:
            Dictionary with keys 'AHEAD', 'BEHIND', 'LEFT', 'RIGHT' mapping to cell status strings.
        """
        facing_norm = facing_direction.strip().upper()
        dx, dy = CARDINAL_DELTAS.get(facing_norm, (1, 0))

        # Egocentric vector arithmetic for 2D grid (+X right, +Y down):
        # AHEAD  = (dx, dy)
        # BEHIND = (-dx, -dy)
        # LEFT   = (dy, -dx)
        # RIGHT  = (-dy, dx)
        offsets = {
            "AHEAD": (dx, dy),
            "BEHIND": (-dx, -dy),
            "LEFT": (dy, -dx),
            "RIGHT": (-dy, dx),
        }

        context: Dict[str, str] = {}
        for pov_label, (ox, oy) in offsets.items():
            neighbor_coord = (x + ox, y + oy)
            context[pov_label] = self.get_cell(neighbor_coord[0], neighbor_coord[1])

        return context

    # ──────────────────────────────────────────────────────────────
    # Recent failures query
    # ──────────────────────────────────────────────────────────────
    def get_recent_failures(
        self,
        target_pos: Any = None,
        limit: int = 3,
        pos: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent failed actions, optionally filtered by coordinate.

        Args:
            target_pos: Either a (x, y) coordinate tuple, or an int limit, or None.
            limit: Maximum entries to return (default: 3).
            pos: Optional coordinate tuple if target_pos was omitted.

        Returns:
            List of failure entry dicts.
        """
        filter_pos = None
        effective_limit = limit

        if isinstance(target_pos, (list, tuple)):
            filter_pos = self._normalize_pos(target_pos)
        elif isinstance(target_pos, int):
            effective_limit = target_pos

        if pos is not None:
            filter_pos = self._normalize_pos(pos)

        failures = self.recent_failures
        if filter_pos is not None:
            failures = [f for f in failures if f.get("pos") == filter_pos]

        return failures[-effective_limit:] if failures else []



    def reset(self) -> None:
        """Clear all spatial and outcome memory."""
        self.cells.clear()
        self.visits.clear()
        self.outcomes.clear()
        self.recent_failures.clear()
