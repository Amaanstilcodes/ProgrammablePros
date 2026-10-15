"""Vault Runner Game Module.

A 2-D 8x8 grid heist game where an autonomous bot must collect a KEY
and then escape through the EXIT while avoiding guards, lasers, walls,
and hidden walls.

Implements:
  Strategy 1 — State Compression via compress_vault_state()
  Strategy 2 — Hypothesis questions via goal_phrase parameter (in laya_agent)
  Strategy 3 — Per-action asymmetric payoffs via get_action_risk()
  Strategy 4 — Failure-Adaptive Memory triggered by hidden-wall & blocked EXIT hits
"""

from __future__ import annotations

import random
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# Grid constants
# ──────────────────────────────────────────────────────────────────
GRID_W: int = 8
GRID_H: int = 8
MAX_STEPS: int = 40

# Tile type identifiers (stored in self._layout)
TILE_FLOOR: str = "."
TILE_WALL: str = "W"
TILE_HIDDEN_WALL: str = "H"   # NOT exposed in state — discovered on collision
TILE_LASER: str = "L"
TILE_TRAP: str = "T"          # Trap floor: one-time -8 penalty then becomes floor
TILE_KEY: str = "K"
TILE_EXIT: str = "E"
TILE_START: str = "S"
TILE_GUARD: str = "G"         # Guard spawn — placed in layout; actual guard tracked separately

# Actions
ACTION_UP: str = "up"
ACTION_DOWN: str = "down"
ACTION_LEFT: str = "left"
ACTION_RIGHT: str = "right"
ACTION_WAIT: str = "wait"
ACTION_SPRINT: str = "sprint"

ALL_ACTIONS: List[str] = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT, ACTION_SPRINT]

# Direction vectors
DIRECTION_DELTA: Dict[str, Tuple[int, int]] = {
    ACTION_UP:    (0, -1),
    ACTION_DOWN:  (0,  1),
    ACTION_LEFT:  (-1, 0),
    ACTION_RIGHT: (1,  0),
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _clamp(v: int, lo: int, hi: int) -> int:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def _bfs_reachable(
    layout: List[List[str]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    via: Optional[Tuple[int, int]] = None,
) -> bool:
    """BFS reachability check ignoring HIDDEN_WALLs and guards.

    Args:
        layout: 2-D grid list-of-lists.
        start: (x, y) origin.
        goal:  (x, y) destination.
        via:   Optional waypoint that must also be reachable before goal.

    Returns:
        True if start -> (via ->) goal is reachable.
    """
    passable = {TILE_FLOOR, TILE_HIDDEN_WALL, TILE_LASER, TILE_TRAP, TILE_KEY, TILE_EXIT, TILE_START, TILE_GUARD}

    def _reach(src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        visited: set = {src}
        queue: deque = deque([src])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == dst:
                return True
            for dx, dy in DIRECTION_DELTA.values():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if (nx, ny) not in visited and layout[ny][nx] in passable:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    if via is not None:
        return _reach(start, via) and _reach(via, goal)
    return _reach(start, goal)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ──────────────────────────────────────────────────────────────────
# Guard class
# ──────────────────────────────────────────────────────────────────

class _Guard:
    """A guard that patrols back-and-forth along a fixed axis."""

    def __init__(
        self,
        pos: Tuple[int, int],
        patrol: List[Tuple[int, int]],
    ) -> None:
        self.pos: Tuple[int, int] = pos
        self.patrol: List[Tuple[int, int]] = patrol
        self._patrol_idx: int = 0
        self._direction: int = 1   # +1 forward, -1 backward along patrol list

    def step(self) -> Tuple[int, int]:
        """Advance guard one tile along its patrol route and return new pos."""
        self._patrol_idx += self._direction
        if self._patrol_idx >= len(self.patrol):
            self._patrol_idx = len(self.patrol) - 2
            self._direction = -1
        elif self._patrol_idx < 0:
            self._patrol_idx = 1
            self._direction = 1
        self.pos = self.patrol[self._patrol_idx]
        return self.pos

    @property
    def next_pos(self) -> Tuple[int, int]:
        """Peek at the guard's next position without moving."""
        ni = self._patrol_idx + self._direction
        ni = _clamp(ni, 0, len(self.patrol) - 1)
        return self.patrol[ni]

    @property
    def direction_label(self) -> str:
        """Human-readable patrol direction for state compression."""
        dp = self.patrol[_clamp(self._patrol_idx + self._direction, 0, len(self.patrol) - 1)]
        dx = dp[0] - self.pos[0]
        dy = dp[1] - self.pos[1]
        if dx > 0:
            return "RIGHT"
        if dx < 0:
            return "LEFT"
        if dy > 0:
            return "DOWN"
        return "UP"


# ──────────────────────────────────────────────────────────────────
# VaultRunnerGame
# ──────────────────────────────────────────────────────────────────

class VaultRunnerGame:
    """2-D 8x8 grid heist game environment.

    Interface mirrors MockGame:
      reset() -> dict
      get_state() -> dict        (noisy raw payload)
      get_available_actions() -> list[str]
      step(action) -> dict
      get_action_risk(state, action) -> (reward, penalty)
      is_done() -> bool
      render_ascii() -> str

    Strategy 1  -- compress_vault_state() strips raw payload to ~120-char string.
    Strategy 3  -- get_action_risk() returns asymmetric (reward, penalty) per action.
    Strategy 4  -- step() sets 'progress_made=False' on hidden-wall & blocked-EXIT hits
                  so the game-loop records a failure and injects a WARNING next turn.
    """

    # Fixed-layout tile map (regenerated if BFS fails)
    _DEFAULT_LAYOUT: List[str] = [
        "S . . . W . . .",   # row 0 (y=0): S=start at (0,0), wall at (4,0)
        ". W . . . . W .",   # row 1
        ". . . H . . . .",   # row 2: H=hidden wall at (3,2)
        ". . W . . L . .",   # row 3: laser at (5,3)
        ". . . . K . . .",   # row 4: KEY at (4,4)
        ". H . . . . W .",   # row 5: hidden wall at (1,5)
        ". . . T . . . .",   # row 6: trap at (3,6)
        ". . . . . . . E",   # row 7: EXIT at (7,7)
    ]

    def __init__(self, seed: int = 42) -> None:
        """Initialise the game with a reproducible seed.

        Args:
            seed: Random seed for reproducible layouts and guard patrol.
        """
        self._seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self._layout: List[List[str]] = []
        self._guard: Optional[_Guard] = None
        self._guard_patrol: List[Tuple[int, int]] = []
        self._triggered_traps: set = set()

        # Game state fields
        self.player_pos: Tuple[int, int] = (0, 0)
        self.facing: str = ACTION_RIGHT
        self.keys_held: int = 0
        self.steps_used: int = 0
        self.total_penalty: float = 0.0
        self.is_game_over: bool = False
        self.result: str = "IN_PROGRESS"
        self.hidden_wall_discovered: Optional[Tuple[int, int]] = None

        self._session_id: str = str(uuid.uuid4())[:8]

        self._build_layout()
        self.reset()

    # ── Layout construction ────────────────────────────────────────

    def _build_layout(self) -> None:
        """Parse the default layout and verify BFS solvability."""
        for _attempt in range(10):
            layout = [row.split() for row in self._DEFAULT_LAYOUT]
            start = self._find_tile(layout, TILE_START)
            key_pos = self._find_tile(layout, TILE_KEY)
            exit_pos = self._find_tile(layout, TILE_EXIT)
            if start and key_pos and exit_pos:
                if _bfs_reachable(layout, start, exit_pos, via=key_pos):
                    self._layout = layout
                    self._guard_patrol = [(x, 3) for x in range(4)]
                    return
            # Fallback: trivially solvable open layout
            self._layout = [[TILE_FLOOR] * GRID_W for _ in range(GRID_H)]
            self._layout[0][0] = TILE_START
            self._layout[4][4] = TILE_KEY
            self._layout[7][7] = TILE_EXIT
            self._guard_patrol = [(0, 3), (1, 3), (2, 3), (3, 3)]
        assert self._layout, "Could not build a solvable layout after 10 attempts."

    @staticmethod
    def _find_tile(layout: List[List[str]], tile: str) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(layout):
            for x, t in enumerate(row):
                if t == tile:
                    return (x, y)
        return None

    # ── Public interface ───────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Reset game to initial state and return the raw state dict."""
        self.player_pos = self._find_tile(self._layout, TILE_START) or (0, 0)
        self.facing = ACTION_RIGHT
        self.keys_held = 0
        self.steps_used = 0
        self.total_penalty = 0.0
        self.is_game_over = False
        self.result = "IN_PROGRESS"
        self.hidden_wall_discovered = None
        self._triggered_traps = set()

        guard_start = self._guard_patrol[0]
        self._guard = _Guard(guard_start, list(self._guard_patrol))
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Return noisy raw server-style payload (Strategy 1 input).

        Includes metadata noise (session_id, styling strings, full grid, guard info)
        to mimic real API responses -- compress_vault_state() strips all of this.
        """
        gx, gy = self._guard.pos if self._guard else (-1, -1)
        gnx, gny = self._guard.next_pos if self._guard else (-1, -1)

        # Build visible grid (hide H tiles, show triggered traps as floor)
        visible_grid: List[List[str]] = []
        for y, row in enumerate(self._layout):
            visible_row: List[str] = []
            for x, tile in enumerate(row):
                if tile == TILE_HIDDEN_WALL:
                    visible_row.append(TILE_FLOOR)
                elif (x, y) in self._triggered_traps:
                    visible_row.append(TILE_FLOOR)
                else:
                    visible_row.append(tile)
            visible_grid.append(visible_row)

        key_pos = self._find_tile(self._layout, TILE_KEY)

        return {
            "player_pos": self.player_pos,
            "goal_pos": self._find_tile(self._layout, TILE_EXIT),
            "key_pos": key_pos if self.keys_held == 0 else None,
            "keys_held": self.keys_held,
            "facing": self.facing,
            "steps_used": self.steps_used,
            "steps_remaining": MAX_STEPS - self.steps_used,
            "max_steps": MAX_STEPS,
            "current_step": self.steps_used,
            "is_game_over": self.is_game_over,
            "status": self.result,
            "total_penalty": self.total_penalty,
            "available_actions": self.get_available_actions(),
            "hidden_wall_discovered": self.hidden_wall_discovered,
            "guard_pos": (gx, gy),
            "guard_next_pos": (gnx, gny),
            "guard_direction": self._guard.direction_label if self._guard else "NONE",
            # Noisy metadata (stripped by compressor)
            "session_id": self._session_id,
            "game_type": "VAULT_RUNNER",
            "render_style": "DARK_NEON",
            "grid_width": GRID_W,
            "grid_height": GRID_H,
            "visible_grid": visible_grid,
        }

    def get_available_actions(self) -> List[str]:
        """Return full action list (all actions always nominally available)."""
        return list(ALL_ACTIONS)

    def step(self, action: str) -> Dict[str, Any]:
        """Execute one action and return the new state.

        Strategy 4: Sets progress_made=False for hidden-wall collisions and
        blocked EXIT attempts, signalling the game loop to inject a WARNING.

        Args:
            action: One of ALL_ACTIONS.

        Returns:
            Updated state dict with extra fields: progress_made, penalty_incurred, done, result.
        """
        if self.is_game_over:
            return self.get_state()

        self.steps_used += 1
        self.hidden_wall_discovered = None
        action_lc = action.strip().lower()

        penalty_this_step: float = 0.0
        progress_made: bool = False

        if action_lc == ACTION_WAIT:
            penalty_this_step = -1.0
            self.total_penalty += penalty_this_step
            progress_made = True   # intentional wait -- should not trigger warning

        elif action_lc == ACTION_SPRINT:
            penalty_this_step, progress_made = self._do_sprint()
            self.total_penalty += penalty_this_step

        elif action_lc in DIRECTION_DELTA:
            self.facing = action_lc
            penalty_this_step, progress_made = self._do_move(action_lc)
            self.total_penalty += penalty_this_step

        else:
            progress_made = False

        # After player moves, advance guard and check capture
        if not self.is_game_over and self._guard:
            self._guard.step()
            if self._guard.pos == self.player_pos:
                penalty_this_step -= 30.0
                self.total_penalty -= 30.0
                self.is_game_over = True
                self.result = "CAUGHT"
                progress_made = False

        # Check timeout
        if not self.is_game_over and self.steps_used >= MAX_STEPS:
            self.is_game_over = True
            self.result = "TIMEOUT"

        state = self.get_state()
        state["progress_made"] = progress_made
        state["penalty_incurred"] = penalty_this_step
        state["done"] = self.is_game_over
        state["result"] = self.result
        return state

    def _do_move(self, direction: str) -> Tuple[float, bool]:
        """Attempt a single-step cardinal move.

        Returns:
            (penalty, progress_made)
        """
        dx, dy = DIRECTION_DELTA[direction]
        px, py = self.player_pos
        nx, ny = px + dx, py + dy

        if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
            return 0.0, False

        tile = self._layout[ny][nx]

        if tile == TILE_WALL:
            return 0.0, False

        if tile == TILE_HIDDEN_WALL:
            # Strategy 4: hidden-wall discovery triggers WARNING next turn
            self.hidden_wall_discovered = (nx, ny)
            return 0.0, False

        self.player_pos = (nx, ny)
        penalty = 0.0

        if tile == TILE_LASER:
            penalty = -15.0

        elif tile == TILE_TRAP and (nx, ny) not in self._triggered_traps:
            penalty = -8.0
            self._triggered_traps.add((nx, ny))
            self._layout[ny][nx] = TILE_FLOOR

        elif tile == TILE_KEY and self.keys_held == 0:
            self.keys_held = 1
            self._layout[ny][nx] = TILE_FLOOR

        elif tile == TILE_EXIT:
            if self.keys_held >= 1:
                self.is_game_over = True
                self.result = "WON"
            else:
                # Blocked -- no progress, triggers failure warning (Strategy 4)
                self.player_pos = (px, py)
                return 0.0, False

        return penalty, True

    def _do_sprint(self) -> Tuple[float, bool]:
        """Execute SPRINT (2 tiles in facing direction).

        Strategy 3: SPRINT has higher P(progress) in open lanes but a worse
        downside (-10 extra penalty + 1 step wasted) when the second tile is
        a wall/hidden-wall -- creating the asymmetric-penalty EV showcase.

        Returns:
            (penalty, progress_made)
        """
        direction = self.facing
        if direction not in DIRECTION_DELTA:
            return 0.0, False

        dx, dy = DIRECTION_DELTA[direction]
        px, py = self.player_pos
        total_penalty: float = 0.0
        moved: bool = False

        for hop in range(1, 3):
            nx, ny = px + dx * hop, py + dy * hop

            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                break

            tile = self._layout[ny][nx]

            if tile == TILE_WALL:
                if hop == 2:
                    total_penalty -= 10.0  # crash penalty (Strategy 3)
                break

            if tile == TILE_HIDDEN_WALL:
                self.hidden_wall_discovered = (nx, ny)
                if hop == 2:
                    total_penalty -= 10.0
                break

            # LASER: sprint jumps over first laser safely
            if tile == TILE_LASER:
                if hop == 1:
                    continue  # jump over
                else:
                    total_penalty -= 15.0

            if tile == TILE_TRAP and (nx, ny) not in self._triggered_traps:
                total_penalty -= 8.0
                self._triggered_traps.add((nx, ny))
                self._layout[ny][nx] = TILE_FLOOR

            if tile == TILE_KEY and self.keys_held == 0:
                self.keys_held = 1
                self._layout[ny][nx] = TILE_FLOOR

            if tile == TILE_EXIT:
                if self.keys_held >= 1:
                    self.player_pos = (nx, ny)
                    self.is_game_over = True
                    self.result = "WON"
                    return total_penalty, True
                else:
                    break

            self.player_pos = (nx, ny)
            moved = True

        return total_penalty, moved

    def _objective_distance(self) -> int:
        """Manhattan distance to current objective (KEY or EXIT)."""
        if self.keys_held == 0:
            obj = self._find_tile(self._layout, TILE_KEY)
        else:
            obj = self._find_tile(self._layout, TILE_EXIT)
        if obj is None:
            return 0
        return _manhattan(self.player_pos, obj)

    def get_action_risk(
        self, state: Dict[str, Any], action: str
    ) -> Tuple[float, float]:
        """Return (reward_if_success, penalty_if_failure) for an action.

        Strategy 3: Provides per-action asymmetric payoffs so argmax(P) and
        argmax(EV) can diverge (e.g., SPRINT near a crash risk).

        Args:
            state: Current game state dict.
            action: One of ALL_ACTIONS.

        Returns:
            Tuple of (reward, abs_penalty) -- penalty is POSITIVE (caller negates).
        """
        action_lc = action.strip().lower()
        px, py = self.player_pos
        gnext = self._guard.next_pos if self._guard else (-1, -1)
        gpos = self._guard.pos if self._guard else (-1, -1)

        base_reward = 10.0
        base_penalty = 5.0

        if action_lc == ACTION_WAIT:
            return 2.0, 1.0

        if action_lc == ACTION_SPRINT:
            direction = self.facing
            if direction in DIRECTION_DELTA:
                dx, dy = DIRECTION_DELTA[direction]
                t2x, t2y = px + dx * 2, py + dy * 2
                t1x, t1y = px + dx, py + dy
                # Laser at first tile: sprint jumps it -- safe & fast
                if 0 <= t1x < GRID_W and 0 <= t1y < GRID_H:
                    if self._layout[t1y][t1x] == TILE_LASER:
                        return 12.0, 5.0
                # Wall at second tile: crash risk (Strategy 3 asymmetric case)
                if 0 <= t2x < GRID_W and 0 <= t2y < GRID_H:
                    if self._layout[t2y][t2x] in (TILE_WALL, TILE_HIDDEN_WALL):
                        return 8.0, 15.0
            return 12.0, 8.0  # open sprint

        if action_lc in DIRECTION_DELTA:
            dx, dy = DIRECTION_DELTA[action_lc]
            nx, ny = px + dx, py + dy

            # Adjacent to guard's current or next position -- capture risk
            if (nx, ny) == gnext or (nx, ny) == gpos:
                return 10.0, 30.0

            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if self._layout[ny][nx] == TILE_LASER:
                    return 10.0, 15.0

            return base_reward, base_penalty

        return base_reward, base_penalty

    def is_done(self) -> bool:
        """Return True if game has ended."""
        return self.is_game_over

    def render_ascii(self) -> str:
        """Render an ASCII board for console display."""
        px, py = self.player_pos
        gx, gy = self._guard.pos if self._guard else (-1, -1)

        lines = [f"  Vault Runner -- Step {self.steps_used}/{MAX_STEPS}  Keys={self.keys_held}  Penalty={self.total_penalty:.0f}"]
        lines.append("  " + "-" * (GRID_W * 2 + 1))
        for y in range(GRID_H):
            row_str = "  |"
            for x in range(GRID_W):
                if (x, y) == (px, py):
                    ch = "P"
                elif (x, y) == (gx, gy):
                    ch = "G"
                elif self._layout[y][x] == TILE_HIDDEN_WALL:
                    ch = "."   # hidden
                elif (x, y) in self._triggered_traps:
                    ch = "."
                else:
                    ch = self._layout[y][x]
                row_str += ch + " "
            lines.append(row_str + "|")
        lines.append("  " + "-" * (GRID_W * 2 + 1))
        lines.append("  Legend: P=player G=guard K=key E=exit W=wall L=laser T=trap")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Strategy 1 -- State Compressor
# ──────────────────────────────────────────────────────────────────

def compress_vault_state(raw_state: Dict[str, Any]) -> str:
    """Compress noisy raw VaultRunner state to <=120-char dense prompt string.

    Strategy 1: Strips all metadata. Pre-computes distances and adjacency flags.
    Never leaks hidden-wall locations.

    Args:
        raw_state: Full state dict from VaultRunnerGame.get_state().

    Returns:
        Dense single-line string of vital features only.
    """
    px, py = raw_state.get("player_pos", (0, 0))
    gpos = raw_state.get("guard_pos", (-1, -1))
    gx, gy = gpos
    facing = raw_state.get("facing", "right").upper()
    keys_held = raw_state.get("keys_held", 0)
    steps_left = raw_state.get("steps_remaining", MAX_STEPS)
    guard_dir = raw_state.get("guard_direction", "NONE")

    if keys_held == 0:
        obj_pos = raw_state.get("key_pos")
        objective = "KEY"
    else:
        obj_pos = raw_state.get("goal_pos")
        objective = "EXIT"

    obj_dist = _manhattan((px, py), obj_pos) if obj_pos else 0
    guard_dist = _manhattan((px, py), (gx, gy)) if gx >= 0 else 99

    # Laser adjacency from visible grid (hidden walls NOT in visible grid)
    laser_adj_dirs: List[str] = []
    vgrid = raw_state.get("visible_grid", [])
    if vgrid:
        for label, (dx, dy) in [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if vgrid[ny][nx] == TILE_LASER:
                    laser_adj_dirs.append(label)

    laser_info = ("laser_adj=" + ",".join(laser_adj_dirs)) if laser_adj_dirs else "laser_adj=NONE"

    compressed = (
        f"pos=({px},{py}) obj={objective} obj_dist={obj_dist} keys={keys_held} "
        f"steps_left={steps_left} {laser_info} guard_dist={guard_dist} "
        f"guard_dir={guard_dir} facing={facing}"
    )

    # Enforce <=120 char limit
    if len(compressed) > 120:
        compressed = compressed[:117] + "..."
    return compressed
